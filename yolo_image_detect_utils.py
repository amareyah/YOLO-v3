import torch
import torch.nn as nn
import numpy as np


class SequentialCustom(nn.Sequential):
    def __init__(self):
        super().__init__()

    def extra_repr(self):
        conv = ''
        has_bn = 'n'
        act = 'n'

        for module in self._modules.values():
            if isinstance(module, nn.Conv2d):
                conv = "conv2d: in_chan {}; out_chan{}; kern_size {}; strd {}; pad {};". \
                    format(module.in_channels, module.out_channels,
                           module.kernel_size, module.stride, module.padding)
            elif isinstance(module, nn.BatchNorm2d):
                has_bn = 'y'
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
                act = str(module)[0:str(module).index('(')]

        desc = conv + " | BN2d {}".format(has_bn) + " | actv. {}".format(act)
        return desc

    def forward(self, x):
        return super().forward(x)


class Route(nn.Module):
    def __init__(self, ind_arr, learning_params=False, fitting_mode='reshape'):
        super().__init__()
        self.ind_arr = ind_arr
        self.layer_outputs_arr = None
        self.learning_params = learning_params
        self.mode = fitting_mode

    def _fit_feature_map_dimensions_by_matrix_mult(self, a, target_dim, ind):
        """
        a - is a 4D tensor to fit last two dimensions (dim 2, dim 3)
        target_dim - is four element tuple
        learning_params - multiplier matrix parameters should be learneble or fixed (makes sence only on 'matrix_mult' mode)
        fitting_mode = 'reshape' or 'matrix_mult'

        """
        d1, d2, d3, d4 = a.shape

        w1 = torch.Tensor(target_dim[2], d3)
        w1 = nn.init.xavier_uniform_(w1)

        w2 = torch.Tensor(d4, target_dim[3])
        w2 = nn.init.xavier_uniform_(w2)

        if self.learning_params:
            w1 = nn.Parameter(w1)
            w2 = nn.Parameter(w2)
            self.register_parameter('Param w1_{}'.format(ind), w1)
            self.register_parameter('Param w2_{}'.format(ind), w2)

        res = torch.Tensor(d1, d2, target_dim[2], target_dim[3])

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                res[i, j, :, :] = w1.mm(a[i, j, :, :]).mm(w2)

        return res

    @staticmethod
    def _fit_feature_map_dimensions_by_reshape(a, target_dim):
        return a.reshape(target_dim[0], -1, target_dim[2], target_dim[3])

    def check_and_fit_feature_map(self, layer_out, dims, ind):
        if (dims[2] == layer_out.shape[2]) and (dims[3] == layer_out.shape[3]):
            return layer_out
        elif self.mode == 'reshape':
            return self._fit_feature_map_dimensions_by_reshape(layer_out, dims)
        else:
            return self._fit_feature_map_dimensions_by_matrix_mult(layer_out, dims, ind)

    def extra_repr(self):
        return "route: layers {}".format(self.ind_arr)

    def forward(self, x):
        if self.layer_outputs_arr is None:
            raise Exception(
                'You should assign layers outputs array to prop of Route class instance: outputs_arr')
        d0, d1, d2, d3 = x.shape

        if len(self.ind_arr) == 1:
            return self.layer_outputs_arr[self.ind_arr[0]]

        elif len(self.ind_arr) > 1:
            result = self.layer_outputs_arr[self.ind_arr[0]]
            for i in self.ind_arr[1:]:
                layer_output = self.check_and_fit_feature_map(
                    self.layer_outputs_arr[i], result.shape, i)
                result = torch.cat((result, layer_output), dim=1)
            return result


class Shortcut(nn.Module):
    def __init__(self, lay_num):
        super().__init__()
        self.shortcut_layer_num = lay_num
        self.layer_outputs_arr = None

    def extra_repr(self):
        return "shortcut layer num {}".format(self.shortcut_layer_num)

    def forward(self, x):
        if self.layer_outputs_arr is None:
            raise Exception(
                'shortcut_layer_output is None. You should assign layers outputs array to prop of Shorcat class instance: outputs_arr')
        return self.layer_outputs_arr[self.shortcut_layer_num] + x


class Interpolate(nn.Module):
    def __init__(self, stride, mode):
        super().__init__()
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, size=None, scale_factor=self.stride, mode=self.mode, align_corners=False)

    def extra_repr(self):
        return "Interpolate: stride {}; mode '{}'".format(self.stride, self.mode)


class Yolo_Layer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, x):
        return x

    def extra_repr(self):
        return "yolo: anchors {}".format(self.anchors)


class Main_Module(nn.Module):
    def __init__(self, config_path, weights_path_str=None):
        super().__init__()
        self.output_list = []
        self.blocks = self._parse_cfg(config_path)
        self.module_list, self.net_info = self._construct_modules()
        self.predictions_list = []
        if weights_path_str is not None:
            self._load_weights(weights_path_str)

    @staticmethod
    def _parse_cfg(cfg_file_path: str) -> [dict]:
        """
        Takes a configuration file.
        Returns a list of layer_def_list. Each layer_def_list describes a layer_definition in the neural
        network to be built. Block is represented as a dictionary in the list

        """
        with open(cfg_file_path, 'r') as file:
            l_names = file.read().split('\n')
            l_names = [l_name.lstrip().rstrip() for l_name in l_names if (len(l_name) > 0) and (l_name[0] != '#')]
            #l_names = [l_name for l_name in l_names if (l_name[0] != '#')]
            #l_names = [l_name.lstrip().rstrip() for l_name in l_names]

            layer_definition = {}
            layer_def_list = []

            for l_name in l_names:
                if l_name.startswith('[') and l_name.endswith(']'):
                    layer_definition = {'type': l_name[1:-1]}
                    layer_def_list.append(layer_definition)
                else:
                    key, value = l_name.split('=')
                    layer_definition[key.rstrip()] = value.lstrip()
            return layer_def_list

    @staticmethod
    def _get_activation(activ_name: str) -> nn.Module or None:
        if activ_name == 'relu':
            return nn.ReLU(inplace=True)
        elif activ_name == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            return None

    def _construct_modules(self) -> (nn.ModuleList, dict):
        module_list = nn.ModuleList()
        net_info = self.blocks[0]
        prev_filters = 3
        output_filters = []

        for key in net_info.keys():
            try:
                s = net_info[key]
                if '.' in s:
                    v = float(net_info[key])
                else:
                    v = int(net_info[key])
                    net_info[key] = v
            except:
                pass

        for ind, block in enumerate(self.blocks[1:]):
            m = None
            if block['type'] == 'convolutional':
                try:
                    batch_norm_need = block['batch_normalize'] == '1'
                    has_bias = False
                except Exception:
                    batch_norm_need = False
                    has_bias = True

                filter_num = int(block['filters'])
                kern_size = int(block['size'])
                stride_size = int(block['stride'])
                pad = (kern_size - 1) // 2 if block['pad'] == '1' else 0

                conv = nn.Conv2d(in_channels=prev_filters,
                                 out_channels=filter_num,
                                 kernel_size=kern_size,
                                 stride=stride_size,
                                 padding=pad,
                                 bias=has_bias)

                m = SequentialCustom()
                m.has_batchNorm = False

                m.add_module('conv_{}'.format(ind), conv)

                if batch_norm_need:
                    m.add_module('batch_norm_{}'.format(
                        ind), nn.BatchNorm2d(filter_num))
                    m.has_batchNorm = True

                activation_layer = self._get_activation(block['activation'])

                if activation_layer:
                    m.add_module('activation_{0}_{1}'.format(
                        block['activation'], ind), activation_layer)

                output_filters.append(filter_num)
                prev_filters = filter_num

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                m = Interpolate(stride=stride, mode='bilinear')
                output_filters.append(prev_filters)

            elif block['type'] == 'shortcut':
                relative_layer_numb = int(block['from'])
                m = Shortcut(relative_layer_numb + ind)
                output_filters.append(prev_filters)

            elif block['type'] == 'route':
                layer_indexes = [int(l.lstrip().rstrip())
                                 for l in block['layers'].split(',')]
                layer_abs_indexes = [l_ind + ind if l_ind <
                                     0 else l_ind for l_ind in layer_indexes]

                if len(layer_abs_indexes) == 0:
                    raise ValueError(
                        "There are no layers to route from. layer_abs_indexes list is empty. Please specify in config file in Route section layer indexes.")
                elif len(layer_abs_indexes) == 1:
                    total_filters = output_filters[layer_abs_indexes[0]]
                else:
                    total_filters = 0
                    for i in layer_abs_indexes:
                        total_filters += output_filters[i]

                output_filters.append(total_filters)
                prev_filters = total_filters

                m = Route(layer_abs_indexes)

            elif block['type'] == 'yolo':
                mask = [int(i) for i in block['mask'].split(',')]
                anchors = [int(i) for i in block['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1])
                           for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                m = Yolo_Layer(anchors)
                output_filters.append(prev_filters)

            else:
                continue

            if m is not None:
                m.module_type = block['type']
                module_list.append(m)

        return module_list, net_info

    def _load_weights(self, pathstr):
        print("Loading weights...")
        with open(pathstr, mode='rb') as fo:
            header = np.fromfile(fo, dtype=np.int32, count=5, sep="")
            for i, m in enumerate(self.module_list):
                if isinstance(m, SequentialCustom):
                    mm_list = [mm for mm in m.children()]
                    if m.has_batchNorm == True:
                        num_bn_biases = mm_list[1].bias.numel()
                        bn_biase = torch.from_numpy(np.fromfile(
                            file=fo, dtype=np.float32, count=num_bn_biases))
                        bn_weight = torch.from_numpy(np.fromfile(
                            file=fo, dtype=np.float32, count=num_bn_biases))
                        bn_running_mean = torch.from_numpy(np.fromfile(
                            file=fo, dtype=np.float32, count=num_bn_biases))
                        bn_running_var = torch.from_numpy(np.fromfile(
                            file=fo, dtype=np.float32, count=num_bn_biases))

                        bn_biase = bn_biase.reshape_as(mm_list[1].bias.data)
                        bn_weight = bn_weight.reshape_as(
                            mm_list[1].weight.data)
                        bn_running_mean = bn_running_mean.reshape_as(
                            mm_list[1].running_mean)
                        bn_running_var = bn_running_var.reshape_as(
                            mm_list[1].running_var)

                        mm_list[1].bias.data.copy_(bn_biase)
                        mm_list[1].weight.data.copy_(bn_weight)
                        mm_list[1].running_mean.copy_(bn_running_mean)
                        mm_list[1].running_var.copy_(bn_running_var)

                    else:
                        num_conv_bias = mm_list[0].bias.numel()
                        conv_bias = torch.from_numpy(np.fromfile(
                            file=fo, dtype=np.float32, count=num_conv_bias))
                        conv_bias = conv_bias.reshape_as(mm_list[0].bias.data)
                        mm_list[0].bias.data.copy_(conv_bias)

                    num_conv_weights = mm_list[0].weight.numel()
                    conv_weight = torch.from_numpy(np.fromfile(
                        file=fo, dtype=np.float32, count=num_conv_weights))
                    conv_weight = conv_weight.reshape_as(
                        mm_list[0].weight.data)
                    mm_list[0].weight.data.copy_(conv_weight)
        print("Weights have been loaded successfully.")

    def reset_module(self):
        self.predictions_list.clear()
        self.output_list.clear()

    def forward(self, inp_x):

        self.reset_module()

        out_put = inp_x

        for i, m in enumerate(self.module_list):

            if m.module_type in ['route', 'shortcut']:
                m.layer_outputs_arr = self.output_list

            out_put = m(out_put)
            self.output_list.append(out_put)

            if isinstance(m, Yolo_Layer):
                pred_transformed = transform_prediction(
                    out_put, (self.net_info['height'], self.net_info['width']), m.anchors)
                self.predictions_list.append(pred_transformed)
            # print('{}. {} | out: {}'.format(i, m.extra_repr(), out_put.shape))

        all_predictions = torch.cat(self.predictions_list, dim=1)
        return all_predictions


def transform_prediction(prediction: torch.Tensor, img_dim: tuple, anchors: [tuple]) -> torch.Tensor:
    batch_size, channels_num, grid_size_h, grid_size_w = prediction.shape
    device = 'cuda' if prediction.is_cuda else 'cpu'
    num_anchors = len(anchors)  # here we have three anchors
    box_length = channels_num // num_anchors  # 255 // 3 = 85
    prediction_reshaped = prediction.transpose(1, 2).transpose(
        2, 3).reshape(batch_size, -1, box_length)
    stride_h = int(img_dim[0] / grid_size_h)
    stride_w = int(img_dim[1] / grid_size_w)

    torch.sigmoid_(prediction_reshaped[:, :, 0])
    torch.sigmoid_(prediction_reshaped[:, :, 1])

    grid_h = np.arange(grid_size_h)
    grid_w = np.arange(grid_size_w)

    a, b = np.meshgrid(grid_w, grid_h)

    x_offset = torch.FloatTensor(a).reshape(-1, 1).to(device=device)
    y_offset = torch.FloatTensor(b).reshape(-1, 1).to(device=device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).reshape(-1, 2).unsqueeze(0)
    prediction_reshaped[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors).to(device)
    anchors = anchors.repeat(grid_size_h * grid_size_w, 1).unsqueeze(0)
    prediction_reshaped[:, :, 2:4] = torch.exp(
        prediction_reshaped[:, :, 2:4]) * anchors

    prediction_reshaped[:, :, 0] *= stride_w
    prediction_reshaped[:, :, 1] *= stride_h

    torch.sigmoid_(prediction_reshaped[:, :, 4:])

    return prediction_reshaped


def convert_box_coords_to_corners_coords(prediction: torch.Tensor) -> [dict]:
    new_prediction = prediction.clone().detach()

    new_prediction[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    new_prediction[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    new_prediction[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    new_prediction[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)

    return new_prediction


def filter_boxes_by_confidence_threshold(pred: torch.Tensor, obj_threshold: float) -> torch.Tensor:
    """
        Filters prediction bboxes object confidence over obj_threshold, sets score and class number and image index for
        each bounding box returned by main_module.
        arg: pred (the output of main_module) with shape (N, bboxes_number, bbox_dimension):
                    N - is batch number,
                    bboxes_number - total number of bounding boxes returned by network
                    bbox_dimension - index 0:4 goes for bbox coordinates, 5 - is object confidence and 5:85
                    is class probabilities
        returns batch list, with each element as dictionary (each dictionary in batch list represents
                                                                            bbox data for one image).
                    dictionary keys are:
                    img_indx - index of image in the batch (int)
                    has_detection - if there is a decetion for the given image (True/False)
                    scores -> array of scores of bboxes with scores more than obj_threshold
                    bbox -> array of corresponding bbox coordinates
                    class  -> array of corresponding bbox class numbers
        """

    # Calculating total probabilities for classes
    obj_confidence = pred[:, :, 4].unsqueeze(2)
    pred[:, :, 5:] = obj_confidence * pred[:, :, 5:]

    # for each bbox, add new column to represent image index (index of image in a batch)
    pred = torch.cat(
        (pred, torch.zeros(pred.shape[0], pred.shape[1], 1).to(pred.device)), dim=2)
    for i in range(pred.shape[0]):
        pred[i, :, -1] = i

    # Calculating maximum scores for each bbox
    max_scores, max_score_classes = torch.max(
        pred[:, :, 5:-1], dim=2, keepdim=True)

    # Generating bbox attributes new structure: bbox coordinates, obj_confidence, score, class index and image index in the batch
    pred = torch.cat((pred[:, :, :5], max_scores, max_score_classes.float(
    ), pred[:, :, -1].unsqueeze(2)), dim=2)

    # Get image indexes
    img_indx = torch.unique(pred[:, :, -1])

    # For each image in the batch we add detections to this list (one elemenet per image)
    batch_list = []

    for i in img_indx:
        # filter out detections for particular image
        p = pred[pred[:, :, -1] == i]

        # filter out detections with object confidence over obj_threshold
        pr = p[p[:, 4] > obj_threshold]
        # if we have detection for this image
        if len(pr) > 0:
            batch_list.append({'img_indx': int(i), 'has_detection': True,
                               'scores': pr[:, 5], 'bbox': pr[:, :4], 'classes': pr[:, 6].int()})
        else:
            batch_list.append({'img_indx': int(i), 'has_detection': False, })

    return batch_list


def select_box_with_highest_score(scores_arr, box_arr):
    max_score, max_ind = torch.max(scores_arr, dim=0)
    return max_score, box_arr[max_ind], max_ind


def calc_intersection_side(x1, x2, x3, x4):
    def position(a1, a2, a3, a4):
        return a1 <= a2 <= a3 <= a4

    ai = 0.0
    if position(x1, x2, x3, x4):  # 1
        ai = 0.0
    elif position(x1, x2, x4, x3):  # 2
        ai = 0.0
    elif position(x1, x3, x2, x4):  # 3
        ai = x2 - x3
    elif position(x1, x3, x4, x2):  # 4
        ai = x4 - x3
    elif position(x1, x4, x2, x3):  # 5
        ai = x2 - x4
    elif position(x1, x4, x3, x2):  # 6
        ai = x3 - x4
    elif position(x2, x1, x3, x4):  # 7
        ai = 0.0
    elif position(x2, x1, x4, x3):  # 8
        ai = 0.0
    elif position(x2, x3, x1, x4):  # 9
        ai = x1 - x3
    elif position(x2, x3, x4, x1):  # 10
        ai = x4 - x3
    elif position(x2, x4, x1, x3):  # 11
        ai = x1 - x4
    elif position(x2, x4, x3, x1):  # 12
        ai = x3 - x4
    elif position(x3, x1, x2, x4):  # 13
        ai = x2 - x1
    elif position(x3, x1, x4, x2):  # 14
        ai = x4 - x1
    elif position(x3, x2, x1, x4):  # 15
        ai = x1 - x2
    elif position(x3, x2, x4, x1):  # 16
        ai = x4 - x2
    elif position(x3, x4, x1, x2):  # 17
        ai = 0.0
    elif position(x3, x4, x2, x1):  # 18
        ai = 0.0
    elif position(x4, x1, x2, x3):  # 19
        ai = x2 - x1
    elif position(x4, x1, x3, x2):  # 20
        ai = x3 - x1
    elif position(x4, x2, x1, x3):  # 21
        ai = x1 - x2
    elif position(x4, x2, x3, x1):  # 22
        ai = x3 - x2
    elif position(x4, x3, x1, x2):  # 23
        ai = 0.0
    elif position(x4, x3, x2, x1):  # 24
        ai = 0.0
    else:
        ai = 0.0
    return ai


def iou(box1, box2):
    x1 = box1[0]
    y1 = box1[1]
    x2 = box1[2]
    y2 = box1[3]

    x3 = box2[0]
    y3 = box2[1]
    x4 = box2[2]
    y4 = box2[3]

    '''
    print('x1 {}'.format(x1))
    print('y1 {}'.format(y1))
    print('x2 {}'.format(x2))
    print('y2 {}'.format(y2))
    print('x3 {}'.format(x3))
    print('y3 {}'.format(y3))
    print('x4 {}'.format(x4))
    print('y4 {}'.format(y4))
    '''
    iou_val = 0.0

    ai = calc_intersection_side(x1, x2, x3, x4)
    bi = calc_intersection_side(y1, y2, y3, y4)

    inter_area = ai * bi

    if inter_area > 0:
        # print('inter_area {}'.format(inter_area))
        box1_area = abs((x2 - x1) * (y1 - y2))
        box2_area = abs((x4 - x3) * (y3 - y4))
        union_area = box1_area + box2_area - inter_area
        # print('box1_area {}'.format(box1_area))
        # print('box2_area {}'.format(box2_area))
        # print('union_area {}'.format(union_area))
        iou_val = inter_area / union_area
    # print(iou)
    return iou_val


def calculate_iou_for_all_boxes(max_score_box, box_arr):
    iou_list = []
    for box in box_arr:
        iou_list.append(iou(max_score_box, box))
    iou_array = torch.tensor(iou_list)
    # iou_array = torch.rand((box_arr.shape[0],)) #this is only for testing purpose. comment it out in real calculations!!
    return iou_array


def yolo_non_max_suppression(scores, boxes, classes, iou_threshold=0.5):
    predicted_scores = []
    predicted_boxes = []
    predicted_classes = []

    # reduced_scores_arr = scores
    # reduced_box_arr = boxes

    unique_classes = classes.unique(sorted=True)

    for class_ind in unique_classes:
        class_filter_mask = classes == class_ind

        reduced_scores_arr = scores[class_filter_mask]
        reduced_box_arr = boxes[class_filter_mask]

        # print('reduced_scores_arr:\n {}'.format(reduced_scores_arr))
        # print('reduced_box_arr:\n {}'.format(reduced_box_arr))

        while (len(reduced_scores_arr)) > 0:
            # find the box with the highest score
            max_score, max_score_box, max_ind = select_box_with_highest_score(
                reduced_scores_arr, reduced_box_arr)
            # print('max_score: {}'.format(max_score.reshape(1,)))
            # print('max_score_box: {}'.format(max_score_box))

            # append this the max_score and max_box to predictions array
            predicted_scores.append(max_score.item())
            predicted_boxes.append(max_score_box.cpu().numpy().tolist())
            predicted_classes.append(class_ind)

            # compute overlap of max_score_box with all other boxes of the same class
            iou_array = calculate_iou_for_all_boxes(
                max_score_box, reduced_box_arr)
            # print('iou_array {}\n'.format(iou_array))

            # remove boxes with big overlap
            iou_treshold_mask = iou_array < iou_threshold
            reduced_box_arr = reduced_box_arr[iou_treshold_mask]
            reduced_scores_arr = reduced_scores_arr[iou_treshold_mask]

    # print('reduced_scores_arr:\n {}'.format(reduced_scores_arr))
    # print('reduced_box_arr:\n {}\n\n'.format(reduced_box_arr))

    return predicted_scores, predicted_boxes, predicted_classes


def load_classes(path_str):
    with open(path_str) as f:
        class_labels_list = f.readlines()
        return [class_label.strip() for class_label in class_labels_list]
