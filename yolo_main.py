import torch
import yolo_image_detect_utils as detect
import yolo_drawing_utils as draw
import time
import os
import os.path as osp
from PIL import Image

config_file_path = 'cfg/yolov3.cfg'
yolo_weights_file_path = 'cfg/yolov3.weights'
confidence_threshold = 0.55
nms_threshold = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Creating model...")
dn = detect.Main_Module(config_file_path, yolo_weights_file_path).eval().to(device)
print("Model created successfully.\n")

assert dn.net_info["height"] % 32 == 0
assert dn.net_info["width"] % 32 == 0
assert dn.net_info["height"] > 32
assert dn.net_info["width"] > 32

# I define here an image folder, with images to process by model.
image_path = 'images'
image_path = osp.realpath(image_path)

# This is destination folder, where processed images will be saved.
destination_path = 'out'
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

img_filenames = os.listdir(image_path)
image_path_list = [osp.join(image_path, filename)
                   for filename in img_filenames]

# this is a class names list for those 80 classes, for which the model is trained.
class_names = detect.load_classes('cfg/coco_classes.txt')

# this loop cycles over image_path_list and for each image makes detection.
for i, (image_path, filename) in enumerate(zip(image_path_list, img_filenames)):
    print("Processing file {}".format(filename))
    start = time.time()
    with Image.open(image_path) as image:

        # preprocess image (resize with padding, normalize..)
        img_data, bbox_transformation_data_for_original_img = draw.preprocess_image(
            image_path, (dn.net_info['height'], dn.net_info['width']))

        img_data = img_data.to(device)

        out = dn(img_data)
        out = out.detach()

        # convert bbox coordinates (center point and lengths) to box corner coordinates
        out = detect.convert_box_coords_to_corners_coords(out)

        # Filter boxes with score more than givven threshold
        pred = detect.filter_boxes_by_confidence_threshold(out, confidence_threshold)

        '''
        Becase we have input as a batch of images, the model output is also a batch of predictions.
        the batch is represented as a list.
        each element is a prediction for one particular image. elements are represented as dictionary.
        in my case, my input batch consists of one image. therefore prediction batch has only one element --> pred[0].
        i check here if there is a detection on the given image.
        '''
        if pred[0]['has_detection']:

            # Run non_max_supression for this particular image
            pred_scores, pred_bboxes, pred_classes = detect.yolo_non_max_suppression(
                scores=pred[0]['scores'], boxes=pred[0]['bbox'], classes=pred[0]['classes'], iou_threshold=nms_threshold)

            # transform bbox coordinates in accordance to original image
            pred_bboxes = draw.transform_bbox_to_fit_original_image(
                pred_bboxes, bbox_transformation_data_for_original_img)

            # Generate colors for drawing bounding boxes.
            colors = draw.generate_colors(class_names)

            # Draw bounding boxes on the image file
            draw.draw_boxes(image, pred_scores, pred_bboxes,
                            pred_classes, class_names, colors)

            # Save the predicted bounding box on the image
            image.save(os.path.join("out", filename), quality=100)

            image.close()
        else:
            print('No detection.')
    end = time.time()
    print('Detection time: {:.2f} sec \n'.format(end - start))

    torch.cuda.empty_cache()
