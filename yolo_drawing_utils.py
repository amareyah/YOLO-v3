import colorsys
import random
from torch import cat, reshape, from_numpy

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.JpegImagePlugin import JpegImageFile


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def image_padding(img, inp_dim):
    '''resizing image to model image size, with unchanged aspect ratio using padding'''

    # original image size
    img_w = img.size[0]
    img_h = img.size[1]

    # darknet model input size
    p_h, p_w = inp_dim

    # original image size in padded image
    new_w = int(img_w * min(p_w / img_w, p_h / img_h))
    new_h = int(img_h * min(p_w / img_w, p_h / img_h))

    # actualy resize original image to fit in padded image (maintaining the aspect ratio)
    resized_image = img.resize((new_w, new_h), Image.BICUBIC)

    # padded (grey area) size. height and width (only one side).
    # we use them here and later to resize bbox to fit the original image
    c_w = (p_w - new_w) // 2
    c_h = (p_h - new_h) // 2

    # scale size between the original image and the padded image. we use this scale later for bbox resizing.
    scale = img_w / p_w

    # np array sized to padded image and filled with grey color.
    canvas = np.full((inp_dim[0], inp_dim[1], 3), 128)

    # fitting original scaled image to padded gray image
    canvas[c_h: c_h + new_h, c_w: c_w + new_w, :] = resized_image

    return canvas, (c_h, c_w, scale)


def preprocess_image(img_path, target_size) -> tuple:
    with Image.open(img_path) as img:
        resized_image_data, bbox_transformation_data_for_original_img = image_padding(img, target_size)
        data = resized_image_data.transpose(2, 0, 1)
        data = data / 255.
        data = np.expand_dims(data, 0)  # Add batch dimension.

    return from_numpy(data).float(), bbox_transformation_data_for_original_img


def draw_boxes(image: JpegImageFile, out_scores: [], out_boxes: [], out_classes: [], class_names: [], colors: []):
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
    thickness =  2 #(image.size[0] + image.size[1]) // 500

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} | prob.: {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box

        left = max(0, np.floor(left + 0.5).astype('int32'))
        top = max(0, np.floor(top + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

        bbox_label = '| bbox: left-top:({},{}); right-bottom({},{})'.format(left, top, right, bottom)
        print(label, bbox_label)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


def transform_bbox_to_fit_original_image(pred_bboxes: [[]], scaling_data: tuple) -> np.array:
    pred_bboxes = np.array(pred_bboxes)
    c_h, c_w, scale = scaling_data
    pred_bboxes[:, [0, 2]] = pred_bboxes[:, [0, 2]] - c_w
    pred_bboxes[:, [1, 3]] = pred_bboxes[:, [1, 3]] - c_h
    pred_bboxes = pred_bboxes * scale

    return pred_bboxes
