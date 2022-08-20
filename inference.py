import torch as t
from model import YOLO
import os
import cv2
import numpy as np
from torchvision.ops import nms
from numpy import random as rd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

B = 2
S = 7
C = 1
img_size = (640, 640)
img_pth = r"datasets/images/train/qiushaya76.jpg"
use_bset_model = False
conf_thres = 0.4
nms_iou_thres = 0.7
class_names = ["qiushaya"]  # 保持和voc2yolo.py中顺序一致
colors = [[rd.randint(0, 255) for _ in range(3)] for _ in class_names]


def load_model(S, B, C, use_best_model):
    """

    :param S: grid cell count, parameter S in original paper
    :param B: box count of every grid cell, parameter B in original paper
    :param C: class count, parameter C in original paper
    :param use_best_model: True will use best model, False use epoch model
    :return:
    """
    model = YOLO(S, B, C)
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def load_one_img(img_pth, img_size):
    """

    :param img_pth: image path
    :param img_size: image size, tuple
    :return:
    """
    img = cv2.imread(img_pth)
    orig_cv2_img = img.copy()
    orig_img_size = img.shape[:2]  # (orig_h, orig_w)
    img = cv2.resize(img, img_size, cv2.INTER_CUBIC)
    img = img / 255
    img = t.from_numpy(np.transpose(img, axes=[2, 0, 1])).type(t.FloatTensor).unsqueeze(0).cuda(0)
    return img, orig_img_size, orig_cv2_img


def inference(model, img):
    """

    :param model: return value of load_model
    :param img: return value of load_one_img
    :return:
    """
    with t.no_grad():
        model_output = model(img)[0].cpu().detach().numpy()  # (channels, h_grid_indexs, w_grid_indexs)
    return model_output


def postprocess(model_output, conf_thres, nms_iou_thres, B, S, orig_img_size):
    """

    :param model_output: model inference output
    :param conf_thres: confidence threshold value
    :param nms_iou_thres: nms iou threshold value
    :param B: box count of every grid cell, parameter B in original paper
    :param S: grid cell count, parameter S in original paper
    :param orig_img_size: original image, cv2 bgr format
    :return: cls_x1_y1_x2_y2_conf, dict data type, key is class index, value is bounding box information of current key (class index)
    like: {
    0: [
        [x11, y11, x12, y12, conf1],
        [x21, y21, x22, y22, conf2],
        ...
    ],
    1: [
        [],
        [],
        ...
    ],
    ...
    }
    """
    unit_w_grid_size = orig_img_size[1] / S
    unit_h_grid_size = orig_img_size[0] / S
    conf_info = model_output[4:B * 5:5, :, :]  # (B, h_grid_indexs, w_grid_indexs)
    box_indexs, h_grid_indexs, w_grid_indexs = np.where(conf_info > conf_thres)
    obj_box_indexs = box_indexs.tolist()
    obj_classes = model_output[B * 5:, h_grid_indexs, w_grid_indexs]
    class_indexs = np.argmax(obj_classes, axis=0).tolist()
    h_w_box_cls = dict(list(zip([tuple(i) for i in list(zip(h_grid_indexs, w_grid_indexs, obj_box_indexs))], class_indexs)))
    unique_class_indexs = np.unique(class_indexs)
    cls_x1_y1_x2_y2_conf = {}
    for ci in unique_class_indexs:
        if ci not in cls_x1_y1_x2_y2_conf:
            cls_x1_y1_x2_y2_conf[ci] = []
        for k, v in h_w_box_cls.items():
            if v == ci:
                h_grid_index = k[0]
                w_grid_index = k[1]
                info = model_output[k[2] * 5:(k[2] + 1) * 5, k[0], k[1]]
                w_box = info[2] * orig_img_size[1]
                h_box = info[3] * orig_img_size[0]
                x_center = w_grid_index * unit_w_grid_size + info[0] * unit_w_grid_size
                y_center = h_grid_index * unit_h_grid_size + info[1] * unit_h_grid_size
                x_tl = x_center - w_box / 2 if x_center - w_box / 2 > 0 else 0
                y_tl = y_center - h_box / 2 if y_center - h_box / 2 > 0 else 0
                x_br = x_center + w_box / 2 if x_center + w_box / 2 < orig_img_size[1] else orig_img_size[1] - 1
                y_br = y_center + h_box / 2 if y_center + h_box / 2 < orig_img_size[0] else orig_img_size[0] - 1
                conf = info[4]
                cls_x1_y1_x2_y2_conf[ci].append([int(x_tl), int(y_tl), int(x_br), int(y_br), conf])
    for k in cls_x1_y1_x2_y2_conf.keys():
        current_class_boxes = t.from_numpy(np.array(cls_x1_y1_x2_y2_conf[k]))
        bbx_index_after_nms = nms(current_class_boxes[:, :4], current_class_boxes[:, 4], nms_iou_thres)
        bounding_box_keept = current_class_boxes[bbx_index_after_nms, :]
        cls_x1_y1_x2_y2_conf[k] = bounding_box_keept.cpu().detach().numpy().tolist()
    return cls_x1_y1_x2_y2_conf


def draw_box(orig_cv2_img, cls_x1_y1_x2_y2_conf, thickness, lineType):
    """

    :param orig_cv2_img: original cv2 image
    :param cls_x1_y1_x2_y2_conf: return value of postprocess
    :param thickness: bounding box line thickness
    :param lineType: bounding box line type
    :return: cv2 BGR image with bounding boxes
    """
    for k, v in cls_x1_y1_x2_y2_conf.items():
        class_name = class_names[k]
        color = colors[k]
        for box in v:
            x_tl, y_tl, x_br, y_br, conf = box
            print(class_name, conf)
            class_name_new = "%s:%.2f" % (class_name, conf)
            x_tl = int(x_tl)
            y_tl = int(y_tl)
            x_br = int(x_br)
            y_br = int(y_br)
            cv2.rectangle(orig_cv2_img, (x_tl, y_tl), (x_br, y_br), color, thickness, lineType)
            t_size = cv2.getTextSize(class_name_new, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textlbottom = np.array((x_tl, y_tl)) + np.array(list(t_size))
            cv2.rectangle(orig_cv2_img, (x_tl, y_tl), tuple(textlbottom), color, -1)
            cv2.putText(orig_cv2_img, class_name_new, (int(x_tl + (t_size[1] / 2 + 4)) - 1, y_tl + 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    cv2.imshow("result", orig_cv2_img)
    cv2.waitKey()
    return orig_cv2_img


if __name__ == "__main__":
    model = load_model(S, B, C, True)
    img, orig_img_size, orig_cv2_img = load_one_img(img_pth, img_size)
    model_output = inference(model, img)
    cls_x1_y1_x2_y2_conf = postprocess(model_output, conf_thres, nms_iou_thres, B, S, orig_img_size)
    draw_box(orig_cv2_img, cls_x1_y1_x2_y2_conf, 2, 4)