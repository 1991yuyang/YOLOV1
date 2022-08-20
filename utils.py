import torch
import numpy as np
import torch as t


def get_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算交并比（交集/并集）
    iou = inter_area / (area1 + area2 - inter_area)  # 注意：这里inter_area不能乘以2，乘以2就相当于把交集部分挖空了
    return iou


def search_obj_noobj_grid_cell(label_ret):
    obj_sample_indexs, obj_h_grid_indexs, obj_w_grid_indexs = np.where(label_ret[:, 4, :, :].cpu().detach().numpy() == 1)
    obj_sample_indexs = obj_sample_indexs.tolist()
    obj_h_grid_indexs = obj_h_grid_indexs.tolist()
    obj_w_grid_indexs = obj_w_grid_indexs.tolist()
    noobj_sample_indexs, noobj_h_grid_indexs, noobj_w_grid_indexs = np.where(label_ret[:, 4, :, :].cpu().detach().numpy() == 0)
    noobj_sample_indexs = noobj_sample_indexs.tolist()
    noobj_h_grid_indexs = noobj_h_grid_indexs.tolist()
    noobj_w_grid_indexs = noobj_w_grid_indexs.tolist()
    return obj_sample_indexs, obj_h_grid_indexs, obj_w_grid_indexs, noobj_sample_indexs, noobj_h_grid_indexs, noobj_w_grid_indexs


def get_part_of_loss(label_ret, model_output, B, S, orig_image_size):
    obj_sample_indexs, obj_h_grid_indexs, obj_w_grid_indexs, noobj_sample_indexs, noobj_h_grid_indexs, noobj_w_grid_indexs = search_obj_noobj_grid_cell(label_ret)
    obj_orig_image_size = orig_image_size[obj_sample_indexs, :]  # (obj_sample_count, 2)
    obj_coord_gt = label_ret[obj_sample_indexs, :4, obj_h_grid_indexs, obj_w_grid_indexs]  # (obj_sample_count, 4)
    obj_class_gt = label_ret[obj_sample_indexs, 5:, obj_h_grid_indexs, obj_w_grid_indexs]  # (obj_sample_count, C)
    temp_obj_conf_gt = label_ret[obj_sample_indexs, 4, obj_h_grid_indexs, obj_w_grid_indexs]  # (obj_sample_count,)
    noobj_conf_gt_1 = t.cat([label_ret[noobj_sample_indexs, 4:5, noobj_h_grid_indexs, noobj_w_grid_indexs]] * B, dim=1)  # (noobj_sample_count, B)
    # noobj_class_gt = label_ret[noobj_sample_indexs, 5:, noobj_h_grid_indexs, noobj_w_grid_indexs]  # (noobj_sample_count, C)
    temp_obj_model_output_coord = t.cat([model_output[obj_sample_indexs, i * 5:i * 5 + 4, obj_h_grid_indexs, obj_w_grid_indexs] for i in range(B)], dim=1)  # (obj_sample_count, B * 4)
    obj_model_output_class = model_output[obj_sample_indexs, B * 5:, obj_h_grid_indexs, obj_w_grid_indexs]  # (obj_sample_count, C)
    # noobj_model_output_class = model_output[noobj_sample_indexs, B * 5:, noobj_h_grid_indexs, noobj_w_grid_indexs]  # (noobj_sample_count, C)
    noobj_model_output_conf_1 = model_output[noobj_sample_indexs, 4:B * 5:5, noobj_h_grid_indexs, noobj_w_grid_indexs]  # (noobj_sample_count, B)
    temp_obj_model_output_conf = model_output[obj_sample_indexs, 4:B * 5:5, obj_h_grid_indexs, obj_w_grid_indexs]  # (obj_sample_count, B)
    obj_model_output_coord = []
    obj_model_output_conf = []
    max_ious = []
    noobj_conf_gt_2 = []
    noobj_model_output_conf_2 = []
    for i in range(len(obj_sample_indexs)):
        w_grid_index = obj_w_grid_indexs[i]
        h_grid_index = obj_h_grid_indexs[i]
        max_iou = -float("inf")
        max_b = None
        orig_h, orig_w = obj_orig_image_size[i].cpu().detach().numpy().tolist()
        h_grid_size = orig_h / S
        w_grid_size = orig_w / S
        gt_center_x, gt_center_y, gt_w, gt_h = obj_coord_gt[i].cpu().detach().numpy().tolist()
        gt_center_x_real = w_grid_size * w_grid_index + gt_center_x * w_grid_size
        gt_center_y_real = h_grid_size * h_grid_index + gt_center_y * h_grid_size
        gt_real_w = gt_w * orig_w
        gt_real_h = gt_h * orig_h
        gt_tl_x = gt_center_x_real - gt_real_w / 2
        gt_tl_y = gt_center_y_real - gt_real_h / 2
        gt_br_x = gt_center_x_real + gt_real_w / 2
        gt_br_y = gt_center_y_real + gt_real_h / 2
        for b in range(B):
            center_x, center_y, w, h = temp_obj_model_output_coord[i, b * 4:(b + 1) * 4].cpu().detach().numpy().tolist()
            center_x_real = w_grid_size * w_grid_index + center_x * w_grid_size
            center_y_real = h_grid_size * h_grid_index + center_y * h_grid_size
            real_w = w * orig_w
            real_h = h * orig_h
            tl_x = center_x_real - real_w / 2
            tl_y = center_y_real - real_h / 2
            br_x = center_x_real + real_w / 2
            br_y = center_y_real + real_h / 2
            iou = get_iou([gt_tl_x, gt_tl_y, gt_br_x, gt_br_y], [tl_x, tl_y, br_x, br_y])
            if iou > max_iou:
                max_iou = iou
                max_b = b
                max_iou_coord = temp_obj_model_output_coord[i:i + 1, b * 4:(b + 1) * 4]
                max_iou_conf = temp_obj_model_output_conf[i:i + 1, b]
        for _b in range(B):
            if _b == max_b:
                continue
            noobj_model_output_conf_2.append(temp_obj_model_output_conf[i:i + 1, _b])
            noobj_conf_gt_2.append(0)
        max_ious.append(max_iou)
        obj_model_output_coord.append(max_iou_coord)
        obj_model_output_conf.append(max_iou_conf)
    obj_model_output_coord = t.cat(obj_model_output_coord, dim=0)
    obj_model_output_conf = t.cat(obj_model_output_conf, dim=-1)
    obj_conf_gt = temp_obj_conf_gt * t.tensor(max_ious).to(temp_obj_conf_gt.device)
    noobj_model_output_conf_2 = t.cat(noobj_model_output_conf_2, dim=-1)
    noobj_conf_gt_2 = t.tensor(noobj_conf_gt_2).type(t.FloatTensor).to(noobj_model_output_conf_2.device)
    return obj_coord_gt, obj_class_gt, obj_conf_gt, noobj_conf_gt_1, noobj_conf_gt_2, obj_model_output_coord, obj_model_output_class, obj_model_output_conf,  noobj_model_output_conf_1, noobj_model_output_conf_2


if __name__ == "__main__":
    d = np.array([1, 2, 3, 4])
    d2 = np.array([1, 2, 3, 4])
    from data import make_loader
    from model import YOLO

    model = YOLO(7, 2, 1).cuda(0)
    data_loader = make_loader(7, "datasets", True, 1, "jpg", 640, 4, 1)
    for img, label_ret, orig_img_size in data_loader:
        output = model(img.cuda(0))
        get_part_of_loss(label_ret.cuda(0), output.cuda(0), 2, 7, orig_img_size.cuda(0))