from utils import get_part_of_loss
from torch import nn
import torch as t


class Loss(nn.Module):

    def __init__(self, lamda_coord, lamda_noobj, B, S):
        super(Loss, self).__init__()
        self.lamda_coord = lamda_coord
        self.lamda_noobj = lamda_noobj
        self.B = B
        self.S = S

    def forward(self, label_ret, model_output, orig_image_size):
        obj_coord_gt, obj_class_gt, obj_conf_gt, noobj_conf_gt_1, noobj_conf_gt_2, obj_model_output_coord, obj_model_output_class, obj_model_output_conf, noobj_model_output_conf_1, noobj_model_output_conf_2 = get_part_of_loss(label_ret, model_output, self.B, self.S, orig_image_size)
        coord_loss_center = self.lamda_coord * t.sum(t.pow((obj_coord_gt[:, :2] - obj_model_output_coord[:, :2]), 2))
        coord_loss_wh = self.lamda_coord * t.sum(t.pow((t.sqrt(obj_coord_gt[:, 2:]) - t.sqrt(obj_model_output_coord[:, 2:])), 2))
        coord_loss = coord_loss_center + coord_loss_wh
        class_loss = t.sum(t.pow(obj_class_gt - obj_model_output_class, 2))
        obj_conf_loss = t.sum(t.pow(obj_conf_gt - obj_model_output_conf, 2))
        noobj_conf_loss_1 = self.lamda_noobj * t.sum(t.pow(noobj_conf_gt_1 - noobj_model_output_conf_1, 2))
        noobj_conf_loss_2 = self.lamda_noobj * t.sum(t.pow(noobj_conf_gt_2 - noobj_model_output_conf_2, 2))
        noobj_conf_loss = noobj_conf_loss_1 + noobj_conf_loss_2
        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
        return total_loss, coord_loss, obj_conf_loss, noobj_conf_loss, class_loss