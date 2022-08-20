from torch.utils import data
import os
import cv2
import numpy as np
import torch as t

"""
label is yolo format
every label file is a txt file
the format of every line of txt file is: class_id center_x center_y w h

data_root_dir
    images
        train
            1.jpg
            2.jpg
            ...
        val
            1.jpg
            2.jpg
            ...
    labels
        train
            1.txt
            2.txt
            ...
        val
            1.txt
            2.txt
            ...
"""


class MySet(data.Dataset):

    def __init__(self, S, data_root_dir, is_train, C, img_suffix, img_size):
        self.S = S
        self.is_train = is_train
        self.unit_grid_cell_ratio = 1 / S
        self.channels = 5 + C
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.data_root_dir = data_root_dir
        if is_train:
            self.img_dir = os.path.join(data_root_dir, "images", "train")
            self.label_dir = os.path.join(data_root_dir, "labels", "train")
        else:
            self.img_dir = os.path.join(data_root_dir, "images", "val")
            self.label_dir = os.path.join(data_root_dir, "labels", "val")
        names = [name.split(".")[0] for name in os.listdir(self.img_dir)]
        self.img_pths = [os.path.join(self.img_dir, "%s.%s" % (name, img_suffix)) for name in names]
        self.label_pths = [os.path.join(self.label_dir, "%s.%s" % (name, "txt")) for name in names]

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        label_pth = self.label_pths[index]
        img = cv2.imread(img_pth)
        orig_img_size = t.tensor(img.shape[:2]).type(t.FloatTensor)  # (orig_h, orig_w)
        img = cv2.resize(img, self.img_size, cv2.INTER_CUBIC)
        label_ret = np.zeros(shape=(self.channels, self.S, self.S), dtype=np.float32)  # (5 + classes, h_grid_count, w_grid_count)
        with open(label_pth, "r", encoding="utf-8") as file:
            label = [[float(i) for i in line.strip(" ").split(" ")] for line in file.read().strip("\n").split("\n")]
        for bbox in label:
            """
            iterate every bounding box of this image
            """
            class_index, center_x, center_y, w, h = bbox
            class_index = int(class_index)
            x_grid_index = int(center_x // self.unit_grid_cell_ratio)
            y_grid_index = int(center_y // self.unit_grid_cell_ratio)
            x_offset = (center_x % self.unit_grid_cell_ratio) / self.unit_grid_cell_ratio
            y_offset = (center_y % self.unit_grid_cell_ratio) / self.unit_grid_cell_ratio
            label_ret[:, y_grid_index, x_grid_index][:5] = np.array([x_offset, y_offset, w, h, 1])
            label_ret[:, y_grid_index, x_grid_index][5 + class_index] = 1
        img = img / 255
        img = t.from_numpy(np.transpose(img, axes=[2, 0, 1])).type(t.FloatTensor)
        return img, label_ret, orig_img_size

    def __len__(self):
        return len(self.img_pths)


def make_loader(S, data_root_dir, is_train, C, img_suffix, img_size, batch_size, num_workers):
    loader = iter(data.DataLoader(MySet(S, data_root_dir, is_train, C, img_suffix, img_size), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    s = make_loader(7, "./datasets", True, 1, "jpg", 640, 4, 1)
    for d, l, orig_img_size in s:
        sample_indexs, h_grid_indexs, w_grid_indexs = search_obj_grid_cell(l)
        print(l[sample_indexs, 4, h_grid_indexs, w_grid_indexs])