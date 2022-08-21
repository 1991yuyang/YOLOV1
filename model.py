import torch as t
from torch import nn
from torchvision import models


class YOLO(nn.Module):

    def __init__(self, S, B, C):
        """

        :param S: grid cell count, parameter S of original paper
        :param B: bounding box count of every grid cell, parameter B of original paper
        :param C: number of classes, parameter C of original paper
        """
        super(YOLO, self).__init__()
        output_channels = B * 5 + C
        self.backbone = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=S),
            nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)  # (N, channels, h_grid, w_grid), channels: x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, ..., class1, class2, ...
        return x


if __name__ == "__main__":
    d = t.randn(2, 3, 64, 64)
    model = YOLO(7, 2, 20)
    out = model(d)
    print(out.size())
