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
        backbone = models.resnet18(pretrained=True)
        self.stage0 = nn.Sequential(*list(backbone.children())[:4])
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=S),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        )
        self.head2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=S),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        )
        self.head3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=S),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        )
        self.head4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=S),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        stage0_output = self.stage0(x)
        stage1_output = self.stage1(stage0_output)
        stage2_output = self.stage2(stage1_output)
        stage3_output = self.stage3(stage2_output)
        stage4_output = self.stage4(stage3_output)
        head1_result = self.head1(stage1_output)
        head2_result = self.head2(stage2_output)
        head3_result = self.head3(stage3_output)
        head4_result = self.head4(stage4_output)
        output = (self.sigmoid(head1_result) + self.sigmoid(head2_result) + self.sigmoid(head3_result) + self.sigmoid(head4_result)) / 4  # (N, channels, h_grid, w_grid), channels: x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, ..., class1, class2, ...
        return output


if __name__ == "__main__":
    d = t.randn(2, 3, 64, 64)
    model = YOLO(7, 2, 20)
    out = model(d)
    print(out.size())