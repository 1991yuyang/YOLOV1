import torch as t
from torch import nn, optim
from torch.utils import data
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


epoch = 40
batch_size = 32
lr = 0.1
lr_de_epoch = 10
lr_de_rate = 0.1
softmax_op = nn.Softmax(dim=1).cuda(0)
criterion = nn.CrossEntropyLoss().cuda(0)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class MyNet(nn.Module):

    def __init__(self, conv_block_count):
        super(MyNet, self).__init__()
        self.convs = nn.Sequential()
        for c in range(conv_block_count):
            in_channels = 2 ** c
            out_channels = 2 ** (c + 1)
            self.convs.add_module("conv_%d" % (c,), ConvBlock(in_channels=in_channels, out_channels=out_channels))
        else:
            in_features = out_channels
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.clsf = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024, bias=False),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x).view((x.size()[0], -1))
        x = self.clsf(x)
        return x


class MySet(data.Dataset):

    def __init__(self, x, y):
        super(MySet, self).__init__()
        self.x = x.reshape((x.shape[0], 1, 8, 8))
        self.y = y

    def __getitem__(self, index):
        d = t.tensor(self.x[index]).type(t.FloatTensor)
        l = t.tensor(self.y[index]).type(t.LongTensor)
        return d, l

    def __len__(self):
        return self.x.shape[0]


def make_loader(x, y, batch_size):
    loader = iter(data.DataLoader(MySet(x, y), batch_size=batch_size, shuffle=True, drop_last=True))
    return loader


def calc_accu(model_output, target):
    accu = (t.argmax(softmax_op(model_output), dim=1) == target).sum().item() / model_output.size()[0]
    return accu


def train_epoch(model, train_loader, current_epoch, criterion, optimizer):
    model.train()
    steps = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(0)
        l_train_cuda = l_train.cuda(0)
        train_output = model(d_train_cuda)
        train_loss = criterion(train_output, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if current_step % 10 == 0:
            train_accu = calc_accu(train_output, l_train_cuda)
            print("epoch:%d/%d, step:%d/%d, loss:%.5f, accu:%.5f" % (current_epoch, epoch, current_step, steps, train_loss.item(), train_accu))
        current_step += 1
    return model


def valid_epoch(model, valid_loader, criterion, current_epoch):
    model.eval()
    steps = len(valid_loader)
    accum_loss = 0
    accum_accu = 0
    for d_valid, l_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(0)
        l_valid_cuda = l_valid.cuda(0)
        with t.no_grad():
            valid_output = model(d_valid_cuda)
            valid_loss = criterion(valid_output, l_valid_cuda)
            valid_accu = calc_accu(valid_output, l_valid_cuda)
            accum_accu += valid_accu
            accum_loss += valid_loss.item()
    avg_accu = accum_accu / steps
    avg_loss = accum_loss / steps
    print("######valid epoch:%d#######" % (current_epoch))
    print("loss:%.5f, accu:%.5f" % (avg_loss, avg_accu))
    return model


def main():
    x, y = load_digits()["data"], load_digits()["target"]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=32, test_size=0.25)
    model = MyNet(conv_block_count=3)
    model = nn.DataParallel(module=model, device_ids=[0])
    model = model.cuda(0)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    lr_sch = optim.lr_scheduler.StepLR(optimizer, lr_de_epoch, lr_de_rate)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(x_train, y_train, batch_size)
        valid_loader = make_loader(x_valid, y_valid, batch_size)
        model = train_epoch(model, train_loader, current_epoch, criterion, optimizer)
        model = valid_epoch(model, valid_loader, criterion, current_epoch)
        lr_sch.step()


if __name__ == "__main__":
    main()