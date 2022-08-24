import torch as t
from torch import nn, optim
from model import YOLO
from data import make_loader
import os
from loss import Loss
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))


data_root_dir = r"datasets"
B = 2
S = 7
C = 1
init_lr = 0.01
final_lr = 0.0001
epoch = 300
batch_size = 2
img_size = 640
lamda_coord = 5
lamda_noobj = 0.5
print_step = 1
num_workers = 4
weight_decay = 0.0005
img_suffix = "jpg"
best_valid_loss = float("inf")
criterion = Loss(lamda_coord, lamda_noobj, B, S).cuda(device_ids[0])


def train_epoch(model, current_epoch, criterion, optimizer, train_loader):
    model.train()
    steps = len(train_loader)
    current_step = 1
    for train_img, train_label_ret, train_orig_image_size in train_loader:
        train_img_cuda = train_img.cuda(device_ids[0])
        train_label_ret_cuda = train_label_ret.cuda(device_ids[0])
        train_orig_image_size_cuda = train_orig_image_size.cuda(device_ids[0])
        train_output = model(train_img_cuda)
        train_total_loss, train_coord_loss, train_obj_conf_loss, train_noobj_conf_loss, train_class_loss = criterion(train_label_ret_cuda, train_output, train_orig_image_size_cuda)
        optimizer.zero_grad()
        train_total_loss.backward()
        optimizer.step()
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, coord_loss:%.5f, obj_conf_loss:%.5f, noobj_conf_loss:%.5f, class_loss:%.5f, total_loss:%.5f" % (current_epoch, epoch, current_step, steps, train_coord_loss.item(), train_obj_conf_loss.item(), train_noobj_conf_loss.item(), train_class_loss.item(), train_total_loss.item()))
        current_step += 1
    print("saving epoch model......")
    t.save(model.module.state_dict(), "epoch.pth")
    return model


def valid_epoch(model, current_epoch, valid_loader, criterion):
    global best_valid_loss
    model.eval()
    steps = len(valid_loader)
    accum_coord_loss = 0
    accum_obj_conf_loss = 0
    accum_noobj_conf_loss = 0
    accum_class_loss = 0
    accum_total_loss = 0
    for val_img, val_label_ret, val_orig_image_size in valid_loader:
        val_img_cuda = val_img.cuda(device_ids[0])
        val_label_ret_cuda = val_label_ret.cuda(device_ids[0])
        val_orig_image_size_cuda = val_orig_image_size.cuda(device_ids[0])
        with t.no_grad():
            val_output = model(val_img_cuda)
            val_total_loss, val_coord_loss, val_obj_conf_loss, val_noobj_conf_loss, val_class_loss = criterion(val_label_ret_cuda, val_output, val_orig_image_size_cuda)
            accum_total_loss += val_total_loss.item()
            accum_coord_loss += val_coord_loss.item()
            accum_obj_conf_loss += val_obj_conf_loss.item()
            accum_noobj_conf_loss += val_noobj_conf_loss.item()
            accum_class_loss += val_class_loss.item()
    avg_total_loss = accum_total_loss / steps
    avg_coord_loss = accum_coord_loss / steps
    avg_obj_conf_loss = accum_obj_conf_loss / steps
    avg_noobj_conf_loss = accum_noobj_conf_loss / steps
    avg_class_loss = accum_class_loss / steps
    if avg_total_loss < best_valid_loss:
        best_valid_loss = avg_total_loss
        print("saving best model......")
        t.save(model.module.state_dict(), "best.pth")
    print("##############valid epoch:%d###############" % (current_epoch,))
    print(
        "coord_loss:%.5f, obj_conf_loss:%.5f, noobj_conf_loss:%.5f, class_loss:%.5f, total_loss:%.5f" % (
        avg_coord_loss, avg_obj_conf_loss,
        avg_noobj_conf_loss, avg_class_loss, avg_total_loss))
    return model


def main():
    model = YOLO(S, B, C)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=final_lr)
    for e in range(epoch):
        current_epoch = e + 1
        print("lr:%.5f" % (lr_sch.get_lr()[0],))
        train_loader = make_loader(S, data_root_dir, True, C, img_suffix, img_size, batch_size, num_workers)
        valid_loader = make_loader(S, data_root_dir, False, C, img_suffix, img_size, batch_size, num_workers)
        model = train_epoch(model, current_epoch, criterion, optimizer, train_loader)
        model = valid_epoch(model, current_epoch, valid_loader, criterion)
        lr_sch.step()


if __name__ == "__main__":
    main()