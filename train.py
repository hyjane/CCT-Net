from ast import parse
import os, argparse, math, sys
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from medpy.metric.binary import hd, dc, assd, jc
from utils import load_model
from losses import dice_loss
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, ExponentialLR
import time
from torch.utils.data import TensorDataset, DataLoader

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# 参数设置
parser = argparse.ArgumentParser()
# parser.add_argument('--arch', type=str, default='FAT')
parser.add_argument('--arch', type=str, default='CCT')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset1', type=str, default='datas')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--lr_seg', type=float, default=1e-4)  # 1e-4.  0.0003
parser.add_argument('--n_epochs', type=int, default=200)  # 100
parser.add_argument('--bt_size', type=int, default=8)  # 36
parser.add_argument('--seg_loss', type=int, default=0, choices=[0, 1])
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--patience', type=int, default=500)  # 50

# pre-Train，预训练
parser.add_argument('--pre', type=int, default=0)

# transformer，transformer
parser.add_argument('--trans', type=int, default=1)

# cross-scale framework
parser.add_argument('--cross', type=int, default=0)

parse_config = parser.parse_args()
print(parse_config)


from CCT import CCT_Net

# from FAT_Net import FAT_Net
# if parse_config.arch == 'FAT':
if parse_config.arch == 'CCT':
    parse_config.exp_name += '_{}_{}'.format(parse_config.trans,
                                             parse_config.cross,
                                             )
exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
    parse_config.seg_loss) + '_aug_' + str(parse_config.aug) + '/fold_' + str(
    parse_config.fold)

os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
writer = SummaryWriter('logs/{}/log'.format(exp_name))
save_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

EPOCHS = parse_config.n_epochs
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device_ids = range(torch.cuda.device_count())

torch.set_num_threads(8)

from dataset.datas import norm01, myDataset

dataset = myDataset(split='Train', aug=parse_config.aug)
dataset2 = myDataset(split='Valid', aug=False)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           shuffle=True)
# num_workers=2,
# pin_memory=True,
# drop_last=True)
# 初始化模型
# model = FAT_Net().cuda()
model = CCT_Net().cuda()
# 选用Adam优化器，传入模型参数，设置学习率
optimizer = optim.Adam(model.parameters(), lr=parse_config.lr_seg)

# 学习率衰减策略#余弦退火学习率:让学习率随epoch的变化图类似于cos，更新策略：
scheduler = CosineAnnealingLR(optimizer, T_max=20)
# ExponentialLR,指数衰减
# scheduler = ExponentialLR(optimizer, gamma=0.98)


# # 选用交叉熵损失函数
# creterion = nn.CrossEntropyLoss()
# # 设置batch数目，训练时设为随机
# dataloader = DataLoader(dataset1, parse_config.bt_size, shuffle=True)

##############################
# for epoch in range(parse_config.n_epochs):
#     # DataLoader是一个可迭代对象
#     for datas,label in dataloader:
#         optimizer.zero_grad()# DataLoader是一个可迭代对象
#         predict = model(datas)# 执行一次模型，计算输出结果
#         loss = creterion(predict, label)
#         loss.backward()#损失方向传播，完成待优化参数的梯度求解
#         optimizer.step()#参数更新
#     if (epoch+1) % 5 == 0: # 打印学习率 lr
#         with torch.no_grad():
#             train_predict = model()

def train(epoch):
    model.train()
    iteration = 0
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        # print("data的大小", datas.shape)
        label = batch_data['label'].cuda().float()
        # print("label的大小", label.shape)
        output = model(data)
        output = torch.sigmoid(output)
        dloss = dice_loss(output, label)
        ce_loss = F.binary_cross_entropy(output, label, reduction="mean")
        loss = 0.9*dloss + 0.1*ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration = iteration + 1

        if (batch_idx + 1) % 10 == 0:
            writer.add_scalar('loss/dc_loss', loss,
                              batch_idx + epoch * len(train_loader))

            writer.add_image('label', label[0],
                             batch_idx + epoch * len(train_loader))
            writer.add_image('output', output[0] > 0.5,
                             batch_idx + epoch * len(train_loader))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print("Iteration numbers: ", iteration)


val_loader = torch.utils.data.DataLoader(
    dataset2,
    batch_size=1,  # parse_config.bt_size
    shuffle=False,  # True
    num_workers=0,
    pin_memory=True,
    drop_last=False)  # True


def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    iou_value = 0
    dice_average = 0
    iou_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()

        with torch.no_grad():
            output = model(data)
            output = torch.sigmoid(output)
            output = output.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        # print("output.shape:", output.shape)
        # print("label.shape:", label.shape)
        # loss_dc = dice_loss(output, label)
        # print("loss_dc:", loss_dc)

        assert (output.shape == label.shape)
        dice_ave = dc(output, label)
        iou_ave = jc(output, label)
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1
    dice_average = dice_value / numm
    iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset1 = ", dice_average)
    print("Average iou value of evaluation dataset1 = ", iou_average)
    return dice_average, iou_average, loss


max_dice = 0
max_iou = 0
best_ep = 0

min_loss = 10
min_epoch = 0
loss = 0
# evaluation(0, val_loader)
for epoch in range(1, EPOCHS + 1):  # 打印学习率 lr

    this_lr = optimizer.state_dict()['param_groups'][0]['lr']
    writer.add_scalar('Learning Rate', this_lr, epoch)
    start = time.time()
    train(epoch)
    dice, iou, loss = evaluation(epoch, val_loader)
    # scheduler.step(loss)
    scheduler.step()

    if loss < min_loss:
        min_epoch = epoch
        min_loss = loss
    else:
        if epoch - min_epoch >= parse_config.patience:
            print('Early stopping!')
            break
    # if dice > max_dice:
    #    max_dice = dice
    #    best_ep = epoch
    #    torch.save(model.state_dict(), save_path)
    if iou > max_iou:
        max_iou = iou
        best_ep = epoch
        torch.save(model.state_dict(), save_path)
    else:
        if epoch - best_ep >= parse_config.patience:
            print('Early stopping!')
            break
    torch.save(model.state_dict(), latest_path)
    time_elapsed = time.time() - start
    print('Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))
