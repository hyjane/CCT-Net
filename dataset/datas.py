#将图片转换为数组格式保存.npy文件

import os
import glob
import random
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F

import albumentations as A


def norm01(x):
    return np.clip(x, 0, 255) / 255


def filter_image(p):
    label_data = np.load(p.replace('image', 'label'))
    return np.max(label_data) == 1


class myDataset(data.Dataset):
    def __init__(self, split, aug=False):
        super(myDataset, self).__init__()

        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        root_dir = r'./datas'
        if split == 'Train':
            self.image_paths = glob.glob(
                root_dir + '/Train/Image/*.npy')  # glob.glob()全提函数，获取root_dir+'/Train/Image/'的所有.npy函数
            self.label_paths = glob.glob(root_dir + '/Train/Label/*.npy')
        elif split == 'Valid':
            self.image_paths = glob.glob(root_dir + '/Validation/Image/*.npy')
            self.label_paths = glob.glob(root_dir + '/Validation/Label/*.npy')
        elif split == 'Test':
            self.image_paths = glob.glob(root_dir + '/Test/Image/*.npy')
            self.label_paths = glob.glob(root_dir + '/Test/Label/*.npy')
        self.image_paths.sort()  # 排序
        self.label_paths.sort()

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug

        # 数据增强处理
        self.transf = A.Compose([
            A.HorizontalFlip(p=0.5),  # 以p的概率水平翻转（p=0.5，一半的概率翻转，一半的概率不翻转）
            A.VerticalFlip(p=0.5),  # 垂直翻转
            A.RandomBrightnessContrast(p=0.2),  # 随机增强亮度核对比度。
            A.Rotate()  # 旋转
        ])

    def __getitem__(self, index):

        image_data = np.load(self.image_paths[index])
        # print("image-path:", self.image_paths)
        label_data = np.load(self.label_paths[index]) > 0.5

        if self.aug:
            mask = np.concatenate([  # concatenate，特征融合？？？
                label_data[..., np.newaxis].astype('uint8'),
            ],
                axis=-1)
            #             print(mask.shape)
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)
            image_data, mask_aug = tsf['image'], tsf['mask']
            label_data = mask_aug[:, :, 0]

        image_data = norm01(image_data)
        label_data = np.expand_dims(label_data, 0)
        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        # print("image_data.shape=",image_data.shape)
        image_data = image_data.permute(2, 0, 1)
        return {
            'image_path': self.image_paths[index],
            'label_path': self.label_paths[index],
            'image': image_data,
            'label': label_data
        }

    def __len__(self):
        return self.num_samples