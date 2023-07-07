##用于处理图片.,设置大小
import cv2
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def image_process(d, images_path, labels_path, save_dir):
    dim = d
    image_dir_path = images_path
    mask_dir_path = labels_path

    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)

    image_path_list = list(filter(lambda x: x[-3:] == 'jpg', image_path_list))
    mask_path_list = list(filter(lambda x: x[-3:] == 'png', mask_path_list))

    image_path_list.sort()
    mask_path_list.sort()

    print(len(image_path_list), len(mask_path_list))

    # Dataset
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'jpg':
            print(image_path)
            assert os.path.basename(image_path)[:-4].split(
                '_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
            _id = os.path.basename(image_path)[:-4].split('_')[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)

            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

            save_dir_path = save_dir + '/Image1'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

            save_dir_path = save_dir + '/Label1'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)

def PH2_process(d, images_path, labels_path, save_dir):
    dim = d
    PH2_images_path = images_path
    mask_dir_path = labels_path
    print("mask_path:", mask_dir_path)

    image_path_list = os.listdir(PH2_images_path)
    mask_path_list = os.listdir(mask_dir_path)

    image_path_list = list(filter(lambda x: x[-3:] == 'bmp', image_path_list))
    mask_path_list = list(filter(lambda x: x[-3:] == 'bmp', mask_path_list))

    # Dataset
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'bmp':
            assert os.path.basename(image_path)[:-4] == os.path.basename(mask_path)[:-4].split('_')[0]
            _id = os.path.basename(image_path)[:-4][3]+os.path.basename(image_path)[:-4][4]+\
                  os.path.basename(image_path)[:-4][5]
            image_path = os.path.join(PH2_images_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)
            mask = mask[:, :, 0]

            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

            save_dir_path = save_dir + '/image'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

            save_dir_path = save_dir + '/label'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)
   
   
if __name__ == '__main__':

    for split in ['Train', 'Validation', 'Test']:
        #isic2016 or isic2017 or isic2018 or PH2
        image_path = r'../datas/{}/Image'.format(split)
        label_path =r'../datas/{}/Label'.format(split)
        save_path = r'../datas/{}/448x448'.format(split)

        D = (448, 448)
        image_process(D, image_path, label_path, save_path) #isic2016 or isic2017 or isic2018
        # PH2_process(D, image_path, label_path, save_path) #PH2

