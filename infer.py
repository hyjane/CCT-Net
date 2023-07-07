import os, argparse, sys, tqdm, logging, cv2
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import torch.nn.functional as F
from medpy.metric.binary import hd, hd95, dc, jc, assd
from utils import load_model
from CCT import CCT_Net
from dataset.datas import norm01, myDataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms
parser = argparse.ArgumentParser()
parser.add_argument('--log_name',
                    type=str,
                    default='_1_0_loss_0_aug_1')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--dataset1', type=str, default='datas')
parser.add_argument('--arch', type=str, default='CCT')
parser.add_argument('--net_layer', type=int, default=50)
# pre-Train
parser.add_argument('--pre', type=int, default=0)
# transformer
parser.add_argument('--trans', type=int, default=1)
# cross-scale framework
parser.add_argument('--cross', type=int, default=0)
parse_config = parser.parse_args()

#保存测试分割结果
# parser.add_argument('--save_path', type=str)

print(parse_config)

os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dir_path = os.path.dirname(
    os.path.abspath(__file__)) + "/logs/{}/{}/fold_{}/".format(
        parse_config.dataset, parse_config.log_name, parse_config.fold)

model = CCT_Net().cuda()
model = load_model(model, dir_path + 'model/best.pkl')

# logging
txt_path = os.path.join(dir_path + 'parameter.txt')

logging.basicConfig(filename=txt_path,
                    level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s',
                    datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


dataset = myDataset('Test', aug=False)
print("图片名称：", dataset.image_paths[0])
print("length of images:", len(dataset))
length = len(dataset)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          pin_memory=True,
                                          drop_last=False,
                                          shuffle=False)
save_path = r'C:\Users\LYY\PycharmProjects\pythonProject\skin_lesion_segmentation\ourslves\CCTModule\datas\output2016'


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def test():
    model.eval()
    num = 0
    dice_value = 0
    jc_value = 0
    hd95_value = 0
    assd_value = 0

    labels = []
    pres = []

    for batch_idx, img in tqdm(enumerate(test_loader)):
        print("patch_idx=", batch_idx)
        data = img['image'].to(device).float()
        label = img['label'].to(device).float()
        with torch.no_grad():
            output = model(data)
            output1 = torch.sigmoid(output)
            output = output1.cpu().numpy() > 0.5
            #保存结果:->
            probs = output1.squeeze(0)
            tf = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.ToTensor()
                ]
            )
            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()
            full_mask = (full_mask > 0.5)
            result = mask_to_image(full_mask)
            result.save('%s/%03d.png' % (save_path, batch_idx))
            # cv2.imwrite('%s/%03d.png' % (save_path, batch_idx), result)
            print('%s/%03d.png' % (save_path, batch_idx))
            #<- 保存结果
        label = label.cpu().numpy()

        assert (output.shape == label.shape)
        labels.append(label)
        pres.append(output)
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)
    print(labels.shape, pres.shape)

    for _id in range(labels.shape[0]):
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id])
        try:
            hd95_ave = hd95(labels[_id], pres[_id])
            assd_ave = assd(labels[_id], pres[_id])
        except RuntimeError:
            num += 1
            hd95_ave = 0
            assd_ave = 0

        dice_value += dice_ave
        jc_value += jc_ave
        hd95_value += hd95_ave
        assd_value += assd_ave

    dice_average = dice_value / (labels.shape[0] - num)
    jc_average = jc_value / (labels.shape[0] - num)
    hd95_average = hd95_value / (labels.shape[0] - num)
    assd_average = assd_value / (labels.shape[0] - num)

    logging.info('Dice value of Test dataset1  : %f' % (dice_average))
    logging.info('Jc value of Test dataset1  : %f' % (jc_average))
    logging.info('Hd95 value of Test dataset1  : %f' % (hd95_average))
    logging.info('Assd value of Test dataset1  : %f' % (assd_average))

    print("Average dice value of evaluation dataset1 = ", dice_average)
    print("Average jc value of evaluation dataset1 = ", jc_average)
    print("Average hd95 value of evaluation dataset1 = ", hd95_average)
    print("Average assd value of evaluation dataset1 = ", assd_average)
    return dice_average


if __name__ == '__main__':
    test()
