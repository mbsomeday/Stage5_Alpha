# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch, os, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import my_dataset
from models.VGG import vgg16_bn
from configs.paths_dict import PATHS
from utils.utils import plot_cm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--ds_key_name', type=str)
    parser.add_argument('--txt_name', type=str)

    args = parser.parse_args()
    return args


def ped_test(model, ds_name, test_dataset, test_loader):
    model.eval()

    correct_num = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict['image'].to(DEVICE)
            ped_labels = data_dict['ped_label'].to(DEVICE)

            ped_out = model(images)
            ped_pred = torch.argmax(ped_out, dim=1)

            correct_num += (ped_pred == ped_labels).sum()

            y_true.extend(ped_labels.cpu().numpy())
            y_pred.extend(ped_pred.cpu().numpy())

        test_accuracy = correct_num / len(test_dataset)
        print(f'test_accuracy: {test_accuracy.item()} - {correct_num}/{len(test_dataset)}')

        # 绘制混淆矩阵
        label_names = ['ped', 'nonPed']
        title = f'Ped Cls CM on {ds_name}'
        plot_cm(y_true, y_pred, label_names, title=title)



if __name__ == '__main__':

    args = get_args()
    ds_name = args.ds_name
    batch_size = args.batch_size
    ds_key_name = args.ds_key_name
    txt_name = args.txt_name
    # print(ds_key_name)

    test_dataset = my_dataset(ds_name_list=[ds_name], txt_name=txt_name, key_name=ds_key_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = vgg16_bn(num_class=2).to(DEVICE)
    print(f"Reload model {PATHS['ped_cls_ckpt'][ds_name]}")
    ckpt = torch.load(PATHS['ped_cls_ckpt'][ds_name], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)

    ped_test(model, ds_name=ds_name, test_dataset=test_dataset, test_loader=test_loader)



























