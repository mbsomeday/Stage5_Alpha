# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch, os, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from data.dataset import my_dataset
from models.VGG import vgg16_bn
from configs.paths_dict import PATHS
from utils.utils import plot_cm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_on', type=str)
    parser.add_argument('-d', '--ds_name', type=str, help='dataset that the model is tested on')
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
            # break

        test_accuracy = correct_num / len(test_dataset)
        bc = balanced_accuracy_score(y_true, y_pred)

        print(f'test_accuracy: {test_accuracy} - balanced accuracy: {bc}  \n{correct_num}/{len(test_dataset)}')


        # 绘制混淆矩阵
        label_names = ['ped', 'nonPed']
        title = f'Ped Cls CM on {ds_name}'
        plot_cm(y_true, y_pred, label_names, title=title)



if __name__ == '__main__':

    args = get_args()
    train_on = args.train_on

    ds_name = args.ds_name
    batch_size = args.batch_size
    ds_key_name = args.ds_key_name
    txt_name = args.txt_name
    # print(ds_key_name)


    model = vgg16_bn(num_class=2).to(DEVICE)
    weights_path = PATHS['ped_cls_ckpt'][train_on]
    print(f"Reload model {weights_path}")
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_dataset = my_dataset(ds_name_list=[ds_name], txt_name=txt_name, key_name=ds_key_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ped_test(model, ds_name=ds_name, test_dataset=test_dataset, test_loader=test_loader)



























