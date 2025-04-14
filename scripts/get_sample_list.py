# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torchvision import models
from tqdm import tqdm

from data.dataset import my_dataset, get_data
from utils.utils import DEVICE, load_model
from configs.paths_dict import PATHS


ds_name = 'D4'
path_key = 'org_dataset'
txt_name = 'test.txt'
batch_size = 1
shuffle = False

ped_model = models.efficientnet_b0(weights=None, num_classes=2)
# ped_weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0D3-018-0.941520.pth'
ped_weights_path = PATHS['EfficientNet_ped_cls'][ds_name]
ped_model = load_model(ped_model, ped_weights_path)

ds_model = models.efficientnet_b0(weights=None, num_classes=4)
ds_weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0_dsCls-015-0.880432.pth'
ds_model = load_model(ds_model, ds_weights_path)

ped_model.eval()
ds_model.eval()

test_dataset, test_loader = get_data(ds_name_list=[ds_name], path_key=path_key, txt_name=txt_name, batch_size=batch_size, shuffle=shuffle)

pedR_dsR = []
pedW_dsR = []
pedW_dsW = []
pedR_dsW = []

with torch.no_grad():
    for idx, data_dict in enumerate(tqdm(test_loader)):
        image = data_dict['image'].to(DEVICE)
        ds_label = data_dict['ds_label'].to(DEVICE)
        ped_label = data_dict['ped_label'].to(DEVICE)
        img_path = data_dict['img_path']

        ds_out = ds_model(image)
        ds_pred = torch.argmax(ds_out, dim=1)

        ped_out = ped_model(image)
        ped_pred = torch.argmax(ped_out, dim=1)

        if ds_pred == ds_label:
            if ped_pred == ped_label:
                pedR_dsR.extend(img_path)
            else:
                pedW_dsR.extend(img_path)
        else:
            if ped_pred == ped_label:
                pedR_dsW.extend(img_path)
            else:
                pedW_dsW.extend(img_path)

        # print('ds_label:', ds_label)
        # print('ds_pred:', ds_pred)
        #
        # print('-' * 50)
        # print('ped label:', ped_label)
        # print('ped_pred:', ped_pred)

        # break


save_base = r'D:\my_phd\on_git\Stage5_Alpha\scripts\EfficientNet'
save_dir = os.path.join(save_base, str(ds_name+'_test'))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_dsR_pedR = os.path.join(save_dir, 'pedR_dsR.txt')
file_dsR_pedW = os.path.join(save_dir, 'pedW_dsR.txt')
file_dsW_pedW = os.path.join(save_dir, 'pedW_dsW.txt')
file_dsW_pedR = os.path.join(save_dir, 'pedR_dsW.txt')

def save_txt(file_path, data):
    with open(file_path, 'a') as f:
        for item in data:
            msg = item + '\n'
            f.write(msg)

save_txt(file_dsR_pedR, pedR_dsR)
save_txt(file_dsR_pedW, pedW_dsR)
save_txt(file_dsW_pedW, pedW_dsW)
save_txt(file_dsW_pedR, pedR_dsW)

















