# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch

from models.VGG import vgg16_bn
from data.dataset import my_dataset, get_data
from utils.utils import get_ds_model, get_orgPed_model


ds_name = 'D1'
path_key ='org_dataset'
txt_name ='test.txt'
batch_size = 6
shuffle = False

ds_model = get_ds_model()
ped_model = get_orgPed_model(ds_name)

test_dataset, test_loader = get_data(ds_name_list=[ds_name], path_key=path_key, txt_name=txt_name, batch_size=batch_size, shuffle=shuffle)

dsR_pedR = []
dsR_pedW = []
dsW_pedW = []
dsW_pedR = []

for idx, data_dict in enumerate(test_loader):
    image = data_dict['image']
    ds_label = data_dict['ds_label']
    ped_label = data_dict['ped_label']

    ds_out = ds_model(image)
    ds_pred = torch.argmax(ds_out, dim=1)

    ped_out = ped_model(image)
    ped_pred = torch.argmax(ped_out, dim=1)

    print('ds_label:', ds_label)
    print('ds_pred:', ds_pred)

    print('-' * 50)
    print('ped label:', ped_label)
    print('ped_pred:', ped_pred)

    break



























