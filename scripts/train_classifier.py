# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)


import argparse

from models.VGG import vgg16_bn
from training.training import train_ped_model, train_ds_model, train_pedmodel_camLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, help='datasets that model is trained on, ds_cls task do not need this param')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-g', '--gid', type=str, help='to set the id of gpu')

    args = parser.parse_args()
    return args

args = get_args()
ds_name = args.ds_name
batch_size = args.batch_size
epochs = args.epochs
if args.gid is not None:
    gid = args.gid
    os.environ['CUDA_VISIBLE_DEVICES'] = gid

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 2
model_name = 'vgg16bn'
model = vgg16_bn(num_class=2)
ds_name_list = [ds_name]

my_model = train_pedmodel_camLoss(model_name, model, ds_name_list, batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)
my_model.train_model()


















