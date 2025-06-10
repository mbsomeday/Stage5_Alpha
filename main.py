# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from training.training_template import Ped_Classifier
from utils.utils import DotDict
from configs.pedCls_args import TrainArgs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ped_model_obj', type=str)
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_key', type=str, default='tiny_dataset')
    parser.add_argument('--ds_weights_path', type=str, default=None)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--ped_weights_path', type=str)
    parser.add_argument('--ds_model_obj', type=str)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rand_seed', type=int, default=1)

    args = parser.parse_args()
    return args


# args = get_args()
#
# basic_args = {
#     'ped_model_obj': args.ped_model_obj,
#     'ds_name_list': args.ds_name_list,
#     'data_key': args.data_key,
#     'batch_size': args.batch_size,
#     'isTrain': args.isTrain,
#
# }
#
# train_args = {
#     'epochs': args.epochs,
#     'beta': args.beta,
#     'ds_model_obj': args.ds_model_obj,
#     'ds_weights_path': args.ds_weights_path,
#     'resume': args.resume,
#
# }
#
# isTrain = args.isTrain
# ped_weights_path = args.ped_weights_path
# warmup_epochs = args.warmup_epochs
# rand_seed = args.rand_seed
# data_key = args.data_key
# base_lr = args.base_lr
#
# basic_args = DotDict(basic_args)
# train_args = DotDict(train_args)

opts = TrainArgs().parse()

tt = Ped_Classifier(opts=opts)

# tt.train()

# tt.test()

