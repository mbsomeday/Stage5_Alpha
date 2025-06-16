# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from training.training import train_ped_model_alpha


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    return args


args = get_args()
batch_size = args.batch_size

model_obj = 'models.EfficientNet.efficientNetB0'
ds_name_list = ['D4']



tt = train_ped_model_alpha(model_obj=model_obj,
                           ds_name_list=ds_name_list,
                           batch_size=batch_size,
                           epochs=100,
                           warmup_epochs=3
                           )


















