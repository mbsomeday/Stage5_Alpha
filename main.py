# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from training.training_template import Ped_Classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_obj', type=str)
    parser.add_argument('--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--rand_seed', type=int)
    parser.add_argument('--ds_weights_path', type=str)
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--ped_weights_path', type=str)
    parser.add_argument('--data_key', type=str)
    parser.add_argument('--beta', type=float)

    args = parser.parse_args()
    return args

args = get_args()
model_obj = args.model_obj
ds_name_list = args.ds_name_list
batch_size = args.batch_size
epochs = args.epochs
ds_weights_path = args.ds_weights_path
isTrain = args.isTrain
ped_weights_path = args.ped_weights_path
beta = args.beta
warmup_epochs = args.warmup_epochs
rand_seed = args.rand_seed
data_key = args.data_key

tt = Ped_Classifier(model_obj,
                    ds_name_list=ds_name_list,
                    batch_size=batch_size,
                    epochs=epochs,
                    beta=beta,
                    ds_weights_path=ds_weights_path,
                    ped_weights_path=ped_weights_path,
                    isTrain=isTrain,
                    data_key=data_key,
                    resume=False,
                    warmup_epochs=warmup_epochs,
                    rand_seed=rand_seed
                    )
if isTrain:
    tt.train()
else:
    tt.test()

