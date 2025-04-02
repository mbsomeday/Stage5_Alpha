# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from models.VGG import vgg16_bn
from training.training import train_ped_model_alpha


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_obj', default='models.VGG.vgg16_bn', type=str)
    parser.add_argument('--ds_name', type=str, help='datasets that model is trained on, ds_cls task do not need this param')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-r', '--reload', default=None)
    parser.add_argument('--cam_loss', type=float)

    args = parser.parse_args()
    return args

args = get_args()
model_obj = args.model_obj
ds_name = args.ds_name
batch_size = args.batch_size
reload = args.reload
epochs = args.epochs
camLoss_coefficient = args.cam_loss if args.cam_loss > 0 else None

# num_classes = 2
# model_name = 'vgg16bn'
# model = vgg16_bn(num_class=2)
ds_name_list = [ds_name]

# my_model = train_pedmodel_camLoss(model_name, model, ds_name_list, camLoss_coefficient=0.1,
#                                   batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)

my_training = train_ped_model_alpha(model_obj=model_obj, ds_name_list=ds_name_list, batch_size=batch_size, reload=reload,
                                    save_prefix=None, epochs=epochs, base_lr=0.01, warmup_epochs=2, lr_patience=4,
                                    camLoss_coefficient=camLoss_coefficient, gen_img=False
                                    )
my_training.train_model()


















