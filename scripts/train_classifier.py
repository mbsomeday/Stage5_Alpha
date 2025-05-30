# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

# from training.training import train_ped_model_alpha, train_ds_model_alpha
from utils.utils import get_gpu_info
from training.train_pedCls_CAMLoss import PedCls_with_camLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', default=50, type=int)
    # parser.add_argument('--save_best_cls', default=False, type=bool, help='to decide whether to save the best models for each class')
    parser.add_argument('--ds_weights', type=str)

    parser.add_argument('-m', '--model_obj', default='models.VGG.vgg16_bn', type=str)
    parser.add_argument('-d', '--ds_name', type=str, help='datasets that model is trained on, ds_cls task do not need this param')
    # parser.add_argument('-r', '--reload', default=None)

    args = parser.parse_args()
    return args

args = get_args()
model_obj = args.model_obj
ds_name = args.ds_name
batch_size = args.batch_size
ds_weights_path = args.ds_weights
epochs = args.epochs

# reload = args.reload
# ped_weights = args.ped_weights

ds_name_list = [ds_name]


# 打印当前使用的gpu信息
get_gpu_info()


# ds_weights_path = r'C:\Users\wangj\Desktop\efficientB0\efficientB0_dsCls\efficientNetB0_dsCls-10-0.97636.pth'

pp = PedCls_with_camLoss(model_obj=model_obj,
                         ds_name_list=ds_name_list,
                         batch_size=4,
                         epochs=epochs,
                         ds_weights=ds_weights_path,
                         mode='train')

pp.train_model()

# num_classes = 2
# model_name = 'vgg16bn'
# model = vgg16_bn(num_class=2)

# my_model = train_pedmodel_camLoss(model_name, model, ds_name_list, camLoss_coefficient=0.1,
#                                   batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)

# my_training = train_ped_model_alpha(model_obj=model_obj, ds_name_list=ds_name_list, batch_size=batch_size, reload=reload,
#                                     save_prefix=None, epochs=epochs, base_lr=0.01, warmup_epochs=2, lr_patience=5,
#                                     camLoss_coefficient=camLoss_coefficient, save_best_cls=save_best_cls, gen_img=False)
# my_training.train_model()



# 行人分类baseline,此时无camloss
# ped_training = train_ped_model_alpha(model_obj=model_obj, ds_name_list=ds_name_list, batch_size=batch_size,
#                                         reload=reload, epochs=epochs, base_lr=0.01, warmup_epochs=5, lr_patience=5,
#                                         camLoss_coefficient=None
#                                         )
#
# ped_training.train_model()
# # 行人分类 cam loss训练
# ped_training = train_ped_model_alpha(model_obj=model_obj, ds_name_list=ds_name_list, batch_size=batch_size,
#                                         reload=reload, epochs=epochs, base_lr=0.01, warmup_epochs=5, lr_patience=5,
#                                         camLoss_coefficient=0.2, ds_model_obj=model_obj
#                                         )
# ped_training.train_model()

# ped_test = test_ped_model_alpha(model_obj=model_obj, ped_weights=ped_weights, ds_name_list=ds_name_list, batch_size=batch_size, camLoss_coefficient=0.2, txt_name=txt_name)
#
# ped_test.test_model()

# # 数据集分类训练
# ds_training = train_ds_model_alpha(model_obj=model_obj, batch_size=batch_size, ds_name_list=['D1', 'D2', 'D3', 'D4'], epochs=epochs, base_lr=0.01, warmup_epochs=5, lr_patience=5)
#
# ds_training.train_model()

















