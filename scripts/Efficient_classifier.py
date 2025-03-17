# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch
import torchvision.models as visionModels

from data.dataset import get_data
from training.training import train_model, train_ds_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, help='datasets that model is trained on')
    parser.add_argument('--task', type=str, choices=('ped_cls', 'ds_cls'), help='used to define the num_classes of model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', default=50, type=int)

    args = parser.parse_args()
    return args

args = get_args()
ds_name = args.ds_name
batch_size = args.batch_size
epochs = args.epochs
task = args.task

model_name = 'EfficientB0'
if task == 'ped_cls':
    num_classes = 2
else:
    num_classes = 4

# model
model = visionModels.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)

new_classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=num_classes)
)
model.classifier = new_classifier
print('Replacing classifier layer successfully!')

# # 若固定weights，则使用下面的代码，否则，注释掉
# for name, param in model.named_parameters():
#     # print(f'name: {name} - {param.shape} - {param.requires_grad}')
#     if name not in ['classifier.1.weight', 'classifier.1.bias']:
#         param.requires_grad = False
# print('Model class layer params grad fixed!')
# for name, param in model.named_parameters():
#     print(f'name: {name} - {param.shape} - {param.requires_grad}')


if task == 'ds_cls':
    # 数据集分类
    my_model = train_ds_model(model_name, model, batch_size, epochs)
    my_model.train()

else:
    # 行人分类
    ds_name_list = [ds_name]
    my_model = train_model(model_name, model, ds_name_list, batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)
    my_model.train()


# # data
# ds_name_list = list(['D1', 'D2', 'D3', 'D4'])
# path_key = 'org_dataset'
# txt_name = 'val.txt'
# batch_size = 8
# shuffle = False
#
# val_dataset, val_loader = get_data(ds_name_list, path_key, txt_name, batch_size, shuffle)


# for name, module in model._modules.items():
#     if name == 'avgpool' or name == 'classifier':
#         print('*' * 40)
#         print(f'Name: {name} \n{module}')
#         print('-' * 40)
#     else:
#         print(f'Name: {name}')

    # print('*' * 40)
    # print(name)
    # print(f'Name: {name} \n{module}')


#
# print('after setting:')
# for name, param in model.named_parameters():
#     print(f'name: {name} - {param.shape} - {param.requires_grad}')







