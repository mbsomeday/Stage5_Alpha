# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch

from training.training import train_ped_model_alpha
from test.ds_cls import test_ds_classifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--ds_name', type=str, help='datasets that model is trained on, ds_cls task do not need this param')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weights_path', type=str)
    # parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model_obj', type=str)

    # parser.add_argument('--reload', default=None, type=str)
    # parser.add_argument('--task', type=str, choices=('ped_cls', 'ds_cls'), help='used to define the num_classes of model')

    args = parser.parse_args()
    return args


args = get_args()
weights_path = args.weights_path
batch_size = args.batch_size
# ds_name = args.ds_name
# epochs = args.epochs
# reload = args.reload
model_obj = args.model_obj

# ds_name_list = [ds_name]


# model_obj = 'models.EfficientNet.efficientNetB0'
# weights_path = r'C:\Users\wangj\Desktop\efficientB0\efficientB0_dsCls\efficientNetB0_dsCls-10-0.97636.pth'

test_ds_classifier(model_obj=model_obj, weights_path=weights_path, batch_size=batch_size)


# my_train = train_ped_model_alpha(model_obj=model_obj, ds_name_list=ds_name_list, batch_size=batch_size)
# my_train.train_model()
# my_train.val_on_epoch_end(epoch=-1)

# new_classifier = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2, inplace=True),
#     torch.nn.Linear(in_features=1280, out_features=num_classes)
# )
# model.classifier = new_classifier

# print('Replacing classifier layer successfully!')


# # 若固定weights，则使用下面的代码，否则，注释掉
# for name, param in model.named_parameters():
#     # print(f'name: {name} - {param.shape} - {param.requires_grad}')
#     if name not in ['classifier.1.weight', 'classifier.1.bias']:
#         param.requires_grad = False
# print('Model class layer params grad fixed!')
# for name, param in model.named_parameters():
#     print(f'name: {name} - {param.shape} - {param.requires_grad}')


# if task == 'ds_cls':
#     # 数据集分类
#     # my_model = train_ds_model(model_name, model, batch_size, epochs, reload=reload)
#     # my_model.train_model()
#     pass
#
#
# else:
#     # 行人分类
#     ds_name_list = [ds_name]
#     #  model_obj: str, ds_name_list, batch_size, epochs=50, reload=None, base_lr=0.01, warmup_epochs=0, lr_patience=5
#     my_model = train_ds_model_alpha(model_name, model, ds_name_list, batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)
#     # my_model = train_pedmodel_camLoss(model_name, model, ds_name_list, batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False, reload=reload)
#     my_model.train_model()



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







