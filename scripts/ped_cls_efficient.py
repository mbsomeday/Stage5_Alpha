import torchvision.models as visionModels
import argparse

from data.dataset import get_data
from training.training import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name_list', nargs='+', type=list, help='datasets that model is trained on')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', default=50, type=int)

    args = parser.parse_args()
    return args

args = get_args()
ds_name_list = args.ds_name_list
batch_size = args.batch_size
epochs = args.epochs

model_name = 'EfficientB0'

# model
model = visionModels.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)


my_model = train_model(model_name, model, ds_name_list, batch_size=batch_size, epochs=epochs, save_prefix=None, gen_img=False)



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

# print('------- Below is named parameters -------')
# for name, param in model.named_parameters():
#     print(f'name: {name} - {param.shape} - {param.requires_grad}')
#     if name == 'classifier.1.weight' or name == 'classifier.1.bias':
#         param.requires_grad = False
#
# print('after setting:')
# for name, param in model.named_parameters():
#     print(f'name: {name} - {param.shape} - {param.requires_grad}')







