'''
    CAM of dataset classification model on D1/D2/D3/D4
'''

import os.path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models.VGG import vgg16_bn
from configs.paths_dict import PATHS
from data.dataset import my_dataset

from numpy import seterr
seterr(all='raise')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ds_name = 'D1'

def load_model(num_classes, model_weights):
    model = vgg16_bn(num_class=num_classes)
    checkpoints = torch.load(model_weights, map_location=DEVICE)
    model.load_state_dict(checkpoints['model_state_dict'])
    return model

# model
ds_weights = PATHS['ds_cls_ckpt']
ds_model = load_model(num_classes=4, model_weights=ds_weights)

ped_weights = PATHS['ped_cls_ckpt'][ds_name]
ped_model = load_model(num_classes=2, model_weights=ped_weights)

ds_model.eval()
ped_model.eval()

# data
ds_dataset = my_dataset([ds_name], txt_name='val.txt')
ds_loader = DataLoader(ds_dataset, batch_size=1, shuffle=False)


def get_CAM(features, model, visual_layer_name):
    features_flatten = None
    visual_flag = False
    for index, (name, module) in enumerate(model.features._modules.items()):
        if name != visual_layer_name and not visual_flag:
            features = module(features)
        elif name == visual_layer_name:
            features = module(features)
            visual_flag = True
        else:
            features_flatten = module(features if features_flatten is None else features_flatten)

    return features_flatten


# for i, (name, module) in enumerate(model.features._modules.items()):
#     print(f'{i} - {name} - {module}')

visual_layer_name = '40'

features_flatten = None
visual_flag = False

model = ped_model

for idx, data_dict in enumerate(tqdm(ds_loader)):
    visual_flag = False
    features_flatten = None

    img_path = data_dict['file_path'][0]
    img_name = data_dict['img_name'][0]

    # print('img_name:', img_name)
    image = data_dict['image'][0]
    image = torch.unsqueeze(image, dim=0)
    features = image

    for index, (name, module) in enumerate(model.features._modules.items()):
        if name != visual_layer_name and not visual_flag:
            features = module(features)
        elif name == visual_layer_name:
            features = module(features)
            visual_flag = True
        else:
            features_flatten = module(features if features_flatten is None else features_flatten)

    # dsCls_features_flatten = get_CAM(features, ds_model, visual_layer_name=visual_layer_name)
    # pedCls_features_flatten = get_CAM(features, ped_model, visual_layer_name=visual_layer_name)
    #
    # cam_diff = abs(dsCls_features_flatten - pedCls_features_flatten)
    # cam_diff_sum = cam_diff.sum()
    #
    # print(cam_diff.shape)
    # print(cam_diff_sum)

    features_flatten = torch.flatten(features_flatten, 1)
    out = model.classifier(features_flatten)
    pred = torch.argmax(out, dim=1).item()
    pred_class = out[:, pred]

    print('pred_class.shape:', pred_class.shape)
    print('features.shape:', features.shape)

    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

    grads = features_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]

    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    break   # don't del this

    # # plt.matshow(heatmap)
    # # plt.show()
    #
    # img = cv2.imread(img_path)  # 用cv2加载原始图像
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    # heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    # superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    # cam_name = os.path.join(r'C:\Users\wangj\Desktop\CAM\D1_val', img_name)
    # # cv2.imwrite(cam_name, superimposed_img)
    # # print(f'saved to {cam_name}')
    #
    # if idx > 144:
    #     break









