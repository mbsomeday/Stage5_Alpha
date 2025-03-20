# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from torchvision import models
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import my_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_weights(model, weights):
    ckpt = torch.load(weights, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

def get_models():
    print(f'DEVICE:{DEVICE}')
    if DEVICE == 'cuda':
        ds_weights = r'/kaggle/input/temp-effb0ds/EfficientB0_dsCls-028-0.991572.pth'
    else:
        ds_weights = r'D:\chrom_download\EfficientB0_dsCls-028-0.991572.pth'

    # ped_model = models.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
    # new_classifier = torch.nn.Sequential(
    #     torch.nn.Dropout(p=0.2, inplace=True),
    #     torch.nn.Linear(in_features=1280, out_features=2)
    # )
    # ped_model.classifier = new_classifier
    # load_weights(ped_model, ped_weights)

    ds_model = models.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
    new_classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=4)
    )
    ds_model.classifier = new_classifier
    load_weights(ds_model, ds_weights)

    return ds_model


class GradCAM(nn.Module):
    def __init__(self, grad_layer):
        super(GradCAM, self).__init__()
        self.ds_model = get_models()
        # self.ds_model.eval()
        self.ds_model.to(DEVICE)

        self.grad_layer = grad_layer

        self.feed_forward_features = None
        self.backward_features = None

        self._register_hooks(self.ds_model, self.grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

        # 定义mask的loss
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def _register_hooks(self, model, grad_layer):
        '''
            注册钩子函数
        '''
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Register forward hook and backward hook! Hooked layer: {self.grad_layer}")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def plt_format(self, x):
        '''
            x是 4-D
        '''
        ret_x = x[0].detach().numpy()
        ret_x = np.transpose(ret_x, (1, 2, 0))
        return ret_x

    def calc_cam(self, model, x):
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        model.zero_grad()

        grad_yc = logits[0, pred]
        # print(f'grad_yc: {grad_yc}')
        grad_yc.backward()
        model.zero_grad()
        # print(f'backward之后： {self.backward_features.shape}')

        w = F.adaptive_avg_pool2d(self.backward_features, 1)    # shape: (batch_size, 1280, 1, 1)
        # print(f'w: {w.shape}')
        temp_w = w[0].unsqueeze(0)
        temp_fl = self.feed_forward_features[0].unsqueeze(0)
        ac = F.conv2d(temp_fl, temp_w)
        ac = F.relu(ac)

        Ac = F.interpolate(ac, (224, 224))

        heatmap = Ac

        # 获取mask
        Ac_min = Ac.min()
        Ac_max = Ac.max()

        mask = heatmap.detach().clone()
        mask.requires_grad = False
        mask[mask<Ac_max] = 0
        masked_image = x - x * mask

        return heatmap, mask, masked_image


    def forward(self, model, x, labels):

        # 计算ds_model
        out = self.ds_model(x)
        pred = torch.argmax(out, dim=1)
        # print(f'ds pred: {pred} - {out} - {torch.softmax(out, dim=1)}')

        masked_images = np.ones(shape=x.shape)
        for img_idx, image in enumerate(x):
            image = torch.unsqueeze(image, dim=0)
            print(f'image: {image.shape}')
            # heatmap, mask, masked_image = self.attloss(image)
            heatmap, mask, masked_image = self.calc_cam(model, image)
            print('flag after mask computing')
            # masked_images[img_idx] = masked_image
            masked_images[img_idx] = masked_image

        masked_images = torch.tensor(masked_images)

        out = model(masked_images)
        loss = self.loss_fn(out, labels)

        # ds_cam, ds_mask, ds_masked_image = self.calc_cam(self.ds_model, x)

        return loss




















