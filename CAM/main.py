# https://github.com/dongmo-qcq/FG-CAM/blob/master/main.py

import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2, os
from PIL import Image

from cam_utils import GradCAM, show_cam_on_image, center_crop_img
from VGG import vgg16_bn

from CAM import util
from CAM_Org.FG_CAM import FG_CAM
# from CAM.vgg import vgg16_bn

# from CAM_Beta.VGG import vgg16_bn

from CAM_Org.models.vgg import vgg16_bn



def forward_hook(self, input, output):
    self.X = input[0].detach()
    self.X.requires_grad = True

# 自定义一个网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(32, 64, 11),
            # nn.ReLU(),
            # nn.MaxPool2d(4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 43 * 43, 2)
        )

        self.hook = []

    def improve_resolution(self, I, target_layer):
        print('target_layer:', target_layer)
        for i in range(len(self.features)-1, target_layer, -1):
            I = self.features[i].IR(I)
        return I

    def register_hook(self):
        for m in self.features:
            m.register_forward_hook(forward_hook)

    def remove_hook(self):
        for m in self.hook:
            m.remove()
        self.hook = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 43 * 43)
        x = self.classifier(x)
        return x


# model = Net()
model = vgg16_bn(num_cls=4)

dsCls_weights_path = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth'
pedCls_weights_path = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D1-014-0.9740.pth'

checkpoints = torch.load(dsCls_weights_path, map_location='cpu')
model.load_state_dict(checkpoints['model_state_dict'])

# image_path = r'D:\my_phd\on_git\Stage5_Alpha\CAM\brno_00456_1.jpg'
image_path = r'D:\my_phd\dataset\Stage3\D1_ECPDaytime\nonPedestrian\zagreb_00516_1.jpg'
image = Image.open(image_path).convert('RGB')
image = np.array(image)
image = cv2.resize(image, (224, 224))
image = util.apply_transforms(image)

out = model(image)
print(out)

# fg_cam = FG_CAM(model, 'grad_cam')
# explanation, target_class = fg_cam(image, denoising=False,
#                                    target_layer=13,
#                                    target_class=1)
#
# explanation = torch.relu(explanation)
# explanation = util.visual_explanation(explanation)
#
# plt.imshow(explanation)
# plt.title('M1onD1_pedCls')
# plt.show()

# ----------------------------------
# dsCls_weights_path = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth'
# pedCls_weights_path = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D1-014-0.9740.pth'
#
# model = vgg16_bn(num_cls=4, pretrained=dsCls_weights_path)

# checkpoints = torch.load(pedCls_weights_path, map_location='cpu')
# model.load_state_dict(checkpoints['model_state_dict'])

# image_path = r'D:\my_phd\on_git\Stage5_Alpha\CAM\zagreb_00301_1.jpg'
# image = Image.open(image_path).convert('RGB')
# image = np.array(image)
# image = cv2.resize(image, (224, 224))
# image = util.apply_transforms(image)
#
# model.eval()
# # with torch.no_grad():
# out = model(image)
# _, pred = torch.max(out, 1)
# prob = torch.softmax(out, 1)
#
# print('prob:', prob)
# print(pred)
#
# fg_cam = FG_CAM(model, 'grad_cam')
# explanation, target_class = fg_cam(image, denoising=False,
#                                    target_layer=14,
#                                    target_class=0)
#
# explanation = torch.relu(explanation)
# explanation = util.visual_explanation(explanation)
#
# plt.imshow(explanation)
# plt.title('ds_cls')
# plt.show()





























