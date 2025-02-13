import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from cam_utils import GradCAM, show_cam_on_image
from models.VGG import vgg16_bn

from CAM import util
from CAM.FG_CAM import FG_CAM


def reload_clsModel(weights_path, num_cls):
    model = vgg16_bn(num_cls)

    checkpoints = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()

    return model


pedCls_weights = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D4-013-0.9502.pth'
dsCls_weights = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth'

pedCls_model = reload_clsModel(pedCls_weights, 2)
dsCls_model = reload_clsModel(dsCls_weights, 4)

# Best available weights (currently alias for IMAGENET1K_V2)
# Note that these weights may change across versions
# model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model = util.get_model('vgg16_bn')


def show_CAM(image_path, target_category, model):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image, dtype=np.uint8)
    target_layers = [model.features]
    print('img.shape:', image.shape)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = image_transform(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)

    with torch.no_grad():
        out = model(input_tensor)
        logits = torch.softmax(out, 1)
        _, pred = torch.max(out, 1)
        print('预测为：', pred)

    # 以下为显示CAM
    cam = GradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


# image_path = r'D:\my_phd\on_git\Stage5\0a172b0e-2af0d158_1.jpg'
# image_path = r'D:\my_phd\on_git\Stage5\0c58af71-de7d28b7_1.jpg'
image_path = r'D:\my_phd\on_git\Stage5\5_hens.jpeg'
# show_CAM(image_path, target_category=1, model=pedCls_model)
#
# show_CAM(image_path, target_category=3, model=dsCls_model)
# show_CAM(image_path, target_category=3, model=model)


# 用Fine-Grained CAM
fg_cam = FG_CAM(model, 'score_cam')

input_image = Image.open(image_path).convert('RGB')
image = np.array(input_image)
image = cv2.resize(image, (224, 224))
input = util.apply_transforms(input_image)

model.eval()
with torch.no_grad():
    out = model(input)
    _, pred = torch.max(out, 1)
    print(pred)

explanation, target_class = fg_cam(input, denoising=False,
                                   target_layer=13,
                                   target_class=8)
explanation = torch.relu(explanation)
explanation = util.visual_explanation(explanation)

c = 2

plt.imshow(explanation)
plt.title('score_cam_DenoiseFalse')
plt.show()


