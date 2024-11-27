import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

from cam_utils import GradCAM, show_cam_on_image, center_crop_img
from VGG import vgg16_bn


if __name__ == '__main__':
    # model = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
    # model.eval()
    # target_layers = [model.layer4]

    model = vgg16_bn(4)
    weights_path = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth'
    checkpoints = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    target_layers = [model.features]
    # target_layers = [model.layer4]

    # target_category = 282  # tabby, tabby cat
    # image_path = r'D:\my_phd\on_git\Stage5\cat_dog.png'
    # image_path = r'D:\my_phd\on_git\Stage5\tabby_cat.jpg'
    # image_path = r'D:\my_phd\on_git\Stage5\cock.png'
    # image_path = r'D:\my_phd\on_git\Stage5\cat.png'

    # 检测行人数据集分类
    target_category = 3
    image_path = r'D:\my_phd\on_git\Stage5\0a2cc187-8d6ab554_1.jpg'

    image = Image.open(image_path).convert('RGB')
    image = np.array(image, dtype=np.uint8)
    print('img.shape:', image.shape)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor = image_transform(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)

    with torch.no_grad():
        out = model(input_tensor)
        _, pred = torch.max(out, 1)
        print(pred)

    cam = GradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    # print('gray', grayscale_cam.shape)
    # print(grayscale_cam * 255.0)
    visualization = show_cam_on_image(image.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()












































