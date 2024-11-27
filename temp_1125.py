import torch

from VGG import vgg16_bn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = vgg16_bn(num_class=2)
weights_path = r'/kaggle/input/stage4-baseline-weights/vgg16bn-D1-014-0.9740.pth'
checkpoints = torch.load(weights_path, map_location=DEVICE)
model.load_state_dict(checkpoints['model_state_dict'])
model.to(DEVICE)


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class pedCls_Dataset(Dataset):
    '''
        读取多个数据集的数据
    '''

    # def __init__(self, runOn, ds_name_list, txt_name):
    def __init__(self, ds_dir, txt_name):
        self.ds_dir = ds_dir
        self.txt_name = txt_name
        self.image_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        '''
            读取图片 和 label
        '''
        images = []
        labels = []


        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        for line in data:
            line = line.replace('\\', os.sep)
            line = line.strip().split()
            image_path = os.path.join(self.ds_dir, line[0])
            label = line[-1]
            images.append(image_path)
            labels.append(label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        img = self.image_transformer(img)
        return img, label, image_name













