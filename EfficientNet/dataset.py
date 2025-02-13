import numpy as np
import torch.cuda
from PIL import Image
from torchvision import transforms
import os, random
from torch.utils.data import Dataset, DataLoader
import argparse
import sys, os
from efficientnet_pytorch import EfficientNet



class pedCls_Dataset(Dataset):
    '''
        读取多个数据集的数据
    '''

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



if __name__ == '__main__':
    # from time import time
    # d2_path = r'D:\my_phd\dataset\Stage3\D2_CityPersons'
    # d2 = pedCls_Dataset(ds_dir=d2_path, txt_name='test.txt')
    # d2_loader = DataLoader(d2, batch_size=64)
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientNet.from_name('efficientnet-b5', num_classes=2)
    # start_time = time()
    # for img, label, image_name in d2_loader:
    #     print(img.shape)
    #     # print(label)
    #     out = model(img)
    #     # print(out)
    #
    #     duration = time() - start_time
    #     print(duration)
    #     break

    from torchsummary import summary
    summary(model, (3, 224, 224))

    # print(model)
















