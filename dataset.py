import os, sys

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class CIFARGroup_DSCls(Dataset):
    def __init__(self, group_dir):
        self.group_dir = group_dir
        self.image_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        images = []
        labels = []

        group_list = os.listdir(self.group_dir)

        for cur_group in group_list:
            cur_label = str(int(cur_group[-1]) - 1)

            group_path = os.path.join(self.group_dir, cur_group)
            image_list = list(os.path.join(group_path, img_path) for img_path in os.listdir(group_path))
            images.extend(image_list)

            for i in range(len(image_list)):
                labels.append(cur_label)

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
    group_dir = r'D:\my_phd\dataset\CIFAR10\groups'
    cifar_dataset = CIFARGroup_DSCls(group_dir)
    cifar_loader = DataLoader(cifar_dataset, batch_size=4, shuffle=True)
    for images, labels, names in cifar_loader:
        print(labels)
        print(names)
        break










