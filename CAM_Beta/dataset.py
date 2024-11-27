import torch, os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class my_Dataset(Dataset):
    def __init__(self, ds_dir, txt_path, cls_label):
        self.ds_dir = ds_dir
        self.txt_path = txt_path
        self.cls_label = cls_label
        self.img_transfor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImgLabel()

    def __len__(self):
        return len(self.images)

    def init_ImgLabel(self):
        images, labels = [], []

        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_path)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            item = item.strip().split()
            image_path = os.path.join(self.ds_dir, item[0])

            labels.append(item[-1])
            images.append(image_path)

        return images, labels

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image_name_contents = self.images[idx].split(os.sep)
        image_name = os.path.join(image_name_contents[-3], image_name_contents[-2], image_name_contents[-1])
        image = self.img_transfor(image)
        label = np.array(self.labels[idx]).astype(np.int64)
        return image, label, self.cls_label, image_name



class dsCls_Dataset(Dataset):
    def __init__(self, ds_dir, txt_path, label):
        self.ds_dir = ds_dir
        self.txt_path = txt_path
        self.label = label
        self.img_transfor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImgLabel()

    def __len__(self):
        return len(self.images)

    def init_ImgLabel(self):
        images, labels = [], []

        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_path)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            item = item.strip().split()
            image_path = os.path.join(self.ds_dir, item[0])

            images.append(image_path)
            labels.append(self.label)

        return images, labels

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.img_transfor(image)
        label = np.array(self.labels[idx]).astype(np.int64)
        return image, label


if __name__ == '__main__':
    ds_dir = r'D:\my_phd\dataset\Stage3\D1_ECPDaytime'
    txt_path = 'test.txt'
    # train_dataset = pedCs_Dataset(ds_dir, txt_path)
    train_dataset = my_Dataset(ds_dir, txt_path, cls_label=0)
    train_loader = DataLoader(train_dataset, batch_size=4)

    # for image, label in train_loader:
    #     print(label)
    #     break
















