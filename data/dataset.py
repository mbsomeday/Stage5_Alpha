# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from configs.paths_dict import PATHS


class my_dataset(Dataset):
    def __init__(self, ds_name_list, txt_name, key_name='dataset_dict'):
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.key_name = key_name
        for ds_name in ds_name_list:
            self.ds_label_list.append(int(ds_name[1]) - 1)

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.ped_labels, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ped_labels, ds_labels = [], [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.key_name][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)
            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ped_labels.append(contents[-1])
                ds_labels.append(ds_label)

        return images, ped_labels, ds_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.ped_labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path)
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'file_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict


# if __name__ == '__main__':
#     ds = my_dataset(['D1', 'D2'], txt_name='val.txt')





















