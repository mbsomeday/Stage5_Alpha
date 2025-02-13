import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from configs.paths_dict import PATHS


class my_dataset(Dataset):
    def __init__(self, ds_name_list, txt_name):
        self.ds_name_list = ds_name_list
        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImagesLabels()
        # print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.labels)} images')


    def init_ImagesLabels(self):
        images, labels = [], []

        for ds_name in self.ds_name_list:
            ds_dir = PATHS['dataset_dict'][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                labels.append(contents[-1])

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        image = self.img_transforms(image)
        label = np.array(label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'file_path': image_path,
            'label': label
        }

        return image_dict


























