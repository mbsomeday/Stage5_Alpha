import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class my_dataset(Dataset):
    def __init__(self, ds_path, txt_path):
        self.ds_path = ds_path
        self.txt_path = txt_path
        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def init_Images(self):
        images = []

        with open(self.txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            item = item.strip().split()
            print(item)
            break

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    train_loader = my_dataset(ds_path=r'')





















