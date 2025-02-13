from torch.utils.data import DataLoader

from data.dataset import my_dataset
from models.VGG import vgg16_bn


ds_name_list = ['D3', 'D4']
txt_name = 'val.txt'

cur_dataset = my_dataset(ds_name_list, txt_name)
cur_loader = DataLoader(cur_dataset, batch_size=4)

model = vgg16_bn(num_class=2)

















