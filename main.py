import argparse
from torch.utils.data import DataLoader

from models.VGG import vgg16_bn
from training.training import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_prefix', type=str)

    args = parser.parse_args()
    return args



model_name = 'vgg16'
model = vgg16_bn(num_class=2)
# ds_name_list = ['D3']
# batch_size = 32
# epochs = 3

args = get_args()
ds_name_list = args.ds_name_list
batch_size = args.batch_size
epochs = args.epochs

save_prefix = args.save_prefix

training = train_model(model_name, model, ds_name_list, batch_size, epochs, save_prefix)
training.train()










