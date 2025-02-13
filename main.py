import argparse
from torch.utils.data import DataLoader

from models.VGG import vgg16_bn
from training.training import train_model


# def get_args():
#     parser =
#
model_name = 'vgg16'
model = vgg16_bn(num_class=2)
ds_name_list = ['D3']
batch_size = 3
epochs = 3

training = train_model(model_name, model, ds_name_list, batch_size, epochs)
training.train()










