import argparse
from torch.utils.data import DataLoader

from training.exp_09 import train_clipDS_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_prefix', type=str)

    args = parser.parse_args()
    return args


tt = train_clipDS_model(batch_size=48)
tt.train_model()






