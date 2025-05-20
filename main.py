import argparse
from torch.utils.data import DataLoader

from training.exp_09 import train_clipDS_model
from test_func.crop_test import test_ds_classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_prefix', type=str)

    args = parser.parse_args()
    return args


# tt = train_clipDS_model(batch_size=48)
# tt.train_model()



weights_path = r'/kaggle/working/Stage5_Alpha/vgg16_CropDSCls/vgg16_CropDSCls-44-0.76518.pth'
# weights_path = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'

tt = test_ds_classifier(weights_path=weights_path, batch_size=48)



