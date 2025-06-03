import argparse
from torch.utils.data import DataLoader

from training.train_dsClip import train_clipDS_model
# from test_func.ds_cls import test_ds_classifier
from test_func.ped_cls import test_ped_classifier
from training.training_template import Ped_Classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_obj', type=str)
    parser.add_argument('--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--ds_weights_path', type=str)
    parser.add_argument('--isTrain', action='store_true')

    args = parser.parse_args()
    return args

args = get_args()
model_obj = args.model_obj
ds_name_list = args.ds_name_list
batch_size = args.batch_size
epochs = args.epochs
ds_weights_path = args.ds_weights_path
isTrain = args.isTrain


tt = Ped_Classifier(model_obj,
                    ds_name_list=ds_name_list,
                    batch_size=batch_size,
                    epochs=epochs,
                    ds_weights_path=ds_weights_path,
                    isTrain=isTrain,
                    resume=False
                    )
tt.train()




