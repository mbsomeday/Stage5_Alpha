import argparse
from torch.utils.data import DataLoader

from training.train_dsClip import train_clipDS_model
# from test_func.ds_cls import test_ds_classifier
from test_func.ped_cls import test_ped_classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--ds_name_list', nargs='+', help='dataset list')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--txt_name', type=str)
    # parser.add_argument('--epochs', type=int)
    # parser.add_argument('--save_prefix', type=str)

    args = parser.parse_args()
    return args

args = get_args()
model_obj = 'models.EfficientNet.efficientNetB0'
weights_path = args.weights_path
batch_size = args.batch_size
ds_name_list = [args.ds_name_list]
txt_name = args.txt_name

print(f'ds_name_list:{ds_name_list}')

tt = test_ped_classifier(model_obj, weights_path, ds_name_list, batch_size, txt_name)
tt.train_model()

# args = get_args()
# weights_path = args.weights_path
# batch_size = args.batch_size
# txt_name = args.txt_name


# weights_path = r'/kaggle/working/Stage5_Alpha/vgg16_CropDSCls/vgg16_CropDSCls-44-0.76518.pth'
# weights_path = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'

# model_obj = 'models.EfficientNet.efficientNetB0'
# tt = test_ds_classifier(model_obj=model_obj, weights_path=weights_path, batch_size=batch_size, txt_name=txt_name)


