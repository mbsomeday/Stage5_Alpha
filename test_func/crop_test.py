import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision import models

from data.dataset import dataset_clip
from utils.utils import load_model, get_obj_from_str, DEVICE


def test_ds_classifier(weights_path, batch_size):


    ds_model = models.vgg16(num_classes=4, weights=None)
    ds_model = load_model(ds_model, weights_path).to(DEVICE)
    ds_model.eval()

    ds_name_list = ['D1', 'D2', 'D3', 'D4']
    path_key = 'tiny_dataset'

    ds_dataset = dataset_clip(ds_name_list, path_key, txt_name='test.txt')
    ds_loader = DataLoader(ds_dataset, batch_size=batch_size, shuffle=False)

    correct_num = 0
    y_pred = []
    y_label = []

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(ds_loader)):
            images = data_dict['clip'].to(DEVICE)
            ds_label = data_dict['ds_label'].to(DEVICE)

            logits = ds_model(images)
            probs = torch.argmax(logits, 1)

            y_label.extend(ds_label.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

            correct_num += (ds_label == probs).sum()

        cm = confusion_matrix(y_label, y_pred)
        print(f'Testing cm:\n {cm}')

        ds_accuracy = correct_num / len(ds_dataset)
        print(f'准确率为：{ds_accuracy}')



# if __name__ == '__main__':
#     weights_path = r'/kaggle/working/Stage5_Alpha/vgg16_CropDSCls/vgg16_CropDSCls-44-0.76518.pth'
#     # weights_path = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'
#
#     tt = test_ds_classifier(weights_path=weights_path, batch_size=48)
#















