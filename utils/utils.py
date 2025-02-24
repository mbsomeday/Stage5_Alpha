import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models.VGG import vgg16_bn
from configs.paths_dict import PATHS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_cm(y_true, y_pred, label_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(f'cm:\n {cm}')
    conf_matrix_df = pd.DataFrame(cm, columns=label_names, index=label_names)
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Label')
    plt.xlabel('Prediction')
    # plt.savefig(f'{title}.png')
    print(f'Image save to {title}.png')
    # plt.show()



def get_ds_model():
    '''
        获取 dataset classifier
    '''
    model = vgg16_bn(num_class=4)
    weight_path = PATHS['ds_cls_ckpt']
    checkpoints = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model


def get_orgPed_model(ds_name):
    model = vgg16_bn(num_class=2)
    weight_path = PATHS['org_dataset'][ds_name]
    checkpoints = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model




















