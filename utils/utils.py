import torch, importlib, os
from sklearn.metrics import confusion_matrix
import pandas as pd
from torchvision import utils as vutils

from models.VGG import vgg16_bn
from configs.paths_dict import PATHS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_cm(y_true, y_pred, label_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(f'cm:\n {cm}')
    # conf_matrix_df = pd.DataFrame(cm, columns=label_names, index=label_names)
    # sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    # plt.title(title)
    # plt.ylabel('Label')
    # plt.xlabel('Prediction')
    # # plt.savefig(f'{title}.png')
    # print(f'Image save to {title}.png')
    # # plt.show()



def get_vgg_DSmodel():
    '''
        获取 VGG dataset classifier
    '''
    model = vgg16_bn(num_class=4)
    weight_path = PATHS['ds_cls_ckpt']
    print(f'Loading dataset classifier: {weight_path}')
    checkpoints = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoints['model_state_dict'])
    return model


def get_orgPed_model(ds_name):
    model = vgg16_bn(num_class=2)
    weight_path = PATHS['ped_cls_ckpt'][ds_name]
    print(f'Loading model: {weight_path}')
    checkpoints = torch.load(weight_path, map_location=DEVICE, weights_only=True if DEVICE=='cuda' else False)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model


def load_model(model, weights_path):
    print(f'Loading model from {weights_path}')
    ckpts = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model.load_state_dict(ckpts['model_state_dict'])
    return model

class TemporaryGrad(object):
    '''
    https://blog.csdn.net/qq_44980390/article/details/123672147
    '''
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


def get_obj_from_str(in_str):
    '''
        根据 str类型的函数名 来调用函数
    '''
    module, cls = in_str.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class DotDict(dict):
    '''
        将字典转换为可直接用 . 调用的对象
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value



def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)



def get_gpu_info():
    '''
        打印出显卡信息
    '''
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print(f'GPU num: {n_gpu}, g_name: {gpu_name}')
















