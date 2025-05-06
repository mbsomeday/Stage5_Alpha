# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch, argparse
import torchvision.models as visionModels
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from data.dataset import my_dataset
from utils.utils import TemporaryGrad
from configs.paths_dict import PATHS
from utils.utils import plot_cm, get_gpu_info, get_obj_from_str, load_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_obj', type=str)
    parser.add_argument('-t', '--train_on', type=str)
    parser.add_argument('-d', '--ds_name', type=str, help='dataset that the model is tested on')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--ds_key_name', type=str)
    parser.add_argument('--txt_name', type=str)
    parser.add_argument('-w', '--weights_path', type=str)

    args = parser.parse_args()
    return args


def ped_test(model, ds_name, test_dataset, test_loader):
    print(f'Working machine {DEVICE}')
    model = model.to(DEVICE)
    model.eval()

    correct_num = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict['image'].to(DEVICE)
            ped_labels = data_dict['ped_label'].to(DEVICE)

            ped_out = model(images)
            ped_pred = torch.argmax(ped_out, dim=1)

            correct_num += (ped_pred == ped_labels).sum()

            y_true.extend(ped_labels.cpu().numpy())
            y_pred.extend(ped_pred.cpu().numpy())
            # break

        test_accuracy = correct_num / len(test_dataset)
        bc = balanced_accuracy_score(y_true, y_pred)

        print(f'test_accuracy: {test_accuracy} - balanced accuracy: {bc}  \n{correct_num}/{len(test_dataset)}')

        # 绘制混淆矩阵
        label_names = ['ped', 'nonPed']
        title = f'Ped Cls CM on AE4 Recons {ds_name}'
        plot_cm(y_true, y_pred, label_names, title=title)


def ds_test(model, test_dataset, test_loader):
    model = model.to(DEVICE)
    model.eval()

    correct_num = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict['image'].to(DEVICE)
            ds_labels = data_dict['ds_label'].to(DEVICE)

            ds_out = model(images)
            ds_pred = torch.argmax(ds_out, dim=1)
            correct_num += (ds_pred == ds_labels).sum()

            y_true.extend(ds_labels.cpu().numpy())
            y_pred.extend(ds_pred.cpu().numpy())

        test_accuracy = correct_num / len(test_dataset)
        bc = balanced_accuracy_score(y_true, y_pred)

        print(f'test_accuracy: {test_accuracy} - balanced accuracy: {bc}  \n{correct_num}/{len(test_dataset)}')

        # 绘制混淆矩阵
        label_names = ['D1', 'D2', 'D3', 'D4']
        title = f'Dataset Cls CM on AE4 Recons Datasets'
        plot_cm(y_true, y_pred, label_names, title=title)


class test_ped_model_camLoss():
    def __init__(self, model_obj: str,
                 ped_weights,
                 ds_name_list,
                 txt_name='test.txt',
                 batch_size=48,
                 ):

        # -------------------- 打印训练信息 --------------------
        print('-' * 20, 'Testing Info', '-' * 20)
        print(f'batch_size: {batch_size}')
        print(f'txt_name: {txt_name}')
        print('-' * 20)

        # -------------------- 成员变量 --------------------
        self.ds_model_obj = model_obj
        self.ped_weights = ped_weights
        self.txt_name = txt_name

        # -------------------- 获取 ped model for train --------------------
        self.model = get_obj_from_str(model_obj)(num_class=2)
        self.model = load_model(self.model, self.ped_weights)
        self.model = self.model.to(DEVICE)

        # -------------------- 获取数据 --------------------
        self.ds_name_list = ds_name_list
        self.test_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name=txt_name)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        # -------------------- 获取ds model，目的是融入 cam loss --------------------
        self.ds_model = get_obj_from_str(self.ds_model_obj)(num_class=4)
        # ds_weights = r'/kaggle/input/stage5-weights-effidscls/efficientNetB0_dsCls-10-0.97636.pth'
        # ds_weights = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'
        ds_weights = r'/data/jcampos/jiawei_data/code/ResNet34_dsCls/ResNet34_dsCls-23-0.96353.pth'
        # ds_weights = r'/veracruz/home/j/jwang/data/model_weights/efficientNetB0_dsCls-10-0.97636.pth'
        self.ds_model = load_model(self.ds_model, ds_weights)
        self.ds_model.eval()
        self.ds_model = self.ds_model.to(DEVICE)

        self.feed_forward_features = None
        self.backward_features = None

        # self.grad_layer = 'features'  # efficient
        self.grad_layer = 'layer4'  # resnet
        self._register_hooks(self.ds_model, self.grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

    def _register_hooks(self, model, grad_layer):
        '''
            为 ds_model 注册钩子函数
        '''

        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Register forward hook and backward hook! Hooked layer: {self.grad_layer}")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def calc_cam(self, model, image):
        '''
            输入的image为4D
        '''
        with TemporaryGrad():
            logits = model(image)
            pred = torch.argmax(logits, dim=1)
            model.zero_grad()
            grad_yc = logits[0, pred]
            grad_yc.backward()
            # print(f'反向传播之后：{self.backward_features.shape}')
            model.zero_grad()

            w = F.adaptive_avg_pool2d(self.backward_features, 1)  # shape: (batch_size, 1280, 1, 1)
            # print(f'w: {w.shape}')
            temp_w = w[0].unsqueeze(0)
            temp_fl = self.feed_forward_features[0].unsqueeze(0)
            ac = F.conv2d(temp_fl, temp_w)
            ac = F.relu(ac)

            Ac = F.interpolate(ac, (224, 224))

            heatmap = Ac

            # 获取mask
            Ac_min = Ac.min()
            Ac_max = Ac.max()
            # print(f'Attention map diff: {Ac_max - Ac_min}')
            # scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
            # mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
            # masked_image = images - images * mask

            mask = heatmap.detach().clone()
            mask.requires_grad = False
            mask[mask < Ac_max] = 0
            masked_image = image - image * mask

        return heatmap, mask, masked_image

    def test_model(self):
        self.model.eval()

        val_correct_num = 0
        y_true = []
        y_pred = []
        nonPed_acc_num = 0
        ped_acc_num = 0

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                images = data['image']
                labels = data['ped_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # ------------ 计算 cam loss ------------
                # 试验 case1: 对所有图片都进行cam_loss
                masked_images = np.zeros(shape=images.shape)
                for img_idx, image in enumerate(images):
                    image = torch.unsqueeze(image, dim=0)
                    heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
                    masked_images[img_idx] = masked_image.cpu().detach()
                masked_images = torch.tensor(masked_images)
                masked_images = masked_images.to(DEVICE)
                masked_images = masked_images.type(torch.float32)
                masked_out = self.model(masked_images)
                _, masked_pred = torch.max(masked_out, 1)
                pred = masked_pred


                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                # ------------ 计算各个类别的正确个数 ------------
                nonPed_idx = labels == 0
                nonPed_acc = (labels[nonPed_idx] == pred[nonPed_idx]) * 1
                nonPed_acc_num += nonPed_acc.sum()

                ped_idx = labels == 1
                ped_acc = (labels[ped_idx] == pred[ped_idx]) * 1
                ped_acc_num += ped_acc.sum()

        val_accuracy = val_correct_num / len(self.test_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        print(f'Test cm:\n {cm}')

        print(f'Balanced accuracy: {val_bc:.6f}, accuracy: {val_accuracy:.6f}')






if __name__ == '__main__':
    # 打印当前使用的gpu信息
    get_gpu_info()

    args = get_args()
    ds_name = args.ds_name
    batch_size = args.batch_size
    ds_key_name = args.ds_key_name
    txt_name = args.txt_name
    model_obj = args.model_obj

    if args.weights_path is not None:
        weights_path = args.weights_path
    else:
        train_on = args.train_on
        weights_path = PATHS['EfficientNet_ped_cls'][train_on]

    # pedestrian classification
    # model = vgg16_bn(num_class=2).to(DEVICE)

    from torchvision import models
    # model = models.efficientnet_b0(num_classes=2)

    # model = get_obj_from_str(model_obj)(num_class=2)
    #
    # # model = visionModels.efficientnet_b0(weights=None, progress=True, num_classes=2)
    # print(f"Reload model {weights_path}")
    # ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    # model.load_state_dict(ckpt['model_state_dict'])
    #
    # test_dataset = my_dataset(ds_name_list=[ds_name], txt_name=txt_name, path_key=ds_key_name)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # ped_test(model, ds_name=ds_name, test_dataset=test_dataset, test_loader=test_loader)

    ds_name_list = [ds_name]
    my_test = test_ped_model_camLoss(model_obj=model_obj, ped_weights=weights_path, ds_name_list=ds_name_list, txt_name=txt_name, batch_size=batch_size)
    my_test.test_model()

    # # AE Reconstruction dataset classification
    # model = vgg16_bn(num_class=4).to(DEVICE)
    # weights_path = PATHS['ds_cls_ckpt']
    # print(f"Reload model {weights_path}")
    # ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    # model.load_state_dict(ckpt['model_state_dict'])
    #
    # ds_name_list = ['D1', 'D2', 'D3', 'D4']
    # test_dataset = my_dataset(ds_name_list=ds_name_list, path_key='AE4_dataset',txt_name='test.txt')
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # ds_test(model, test_dataset, test_loader)





















