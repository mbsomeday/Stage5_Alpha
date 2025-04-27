# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from data.dataset import my_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


class my_test():
    def __init__(self):
        self.ped_model = models.efficientnet_b0(weights=None, num_classes=2)
        ped_weights = r'/data/jcampos/jiawei_data/model_weights/Stage5/efficientB0_0.2CAMLoss_twoCls/efficientNetB0_D3_CAMLoss-25-0.95017.pth'
        self.ped_model = load_model(self.ped_model, ped_weights)
        self.ped_model.eval()
        self.ped_model.to(DEVICE)

        self.ds_model = models.efficientnet_b0(weights=None, num_classes=4)
        ds_weights = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'
        self.ds_model = load_model(self.ds_model, ds_weights)
        self.ds_model.eval()
        self.ds_model.to(DEVICE)

        self.feed_forward_features = None
        self.backward_features = None

        self.grad_layer = 'features'
        self._register_hooks(self.ds_model, self.grad_layer)

        ds_name_list = ['D1']
        batch_size = 48

        self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        self.test_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='test.txt')
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

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


    def val_and_test(self):

        val_correct_num = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                images = data['image']
                labels = data['ped_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                masked_images = np.zeros(shape=images.shape)
                for img_idx, image in enumerate(images):
                    image = torch.unsqueeze(image, dim=0)
                    heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
                    masked_images[img_idx] = masked_image.cpu().detach()
                masked_images = torch.tensor(masked_images)
                masked_images = masked_images.to(DEVICE)
                masked_images = masked_images.type(torch.float32)

                out = self.ped_model(masked_images)

                _, pred = torch.max(out, 1)

                val_correct_num += (pred == labels).sum()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())


        val_accuracy = val_correct_num / len(self.val_dataset)
        bc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f'val_accuracy: {val_accuracy}, bc: {bc}')
        print(cm)



if __name__ == '__main__':
    tt = my_test()
    tt.val_and_test()

















