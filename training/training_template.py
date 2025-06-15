import os.path
import shutil

import matplotlib.pyplot as plt
import torch, copy, inspect
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
from torchcam.methods.gradient import LayerCAM, GradCAM
from tqdm import tqdm

from utils.utils import DEVICE, get_obj_from_str, load_model, DotDict, TemporaryGrad
from data.dataset import my_dataset
# from training.train_callbacks import EarlyStopping, Model_Logger      # remote
from train_callbacks import EarlyStopping, Model_Logger     # local



class NotYetUse_Loss(nn.Module):

    def __init__(self, ds_model_obj, ds_weights_path, grad_layer=None):
        super().__init__()
        self.ds_model_obj = ds_model_obj
        self.ds_weights_path = ds_weights_path

        # 实现 ds classifier
        self.ds_model = get_obj_from_str(ds_model_obj)(num_class=2)
        self.ds_model = load_model(self.ds_model, ds_weights_path).to(DEVICE)
        self.ds_model.eval()

        grad_layer = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6',
                      'features.7']

        self.cam_operator = LayerCAM(self.ds_model, target_layer=grad_layer)

    def heatmap_fusion(self, heatmaps):
        resized_hp = []
        for hp in heatmaps:
            hp = transforms.Resize(224)(hp).unsqueeze(0)
            resized_hp.append(hp)

        batch_heatmaps = torch.cat(resized_hp, dim=0)
        fused_heatmaps = torch.sum(batch_heatmaps, 0)

        (cam_min, cam_max) = (fused_heatmaps.min(), fused_heatmaps.max())
        fused_heatmaps = (fused_heatmaps - cam_min) / (((cam_max - cam_min) + 1e-08)).data

        print(f'fused_heatmaps:{fused_heatmaps.shape}')
        return fused_heatmaps

    def forward(self, images):
        ds_logits = self.ds_model(images)
        ds_preds = torch.argmax(ds_logits, 1)

        heatmaps = self.cam_operator(ds_preds[0].item(), scores=ds_logits)
        fused_heatmaps = self.heatmap_fusion(heatmaps)

        temp_hp = transforms.ToPILImage()(fused_heatmaps)
        thresh_hp = copy.deepcopy(temp_hp)

        print('thresh_hp', type(thresh_hp))

        # plt.figure(figsize=(8, 8))
        # plt.subplot(131)
        # plt.imshow(transforms.ToPILImage()(images[0]))
        # plt.subplot(132)
        # plt.imshow(temp_hp)
        # plt.subplot(133)
        # plt.imshow(thresh_hp)
        # plt.show()


class Blur_Image_Patch():
    '''
        根据 heatmap 对图片进行部分 blur
        现在版本：code实现heatmap
    '''

    def __init__(self, model_obj, ds_weights_path):
        self.ds_weights_path = ds_weights_path
        self.num_classes = 4
        self.ds_model = get_obj_from_str(model_obj)(num_class=self.num_classes)
        self.ds_model = load_model(self.ds_model, self.ds_weights_path).to(DEVICE).eval()

        self.grad_layer = 'features'
        self.attention_thresh = 0.5
        self.forward_feature = None
        self.backward_grads = None

        # self.cam_operator = GradCAM(self.ds_model, target_layer=[self.grad_layer])

        self._register_hooks()
        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

    def _register_hooks(self):
        def forward_hook(module, in_features, out_features):
            self.forward_feature = out_features

        def backward_hook(module, in_grad, out_grad):
            self.backward_grads = out_grad[0]

        gradient_layer_found = False
        for name, m in self.ds_model.named_modules():
            if name == self.grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Register forward hook and backward hook! Hooked layer: {self.grad_layer}")
                gradient_layer_found = True
                break
        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % self.grad_layer)


    def calc_cam(self, images):
        with torch.enable_grad():
            _, _, img_h, img_w = images.size()
            logits = self.ds_model(images)
            preds = torch.argmax(logits, dim=1)
            pred_idx = F.one_hot(preds, num_classes=self.num_classes)

            self.ds_model.zero_grad()
            logits.backward(gradient=pred_idx)
            self.ds_model.zero_grad()

            backward_features = self.backward_grads
            fl = self.forward_feature

            weights = F.adaptive_avg_pool2d(backward_features, 1)   # [batch_size, 1280, 1, 1]
            Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
            Ac = F.relu(Ac)
            Ac = F.interpolate(Ac, size=images.size()[2:])
            heatmap = Ac

            Ac_min = Ac.min()
            Ac_max = Ac.max()
            # scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)

            mask = heatmap.detach().clone()
            mask.requires_grad = False
            mask[mask < Ac_max] = 0
            masked_image = images - images * mask

            # plt_transform = transforms.ToPILImage()
            # img_idx = 0
            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(plt_transform(images[img_idx]))
            # plt.subplot(132)
            # plt.imshow(plt_transform(mask[img_idx]))
            # plt.subplot(133)
            # plt.imshow(plt_transform(masked_image[img_idx]))
            # plt.show()

        return heatmap, mask, masked_image

    def __call__(self, images):

        '''
            自己实现CAM
        '''
        # ds_logits = self.ds_model(images)
        # ds_preds = torch.argmax(ds_logits, 1)
        # print(f'ds_logits:{ds_logits}')
        # print(f'ds_preds:{ds_preds}')
        #
        # grad_fn = torch.zeros_like(ds_logits)
        # grad_fn[torch.arange(0, ds_logits.shape[0]), ds_preds] = 1
        #
        # self.ds_model.zero_grad()
        # ds_logits.backward(grad_fn)
        # self.ds_model.zero_grad()
        #
        # grad_weights = F.adaptive_avg_pool2d(self.backward_grad, 1)  # shape: (batch_size, 1280, 1, 1)
        # print(f'grad_weights:{grad_weights.shape}')
        #
        # ww_heatmaps = F.relu(torch.sum((self.forward_feature * grad_weights), dim=1))
        # print(f'ww_heatmaps:{ww_heatmaps.shape}')
        # ww_heatmaps = F.interpolate(ww_heatmaps.unsqueeze(0), (224, 224)).squeeze(0)

        '''
            用 torchcam 生成特征图
        '''
        # ds_logits = self.ds_model(images)
        # ds_preds = torch.argmax(ds_logits, 1).tolist()
        # heatmap_list = self.cam_operator(class_idx=ds_preds, scores=ds_logits)
        # heatmaps = heatmap_list[0]
        #
        # for hp in heatmaps:
        #     cur_max = hp.max()
        #     hp[hp < cur_max] = 0
        #
        # # heatmaps[heatmaps < self.attention_thresh] = 0
        # heatmaps_resized = F.interpolate(heatmaps.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        # heatmaps_resized = heatmaps_resized.squeeze().unsqueeze(1)
        # fade_images = images - images * heatmaps_resized

        '''
            展示 images
        '''
        # image_idx1 = 3
        # plt_transform = transforms.ToPILImage()
        # plt.figure(figsize=(16, 8))
        # plt.subplot(141)
        # plt.imshow(plt_transform(images[image_idx1]))
        # plt.title('org image')
        # plt.subplot(142)
        # plt.imshow(plt_transform(heatmaps[image_idx1]))
        # plt.title('torchcam')
        # plt.subplot(143)
        # plt.imshow(plt_transform(fade_images[image_idx1]))
        # plt.subplot(144)
        # plt.imshow(plt_transform(masked_image))
        # plt.show()

        return self.calc_cam(images)


class Ped_Classifier():
    def __init__(self, opts):
        self.opts = opts
        self.ped_model = get_obj_from_str(self.opts.ped_model_obj)(num_class=2).to(DEVICE)

        if self.opts.isTrain:
            self.training_setup()
        else:
            # 若是测试，则创建 test 文件夹用于存储结果
            self.callback_save_path = os.path.join(os.getcwd(), 'Test')
            if not os.path.exists(self.callback_save_path):
                os.mkdir(self.callback_save_path)

        self.print_args()


    def training_setup(self):
        '''
            初始化训练的各种参数
        '''
        # ********** 变量配置 **********
        self.opts.ds_model_obj = self.opts.ped_model_obj if self.opts.ds_model_obj is None else self.opts.ds_model_obj

        # ********** 创建 callback save dir **********
        self.callback_save_dir = self.opts.ped_model_obj.rsplit('.')[-1]
        for ds_name in self.opts.ds_name_list:
            info = '_' + ds_name
            self.callback_save_dir += info
        self.callback_save_dir += '_' + str(self.opts.rand_seed) + '_' + str(self.opts.beta) + self.opts.operator.lower() +'Loss'
        self.callback_save_path = os.path.join(os.getcwd(), self.callback_save_dir)
        print(f'Callback_savd_dir:{self.callback_save_path}')
        if not os.path.exists(self.callback_save_path):
            os.mkdir(self.callback_save_path)

        # ********** blur，fade，等操作 **********
        if self.opts.operator.lower() == 'fade':
            self.image_operator = Blur_Image_Patch(model_obj=self.opts.ds_model_obj, ds_weights_path=self.opts.ds_weights_path)
        elif self.opts.operator.lower() == 'blur':
            # self.image_operator = Blur_Image_Patch(model_obj=self.opts.ds_model_obj, ds_weights_path=self.opts.ds_weights_path)
            print('test blur')
        else:
            raise ValueError(f'The type of image operator evokes error, current:{self.opts.operator}')

        # ********** 数据准备 **********    augmentation_train
        self.train_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.opts.batch_size, shuffle=False)

        self.train_nonPed_num, self.train_ped_num = self.train_dataset.get_ped_cls_num()
        self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        # ********** loss & scheduler **********
        self.optimizer = torch.optim.RMSprop(self.ped_model.parameters(), lr=self.opts.base_lr, weight_decay=1e-5, eps=0.001)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # ********** 中断后重新训练 **********
        if self.opts.resume is True:
            self.reload()
        else:
            self.start_epoch = 0
            self.best_val_bc = -np.inf
            self.ped_model = self.init_model(self.ped_model)

        # ********** callbacks **********
        self.early_stopping = EarlyStopping(self.callback_save_path, top_k=self.opts.top_k, cur_epoch=self.start_epoch, patience=self.opts.patience,
                                            best_monitor_metric=self.best_val_bc)

        train_num_info = [len(self.train_dataset), self.train_nonPed_num, self.train_ped_num]
        val_num_info = [len(self.val_dataset), self.val_nonPed_num, self.val_ped_num]

        self.epoch_logger = Model_Logger(save_dir=self.callback_save_path,
                                         model_name=self.opts.ped_model_obj.split('.')[-1],
                                         ds_name_list=self.opts.ds_name_list,
                                         train_num_info=train_num_info,
                                         val_num_info=val_num_info,
                                         )

    def print_args(self):
        '''
            参数打印 并 保存到txt文件中
        '''
        print('-' * 40 + ' Args ' + '-' * 40)

        info = []
        for k, v in vars(self.opts).items():
            msg = f'{k}: {v}'
            print(msg)
            info.append(msg)

        # 将本次实验的参数写入txt中
        write_to_txt = os.path.join(self.callback_save_path, 'Args.txt')
        if os.path.exists(write_to_txt):
            os.remove(write_to_txt)
        with open(write_to_txt, 'a') as f:
            for item in info:
                f.write(item+'\n')

    def init_model(self, model):
        '''
            对模型权重进行初始化，保障多次训练结果变动不会变化太大
            适用 kaiming init，suitable for ReLU
            初始化种类参考： https://blog.csdn.net/shanglianlm/article/details/85165523
        '''
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 或 kaiming_uniform_
                nn.init.orthogonal_(m.weight)     # 正交初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return model


    def reload(self):
        '''
            中断后重新训练的情况，本函数加载模型，optimizer等参数
        '''
        ckpts = torch.load(self.opts.ped_weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                           weights_only=False)
        self.ped_model.load_state_dict(ckpts['model_state_dict'])

        self.start_epoch = ckpts['epoch'] + 1
        self.best_val_bc = ckpts['best_val_bc']
        self.base_lr = ckpts['lr']
        self.optimizer.load_state_dict(ckpts['optimizer_state_dict'])
        # self.scheduler.load_state_dict(ckpts['scheduler_state_dict'])

    def inif_pred_info(self):
        pred_info = {
            'y_pred': [],
            'nonPed_acc_num': 0,
            'ped_acc_num': 0,
            'correct_num': 0,
            'loss': 0.0
        }
        return pred_info

    def handle_pred_info(self, y_true: list, org_pred: dict, opered_pred=None, info_type='Train'):
        '''
            整合训练过程中的accuracy和loss等数据并进行 输出 和 返回
        '''
        epoch_info = {}

        # 若有 operated images
        if opered_pred is not None:
            correct_num = org_pred['correct_num'] + opered_pred['correct_num']
            accuracy = correct_num / (2 * len(y_true))

            bc_y_true = y_true + y_true
            bc_y_pred = org_pred['y_pred'] + opered_pred['y_pred']
            balanced_accuracy = balanced_accuracy_score(bc_y_true, bc_y_pred)

            loss = org_pred['loss'] + opered_pred['loss']

            epoch_info['org_bc'] = balanced_accuracy_score(y_true, org_pred['y_pred'])      # 在训练baseline的时候，不需要这个
            epoch_info['operated_bc'] = balanced_accuracy_score(y_true, opered_pred['y_pred'])      # 在训练baseline的时候，不需要这个

            show_info01 = f"\norg_bc:{epoch_info['org_bc']:.4f}, operated_bc:{epoch_info['operated_bc']:.4f}\n"

        # 只用 original image 训练的情况
        else:
            correct_num = org_pred['correct_num']
            accuracy = correct_num / len(y_true)
            balanced_accuracy = balanced_accuracy_score(y_true, org_pred['y_pred'])
            loss = org_pred['loss']
            show_info01 = ''

        epoch_info['accuracy'] = accuracy
        epoch_info['balanced_accuracy'] = balanced_accuracy
        epoch_info['loss'] = loss

        msg = f'Overall accuracy: {accuracy:.6f}, Overall balanced accuracy:{balanced_accuracy:.6f}, loss:{loss}' + show_info01

        print('-' * 30, str(info_type) + ' Info' + '-' * 30)
        print(msg)

        return DotDict(epoch_info)

    def train_one_epoch(self):
        self.ped_model.train()

        y_true = []
        org_dict = self.inif_pred_info()
        opered_dict = self.inif_pred_info() if self.opts.beta > 0.0 else None

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            images = data['image'].to(DEVICE)
            ped_labels = data['ped_label'].to(DEVICE)

            logits_org = self.ped_model(images)
            pred_org = torch.argmax(logits_org, 1)
            loss_org = self.loss_fn(logits_org, ped_labels)

            if self.opts.beta > 0.0:
                fade_images = self.image_operator(images)
                logits_opered = self.ped_model(fade_images)
                pred_opered = torch.argmax(logits_opered, 1)
                loss_opered = self.loss_fn(logits_opered, ped_labels)

                opered_dict['loss'] += loss_opered.item()
                loss_value = (1 - self.opts.beta) * loss_org + self.opts.beta * loss_opered
            else:
                loss_value = loss_org

            org_dict['loss'] += loss_org.item()

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # ------------ 对 pred 进行记录 ------------
            y_true.extend(ped_labels.cpu().numpy())
            nonPed_idx = (ped_labels == 0)
            ped_idx = (ped_labels == 1)

            org_dict['y_pred'].extend(pred_org.cpu().numpy())
            org_dict['correct_num'] += (pred_org == ped_labels).sum()
            org_dict['nonPed_acc_num'] += ((ped_labels[nonPed_idx] == pred_org[nonPed_idx]) * 1).sum()
            org_dict['ped_acc_num'] += ((ped_labels[ped_idx] == pred_org[ped_idx]) * 1).sum()

            if self.opts.beta > 0.0:
                opered_dict['y_pred'].extend(pred_opered.cpu().numpy())
                opered_dict['correct_num'] += (pred_opered == ped_labels).sum()
                opered_dict['nonPed_acc_num'] += ((ped_labels[nonPed_idx] == pred_opered[nonPed_idx]) * 1).sum()
                opered_dict['ped_acc_num'] += ((ped_labels[ped_idx] == pred_opered[ped_idx]) * 1).sum()

            # if batch_idx == 3:
            #     break

        train_epoch_info = self.handle_pred_info(y_true, org_pred=org_dict, opered_pred=opered_dict, info_type='Train')

        return train_epoch_info

    def val_on_epoch_end(self):
        self.ped_model.eval()

        y_true = []
        y_pred = []
        nonPed_acc_num = 0
        ped_acc_num = 0
        val_correct_num = 0

        # self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader)):
                images = data['image'].to(DEVICE)
                ped_labels = data['ped_label'].to(DEVICE)

                logits = self.ped_model(images)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(ped_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                val_correct_num += (preds == ped_labels).sum()

                nonPed_idx = (ped_labels == 0)
                nonPed_acc_num += (ped_labels[nonPed_idx] == preds[nonPed_idx]).sum()
                ped_idx = (ped_labels == 1)
                ped_acc_num += ((ped_labels[ped_idx] == preds[ped_idx]) * 1).sum()

                # break

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        val_epoch_info = {
            'accuracy': val_accuracy,
            'balanced_accuracy': val_bc,
        }

        return DotDict(val_epoch_info)

    def test(self):
        '''
            遍历每个数据集对模型进行测试，并将结果保存到Test文件夹中
        '''
        self.ped_model = load_model(self.ped_model, self.opts.ped_weights_path)
        self.ped_model.eval()

        write_to_txt = os.path.join(self.callback_save_path, 'Test.txt')
        with open(write_to_txt, 'a') as f:
            f.write('-' * 80 + '\n')
            f.write(f'Testing modle: {self.opts.ped_weights_path}.\n')
            f.write('ds_name, test_ba, tnr, tpr, tn, fp, fn, tp\n')

        for ds_name in self.opts.ds_name_list:
            test_dataset = my_dataset(ds_name_list=[ds_name], path_key=self.opts.data_key, txt_name='test.txt')
            test_loader = DataLoader(test_dataset, batch_size=self.opts.batch_size, shuffle=False)

            y_true = []
            y_pred = []
            nonPed_acc_num = 0
            ped_acc_num = 0
            test_correct_num = 0

            test_nonPed_num, test_ped_num = test_dataset.get_ped_cls_num()

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(test_loader)):
                    images = data['image'].to(DEVICE)
                    ped_labels = data['ped_label'].to(DEVICE)

                    logits = self.ped_model(images)
                    preds = torch.argmax(logits, dim=1)

                    y_true.extend(ped_labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

                    test_correct_num += (preds == ped_labels).sum()

                    nonPed_idx = (ped_labels == 0)
                    nonPed_acc_num += (ped_labels[nonPed_idx] == preds[nonPed_idx]).sum()
                    ped_idx = (ped_labels == 1)
                    ped_acc_num += ((ped_labels[ped_idx] == preds[ped_idx]) * 1).sum()

            test_accuracy = test_correct_num / len(test_dataset)
            test_bc = balanced_accuracy_score(y_true, y_pred)

            test_nonPed_acc = nonPed_acc_num / test_nonPed_num
            test_ped_acc = ped_acc_num / test_ped_num

            test_cm = confusion_matrix(y_true, y_pred)

            print('-' * 40 + 'Test Info' + '-' * 40)
            msg = f'DS_name:{ds_name}, Balanced accuracy:{test_bc:.4f}, accuracy: {test_accuracy:.4f}\nNon-ped accuracy:{test_nonPed_acc:.4f}({nonPed_acc_num}/{test_nonPed_num})\nPed accuracy:{test_ped_acc:.4f}({ped_acc_num}/{test_ped_num})'
            print(msg)
            print(f'CM on test set:\n{test_cm}')

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(tn, fp, fn, tp)

            with open(write_to_txt, 'a') as f:
                f.write(f'{ds_name}, {test_bc:.6f}, {test_nonPed_acc:.4f}, {test_ped_acc:.4f}, {tn}, {fp}, {fn}, {tp}\n')


    def update_learning_rate(self, epoch):
        old_lr = self.optimizer.param_groups[0]['lr']

        # warm-up阶段
        if epoch <= self.opts.warmup_epochs:  # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * epoch / self.opts.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * 0.963 ** (epoch / 3)  # gamma=0.963, lr decay epochs=3

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


    def train(self):
        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print('Total Batch:', len(self.train_loader))
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for EPOCH in range(self.start_epoch, self.opts.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(EPOCH + 1) + '=' * 30)
            train_epoch_info = self.train_one_epoch()
            val_epoch_info = self.val_on_epoch_end()

            # ------------------------ 调用callbacks ------------------------
            self.early_stopping(EPOCH + 1, self.ped_model, self.optimizer, val_epoch_info, scheduler=None)
            self.epoch_logger(epoch=EPOCH + 1, training_info=train_epoch_info, val_info=val_epoch_info)

            # ------------------------ 调用callbacks ------------------------
            # 每个epoch end调整learning rate
            self.update_learning_rate(EPOCH)

            if EPOCH > 20 and self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break

            # if EPOCH == 2:
            #     break


if __name__ == '__main__':

    model_obj = 'models.EfficientNet.efficientNetB0'

    ds_weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNetB0_Scratch\efficientNetB0_dsCls-10-0.97636.pth'
    ped_weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNetB0_Scratch\efficientNetB0_D2-21-0.94403.pth'
    # ds_weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNetB0_Scratch\efficientNetB0_D2-21-0.94403.pth'

    # tt = Ped_Classifier(model_obj,
    #                     ds_name_list=['D2'],
    #                     batch_size=4, epochs=100,
    #                     ds_weights_path=ds_weights_path,
    #                     ped_weights_path=ped_weights_path,
    #                     isTrain=False,
    #                     beta=0.0,
    #                     resume=False,
    #                     **testarg_dict
    #                     )
    # tt.train()
    # tt.test()

    test_obj = Blur_Image_Patch(model_obj=model_obj, ds_weights_path=ds_weights_path)

    torch.manual_seed(4)
    ds_name_list = ['D3']
    batch_size = 4
    val_dataset = my_dataset(ds_name_list=ds_name_list, path_key='tiny_dataset', txt_name='val.txt')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, data_dict in enumerate(val_loader):
        images = data_dict['image']
        ped_labels = data_dict['ped_label']

        print(ped_labels)

        test_obj.calc_cam(images)


        break
















