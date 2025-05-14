from torchvision import transforms
from torchvision import models
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from utils.utils import load_model
from data.dataset import my_dataset
from training.training import TemporaryGrad
from utils.utils import get_obj_from_str, load_model
from utils.utils import get_vgg_DSmodel, DotDict
from training.train_callbacks import EarlyStopping, Epoch_logger


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Build Gaussian Kernel, from pytorch
def _get_gaussian_kernel1d(kernel_size: int, sigma: float):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _get_gaussian_kernel2d(kernel_size, sigma, dtype):
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(DEVICE, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(DEVICE, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

def get_gaussianFilter(sigma_list, k_size=5, blurred_expan_factor=100):
    '''
        用给定的n个数字来制作gaussian filter
    '''
    dtype = torch.float32
    kernel_size = (k_size, k_size)
    kernel_list = []

    for i in range(len(sigma_list)):
        blurr_sigma = sigma_list[i] * blurred_expan_factor
        # print(f'blurr_sigma: {blurr_sigma}')
        sigma = (blurr_sigma, blurr_sigma)
        kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype)
        kernel = kernel.expand(3, 1, kernel.shape[0], kernel.shape[1])
        kernel_list.append(kernel)

    all_kernels = torch.stack(kernel_list)   # [n, 3, 1, 5, 5]
    all_kernels = all_kernels.view(-1, 1, k_size, k_size)  # [3*n, 1, 5, 5]

    return all_kernels

def split_batch_into_patches(imgs, patch_size=32, gaussian_kernel=5):
    '''
        将batch中每张image分割为[49, 3, 32, 32]
    :param imgs: [batch_size, 3, 224, 224]
    :param patch_size:
    :param gaussian_kernel: 本实验会固定该数值为5
    :return:
    '''
    # imgs: (bs, 3, 224, 224)
    B, C, H, W = imgs.shape
    # unfold: 按高度和宽度方向分割
    patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # shape: (B, 3, 7, 7, 32, 32)

    # 变换维度，得到 (B, 7, 7, 3, 32, 32)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    # 合并为 (B, 49, 3, 32, 32)
    patches = patches.reshape(B, -1, C, patch_size, patch_size)
    # 合并 batch 和 patch 维度以便 interpolate
    patches = patches.view(B * 49, C, patch_size, patch_size)
    # resize to (padding_size, padding_size)

    kernel_size = [gaussian_kernel, gaussian_kernel]
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]  # 这里遵循pytorch中gaussian blur的操作，先对img部分进行blur
    resized_patches = F.pad(patches, padding, mode="reflect")

    each_patch_size = resized_patches.shape[-1]
    # 还原回 (B, 49, C, padding_size, padding_size)
    resized_patches = resized_patches.view(B, 49, C, each_patch_size, each_patch_size)

    return resized_patches  # shape: (B, 49, 3, each_patch_size, each_patch_size)


# todo:这个函数直接返回patch image和 filtered score
def select_patches_by_score(patches, scores, threshold=0.5):
    """
    patches: (B, 49, 3, 32, 32)
    scores: (B, 7, 7)
    Returns: List[List[Tensor]]: outer list over batch, inner list over selected patches
    """
    B, _, C, H, W = patches.shape
    selected_patches = []
    selected_score = []
    mask_dic = []   # 该字典适用于恢复图片形状的

    # 将 scores 展平为 (B, 49) 来对应每个 patch
    scores_flat = scores.view(B, -1)  # (B, 49)

    for b in range(B):
        mask = scores_flat[b] > threshold
        mask_dic.append(torch.unsqueeze(torch.reshape(mask, (7, 7)), 0))
        temp_score = scores_flat[b][mask]
        selected = patches[b][mask]
        # print(f'selefcted:{selected.shape}')

        # # todo test
        # for p in selected:
        #     img = transforms.ToPILImage()(p)
        #     # 检测edge
        #
        #     img.show()
        #     break

        if selected.shape[0] > 0:
            selected_patches.append(selected)
            selected_score.extend(temp_score)

    selected_patches = torch.cat(selected_patches, 0)
    # print(f'selected_patches{selected_patches.shape}, {len(selected_score)}')

    mask_dic = torch.cat(mask_dic, 0)

    return selected_patches, selected_score, mask_dic


def restore_images_with_blur(images, blurred_patches, mask_dict, batch_heatmaps):
    '''
        将blur的部分替换原图中的部分
    :param images:
    :param blurred_patches:
    :param mask_dict:
    :param batch_heatmaps:
    :return: 形状为[batch_size, 3, 224, 224]的带有blur的images
    '''
    blurred_patches = torch.reshape(blurred_patches, (-1, 3, 32, 32))
    images_with_blur = images.clone().detach()
    batch_size = images.shape[0]

    # print(f'images: {images_with_blur.shape}, blurred_patches:{blurred_patches.shape}, mask_dict:{mask_dict.shape}')
    # print(f'batch_heatmaps:{batch_heatmaps.shape}')
    coords = torch.nonzero(mask_dict, as_tuple=False)   # 获取所有值为True的idx
    # print(f'coordds.shape:{coords.shape}')

    idx_l = [0*i for i in range(batch_size)]

    # 将blurred部分替换原有image的部分
    for idx, (b, i, j) in enumerate(coords):
        idx_l[b] += 1

        x_start, x_end = i * 32, (i + 1) * 32
        y_start, y_end = j * 32, (j + 1) * 32

        # test_black = torch.rand((3, 32, 32))
        # images_with_blur[b, :, x_start:x_end, y_start:y_end] = test_black
        images_with_blur[b, :, x_start:x_end, y_start:y_end] = blurred_patches[idx]

    # print(f'每个图片的patch:{idx_l}')


    return images_with_blur

'''
    这个类的作用是，输入ds_model, batch_org_images，输出batch_blurred_image
'''
class Bathch_Image_Blur():
    def __init__(self, model_obj,
                 ds_weights_path,
                 grad_layer='features',

                 ):

        self.ds_model = get_obj_from_str(model_obj)(num_class=4)
        self.ds_model = load_model(self.ds_model, ds_weights_path)
        self.ds_model = self.ds_model.to(DEVICE)
        self.ds_model.eval()    # 不管是train还是test，都将ds_model设置为test模式

        self.grad_layer = grad_layer
        self.threhold = 0.5 # 用于过滤attention score，注意力分数太小的不进行blur

        self.ds_feed_forward_features = None
        self.ds_backward_features = None
        self._register_hooks(self.ds_model, grad_layer=grad_layer)

    def _register_hooks(self, model, grad_layer):
        '''
            为 ds_model 注册钩子函数
        '''

        def ds_forward_hook(module, grad_input, grad_output):
            self.ds_feed_forward_features = grad_output

        def ds_backward_hook(module, grad_input, grad_output):
            self.ds_backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(ds_forward_hook)
                m.register_full_backward_hook(ds_backward_hook)
                print(
                    f"Register forward hook and backward hook! Hooked layer: {self.grad_layer}")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def batch_blur(self, images):
        '''
            输入batch images，输出blurred batch images
        '''

        # 将 batch image 转化为 小的 patch，方便进行 gaussian 操作
        patched_images = split_batch_into_patches(images)  # shape: (batch_size, 49, 3, 36, 36)，其中36是after padding
        batch_size = patched_images.shape[0]

        with TemporaryGrad():
            ds_org_logits = self.ds_model(images)
            ds_org_pred = torch.argmax(ds_org_logits, 1)
            # print(f'ds_org_pred:{ds_org_pred}')

            self.ds_model.zero_grad()
            batch_indices = torch.arange(ds_org_logits.size(0))  # [0, 1, 2, 3]
            grad_yc = ds_org_logits[batch_indices, ds_org_pred]  # shape: [4]
            grad_yc.backward(torch.ones_like(grad_yc), retain_graph=True)   # 在训练是要保留特征图，否则第二次backward会报错
            self.ds_model.zero_grad()

            # -------------- 生成attention map，参考tell me where to guid my attention进行conv2d操作 --------------
            # avg of the last conv layer
            w = F.adaptive_avg_pool2d(self.ds_backward_features, 1)  # shape: (batch_size, 1280, 1, 1)
            feature_maps = self.ds_feed_forward_features
            feature_maps = feature_maps.view(-1, batch_size * 1280, 7, 7)  # shape: (1, batch_size * 1280, 1, 1)
            batch_heatmaps = F.relu(F.conv2d(feature_maps, w, groups=batch_size))

            # print(f'batch_heatmaps： {batch_heatmaps.shape}')

            hm_min = batch_heatmaps.min()
            hm_max = batch_heatmaps.max()

            batch_heatmaps = (batch_heatmaps - hm_min) / (hm_max - hm_min)

            # 根据 attention score选择patch
            selected_patches, selected_score, mask_dict = select_patches_by_score(patched_images, batch_heatmaps, threshold=self.threhold)

            selected_patches = torch.reshape(selected_patches, (1, -1, 36, 36))
            gaussian_kernels = get_gaussianFilter(selected_score, k_size=5, blurred_expan_factor=100)
            selected_patches = selected_patches.to(DEVICE)
            selected_patches = selected_patches.to(DEVICE)
            blurred_patches = F.conv2d(selected_patches, gaussian_kernels, groups=selected_patches.shape[-3])

            images_with_blur = restore_images_with_blur(images, blurred_patches, mask_dict, batch_heatmaps)

            # print(f'images_with_blur:{images_with_blur.shape}')

            # todo:检查单个heatmap跟batch heatmap是否一致

        return images_with_blur


class PedCls_with_camLoss():
    def __init__(self, model_obj,
                 ds_name_list,
                 ds_weights,
                 batch_size,
                 reload=None,
                 epochs=50,
                 base_lr=0.01,
                 warmup_epochs=0,
                 lr_patience=5,
                 camLoss_coefficient=None,
                 ds_model_obj=None,
                 save_best_cls=False,
                 mode='test'):
        '''

        :param model_obj:
        :param mode: train / test
        '''

        # -------------------- train 和 test 都需要用到的变量 --------------------
        print('-' * 20, 'Basic Info', '-' * 20)

        self.ds_name_list = ds_name_list

        self.ped_model = get_obj_from_str(model_obj)(num_class=2)
        self.ped_model = self.ped_model.to(DEVICE)

        self.cam_operater = Bathch_Image_Blur(model_obj=model_obj, ds_weights_path=ds_weights, grad_layer='features')


        if mode == 'train':
            print('-' * 20, 'Training Info', '-' * 20)

            self.ped_model.train()
            self.ped_model = self.ped_model.to(DEVICE)

            # -------------------- 成员变量 --------------------
            self.warmup_epochs = warmup_epochs
            self.epochs = epochs
            self.base_lr = base_lr
            self.lr_patience = lr_patience
            self.save_best_cls = save_best_cls
            self.camLoss_coefficient = camLoss_coefficient
            self.ds_model_obj = ds_model_obj
            self.start_epoch = 0

            # -------------------- 获取数据 --------------------
            self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

            self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

            self.train_nonPed_num, self.train_ped_num = self.train_dataset.get_ped_cls_num()
            self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

            # -------------------- 训练配置 --------------------
            self.optimizer = torch.optim.RMSprop(self.ped_model.parameters(), lr=self.base_lr, weight_decay=1e-5, eps=0.001)
            self.loss_fn = torch.nn.CrossEntropyLoss()

            # -------------------- Callbacks --------------------
            # 设置保存训练信息的文件夹的名字
            callback_savd_dir = model_obj.rsplit('.')[-1]
            for ds_name in ds_name_list:
                info = '_' + ds_name
                callback_savd_dir += info

            callback_savd_dir += '_1CAMLoss'
            print(f'callback_savd_dir:{callback_savd_dir}')
            self.early_stopping = EarlyStopping(callback_savd_dir, top_k=2)

            train_num_info = [len(self.train_dataset), self.train_nonPed_num, self.train_ped_num]
            val_num_info = [len(self.val_dataset), self.val_nonPed_num, self.val_ped_num]

            self.epoch_logger = Epoch_logger(save_dir=callback_savd_dir, model_name=model_obj.split('.')[-1],
                                             ds_name_list=ds_name_list, train_num_info=train_num_info,
                                             val_num_info=val_num_info,
                                             task='ped_cls'
                                             )
            # -------------------- 如果reload，optmizer，start_epoch等也要重新设置 --------------------

            # todo: 写完 train 后再 test
        elif mode == 'test':
            print('-' * 20, 'Test Info', '-' * 20)

            # -------------------- 获取数据 --------------------
            self.test_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='test.txt')
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

            self.test_nonPed_num, self.test_ped_num = self.test_dataset.get_ped_cls_num()

        else:
            raise ValueError(f'mode should be train or test, current mode is {mode}, not recognized!')

    def lr_decay(self, epoch):
        # warm-up阶段
        if epoch <= self.warmup_epochs:        # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.base_lr * epoch / self.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.base_lr * 0.963 ** (epoch / 3)        # gamma=0.963, lr decay epochs=3


    def train_one_epoch(self):
        self.ped_model.train()

        y_true = []
        y_pred = []
        nonPed_acc_num = 0
        ped_acc_num = 0
        training_correct_num = 0
        training_loss = 0.0

        for batch, data in enumerate(tqdm(self.train_loader)):
            images = data['image']
            labels = data['ped_label']
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            batch_blurred = self.cam_operater.batch_blur(images)

            logits_blur = self.ped_model(batch_blurred)
            pred_blur = torch.argmax(logits_blur, 1)

            loss_blur = self.loss_fn(logits_blur, labels)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred_blur.cpu().numpy())
            training_loss += loss_blur.item()

            self.optimizer.zero_grad()
            loss_blur.backward()
            self.optimizer.step()

            training_correct_num += (pred_blur == labels).sum()
            # ------------ 计算各个类别的正确个数 ------------
            nonPed_idx = labels == 0
            nonPed_acc = (labels[nonPed_idx] == pred_blur[nonPed_idx]) * 1
            nonPed_acc_num += nonPed_acc.sum()

            ped_idx = labels == 1
            ped_acc = (labels[ped_idx] == pred_blur[ped_idx]) * 1
            ped_acc_num += ped_acc.sum()

            # break

        train_accuracy = training_correct_num / len(self.train_dataset)
        training_bc = balanced_accuracy_score(y_true, y_pred)

        train_nonPed_acc = nonPed_acc_num / self.train_nonPed_num
        train_ped_acc = ped_acc_num / self.train_ped_num

        cm = confusion_matrix(y_true, y_pred)
        print(f'training cm:\n {cm}')

        print(f'Training Loss:{training_loss:.6f}, Balanced accuracy: {training_bc:.6f}, accuracy: {train_accuracy:.6f}, [0: {train_nonPed_acc:.6f}({nonPed_acc_num}/{self.train_nonPed_num}), 1: {train_ped_acc:.6f}({ped_acc_num}/{self.train_ped_num}), ({training_correct_num}/{len(self.train_dataset)})]')
        train_epoch_info = {
            'train_accuracy': train_accuracy,
            'training_loss': training_loss,
            'training_correct_num': training_correct_num,
            'training_bc': training_bc,
            'nonPed_acc_num': nonPed_acc_num,
            'train_nonPed_acc': train_nonPed_acc,
            'ped_acc_num': ped_acc_num,
            'train_ped_acc': train_ped_acc
        }
        train_epoch_info = DotDict(train_epoch_info)
        return train_epoch_info


    def val_on_epoch_end(self, epoch):
        self.ped_model.eval()

        val_loss = 0.0
        val_correct_num = 0
        y_true = []
        y_pred = []
        nonPed_acc_num = 0
        ped_acc_num = 0

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['ped_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                batch_blurred = self.cam_operater.batch_blur(images)
                logits_blur = self.ped_model(batch_blurred)
                pred_blur = torch.argmax(logits_blur, 1)

                loss_blur = self.loss_fn(logits_blur, labels)
                val_loss += self.loss_fn(logits_blur, labels)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred_blur.cpu().numpy())

                val_correct_num += (pred_blur == labels).sum()
                nonPed_idx = labels == 0
                nonPed_acc = (labels[nonPed_idx] == pred_blur[nonPed_idx]) * 1
                nonPed_acc_num += nonPed_acc.sum()
                ped_idx = labels == 1
                ped_acc = (labels[ped_idx] == pred_blur[ped_idx]) * 1
                ped_acc_num += ped_acc.sum()

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        val_nonPed_acc = nonPed_acc_num / self.val_nonPed_num
        val_ped_acc = ped_acc_num / self.val_ped_num

        cm = confusion_matrix(y_true, y_pred)
        print(f'val cm:\n {cm}')

        print(f'Val Loss:{val_loss:.6f}, Balanced accuracy: {val_bc:.6f}, accuracy: {val_accuracy:.6f}, [0: {val_nonPed_acc:.4f}({nonPed_acc_num}/{self.val_nonPed_num}), 1: {val_ped_acc:.6f}({ped_acc_num}/{self.val_ped_num}), ({val_correct_num}/{len(self.val_dataset)})]')

        val_epoch_info = {
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_correct_num': val_correct_num,
            'val_bc': val_bc,
            'nonPed_acc_num': nonPed_acc_num,
            'val_nonPed_acc': val_nonPed_acc,
            'ped_acc_num': ped_acc_num,
            'val_ped_acc': val_ped_acc
        }

        val_epoch_info = DotDict(val_epoch_info)
        return val_epoch_info


    def train_model(self):

        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Total EPOCH:', self.epochs)
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.start_epoch, self.epochs):
            # ------------------------ 开始训练 ------------------------
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            train_epoch_info = self.train_one_epoch()
            val_epoch_info = self.val_on_epoch_end(epoch)

            # ------------------------ 训练epoch的callbacks ------------------------
            self.early_stopping(epoch+1, self.ped_model, self.optimizer, val_epoch_info)
            self.epoch_logger(epoch=epoch+1, training_info=train_epoch_info, val_info=val_epoch_info)

            # ------------------------ 学习率调整 ------------------------
            self.lr_decay(epoch + 1)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break


    def test(self):
        pass


    def visualize(self):
        pass






























































