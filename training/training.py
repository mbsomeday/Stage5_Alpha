import os.path
import torch, torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

from data.dataset import my_dataset
from training.train_callbacks import EarlyStopping
from training.grad_loss import GradCAM
from configs.paths_dict import PATHS
from models.VGG import vgg16_bn
from utils.utils import get_vgg_DSmodel

from configs.paths_dict import get_device
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# def load_weights(model, weights):
#     ckpt = torch.load(weights, weights_only=False, map_location=DEVICE)
#     model.load_state_dict(ckpt['model_state_dict'])
#     print(f'Loading weights from {weights}')
#     return model
#
#
# def get_ds_models():
#     '''
#         加载vgg ds model
#     :return:
#     '''
#     print(f'DEVICE:{DEVICE}')
#     ds_model = vgg16_bn(num_class=4)
#     ds_weights = PATHS['ds_cls_ckpt']
#     ds_model = load_weights(ds_model, ds_weights)
#
#     # ds_model = models.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
#     # new_classifier = torch.nn.Sequential(
#     #     torch.nn.Dropout(p=0.2, inplace=True),
#     #     torch.nn.Linear(in_features=1280, out_features=4)
#     # )
#     # ds_model.classifier = new_classifier
#
#     return ds_model


class train_ped_model():
    '''
        已将val monitor替换为balanced accuracy
    '''
    def __init__(self, model_name, model, ds_name_list, batch_size=4, epochs=100, save_prefix=None, gen_img=False):
        self.model_name = model_name
        self.model = model
        self.model = self.model.to(DEVICE)
        print(f'model is on {DEVICE}')

        # -------------------- 训练参数设置开始 --------------------
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # -------------------- 训练参数设置结束 --------------------

        self.epochs = epochs
        self.ds_name_list = ds_name_list

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # callbacks
        if save_prefix is None:
            save_prefix = model_name
            for ds_name in ds_name_list:
                save_prefix += ds_name
        self.early_stopping = EarlyStopping(save_prefix, top_k=2)
        self.gen_img = gen_img
        if self.gen_img and batch_size >= 4:
            print(f'Image will be saved after each epoch.')
            self.image_logger_dir = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(self.image_logger_dir):
                os.mkdir(self.image_logger_dir)

    def train_one_epoch(self):

        self.model.train()

        training_loss = 0.0
        training_correct_num = 0

        for batch, data in enumerate(tqdm(self.train_loader)):
            # 将image和label放到GPU中
            images = data['image']
            labels = data['ped_label']

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = self.model(images)

            loss = self.loss_fn(out, labels)
            training_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

        train_accuracy = training_correct_num / len(self.train_dataset)
        train_acc_100 = train_accuracy * 100

        print('Training Loss:{:.6f}, Training accuracy:{:.6f}% ({} / {})'.format(training_loss, train_acc_100, training_correct_num,
                                                                       len(self.val_dataset)))

    def val_on_epoch_end(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_correct_num = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['ped_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                out = self.model(images)
                loss = self.loss_fn(out, labels)

                _, pred = torch.max(out, 1)
                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

            if self.gen_img:
                grid_images = torch.cat(list(images[:4]), dim=2)
                grid_recons = torch.cat(list(out[:4]), dim=2)
                grid_img = torch.cat((grid_images, grid_recons), dim=1)
                grid = torchvision.utils.make_grid(grid_img, nrow=4)
                image_save_name = '{}.jpg'.format(epoch)
                torchvision.utils.save_image(grid, image_save_name)

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_acc_100 = val_accuracy * 100
        bc = balanced_accuracy_score(y_true, y_pred)

        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {}), balanced accuracy:{:.6f}%'.format(val_loss, val_acc_100, val_correct_num,
                                                                                            len(self.val_dataset), bc))

        return val_loss, val_accuracy, bc

    def train_model(self):

        self.model.to(DEVICE)

        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Maximum EPOCH:', self.epochs)
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            self.train_one_epoch()
            val_loss, val_accuracy, balanced_acc = self.val_on_epoch_end(epoch)

            # 这里放训练epoch的callbacks
            self.early_stopping(epoch+1, self.model, val_accuracy, self.optimizer)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break


class train_ds_model():
    def __init__(self, model_name, model, batch_size=4, epochs=50, reload=None):
        self.model_name = model_name
        self.model = model
        self.model = self.model.to(DEVICE)
        self.epochs = epochs

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        self.ds_name_list = ['D1', 'D2', 'D3', 'D4']
        self.train_dataset = my_dataset(self.ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(self.ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        self.start_epoch = 0

        # callbacks
        self.save_prefix = 'EfficientB0_dsCls'
        self.early_stopping = EarlyStopping(self.save_prefix, top_k=2)

        self.image_logger_dir = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(self.image_logger_dir):
            os.mkdir(self.image_logger_dir)

        # 如果中断后重新训练
        if reload is not None:
            print(f'Reloading weights from {reload}')
            ckpt = torch.load(reload, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch']
            self.early_stopping.best_val_acc = ckpt['best_val_acc']

    def train_one_epoch(self):
        self.model.train()

        training_loss = 0.0
        training_correct_num = 0

        for batch, data in enumerate(tqdm(self.train_loader)):
            # 将image和label放到GPU中
            images = data['image']
            labels = data['ds_label']

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = self.model(images)
            loss = self.loss_fn(out, labels)
            training_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

        train_accuracy = training_correct_num / len(self.train_dataset)
        train_acc_100 = train_accuracy * 100

        print('Training Loss:{:.6f}, Training accuracy:{:.6f}% ({} / {})'.format(training_loss, train_acc_100, training_correct_num,
                                                                       len(self.val_dataset)))

    def val_on_epoch_end(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_correct_num = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['ds_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                out = self.model(images)
                loss = self.loss_fn(out, labels)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                _, pred = torch.max(out, 1)
                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_acc_100 = val_accuracy * 100
        bc = balanced_accuracy_score(y_true, y_pred)

        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {}), balanced accuracy:{:.6f}%'.format(val_loss, val_acc_100, val_correct_num,
                                                                       len(self.val_dataset), bc))

        return val_loss, val_accuracy


    def train_model(self):

        self.model.to(DEVICE)

        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Total EPOCH:', self.epochs)
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.start_epoch, self.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            self.train_one_epoch()
            val_loss, val_accuracy = self.val_on_epoch_end(epoch)

            # 这里放训练epoch的callbacks
            self.early_stopping(epoch+1, self.model, val_accuracy, self.optimizer)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break


class TemporaryGrad(object):
    '''
    https://blog.csdn.net/qq_44980390/article/details/123672147
    '''
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class train_pedmodel_camLoss():
    def __init__(self, model_name, model, ds_name_list, batch_size=4, epochs=100, save_prefix=None, gen_img=False, reload=None, gid=None):
        torch.manual_seed(13)

        self.device = get_device(gid)
        print(f'model is on {self.device}')

        self.model_name = model_name
        self.model = model
        self.model = self.model.to(self.device)

        self.epochs = epochs
        self.ds_name_list = ds_name_list

        # -------------------- 训练参数设置开始 --------------------
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # -------------------- 训练参数设置结束 --------------------

        # 此处修改loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # ------------ 新增加代码，目的是融入 cam loss ------------
        self.ds_model = get_vgg_DSmodel()
        self.ds_model.eval()
        self.ds_model.to(self.device)

        self.feed_forward_features = None
        self.backward_features = None

        self.grad_layer = 'features'

        self._register_hooks(self.ds_model, self.grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

        # ------------ 新代码结束 ------------

        # callbacks
        if save_prefix is None:
            save_prefix = model_name
            for ds_name in ds_name_list:
                save_prefix += ds_name
        self.early_stopping = EarlyStopping(save_prefix, top_k=2)
        self.gen_img = gen_img
        if self.gen_img and batch_size >= 4:
            print(f'Image will be saved after each epoch.')
            self.image_logger_dir = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(self.image_logger_dir):
                os.mkdir(self.image_logger_dir)
        self.start_epoch = 0
        # 如果中断后重新训练
        if reload is not None:
            print(f'Reloading weights from {reload}')
            ckpt = torch.load(reload, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch']
            self.early_stopping.best_val_acc = ckpt['best_val_acc']

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

    def plt_format(self, x):
        '''
            x是 3-D
        '''
        x = x.detach().numpy()
        ret_x = np.transpose(x, (1, 2, 0))
        return ret_x

    def train_one_epoch(self):

        self.model.train()

        training_loss = 0.0
        training_correct_num = 0

        for batch, data in enumerate(tqdm(self.train_loader)):
            # 将image和label放到GPU中
            images = data['image']
            labels = data['ped_label']

            images = images.to(self.device)
            labels = labels.to(self.device)

            out = self.model(images)

            loss_cls = self.loss_fn(out, labels)

            # ------------ todo 新增加代码，目的是融入 cam loss ------------

            # 生成masked image
            masked_images = np.zeros(shape=images.shape)
            # heatmap_list = []
            for img_idx, image in enumerate(images):
                image = torch.unsqueeze(image, dim=0)
                heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
                masked_images[img_idx] = masked_image.cpu().detach()
                # heatmap_list.append(heatmap)

            # masked_images = torch.tensor(masked_images)
            # print('heatmap_list[0]', heatmap_list[0].shape)
            #
            # img_list = [self.plt_format(images[0]),
            #             self.plt_format(heatmap_list[0][0]),
            #             self.plt_format(masked_images[0]),
            #
            #             self.plt_format(images[1]),
            #             self.plt_format(heatmap_list[1][0]),
            #             self.plt_format(masked_images[1])
            #             ]
            # # name_list = ['org', 'heatmap', 'masked_images']
            # plt.figure()
            # print(len(img_list) + 1)
            # for i in range(1, len(img_list) + 1):
            #     plt.subplot(2, 3, i)
            #     # plt.title(name_list[i - 1])
            #     plt.imshow(img_list[i - 1])
            # plt.show()
            masked_images = torch.tensor(masked_images)
            masked_images = masked_images.to(self.device)
            masked_images = masked_images.type(torch.float32)
            masked_out = self.model(masked_images)

            masked_loss = self.loss_fn(masked_out, labels)

            loss = loss_cls + masked_loss
            # print(f'loss: {loss}, masked_loss: {masked_loss}')

            training_loss += loss.item()

            # 将masked image输入ped model计算loss

            # ------------ todo 新代码结束 ------------

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

            # break

        train_accuracy = training_correct_num / len(self.train_dataset)
        train_acc_100 = train_accuracy * 100

        print('Training Loss:{:.6f}, Training accuracy:{:.6f}% ({} / {})'.format(training_loss, train_acc_100, training_correct_num,
                                                                       len(self.val_dataset)))

    def val_on_epoch_end(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_correct_num = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['ped_label']
                images = images.to(self.device)
                labels = labels.to(self.device)

                out = self.model(images)

                # ------------  新增加代码 val，目的是融入 cam loss ------------
                # 在val中也加入cam loss
                masked_images = np.zeros(shape=images.shape)
                # heatmap_list = []
                for img_idx, image in enumerate(images):
                    image = torch.unsqueeze(image, dim=0)
                    heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
                    masked_images[img_idx] = masked_image.cpu().detach()
                    # heatmap_list.append(heatmap)

                masked_images = torch.tensor(masked_images)
                masked_images = masked_images.to(self.device)
                masked_images = masked_images.type(torch.float32)
                masked_out = self.model(masked_images)

                masked_loss = self.loss_fn(masked_out, labels)

                # ------------  新代码结束 ------------

                loss_cls = self.loss_fn(out, labels)

                loss = loss_cls + masked_loss

                _, pred = torch.max(out, 1)
                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

            if self.gen_img:
                grid_images = torch.cat(list(images[:4]), dim=2)
                grid_recons = torch.cat(list(out[:4]), dim=2)
                grid_img = torch.cat((grid_images, grid_recons), dim=1)
                grid = torchvision.utils.make_grid(grid_img, nrow=4)
                image_save_name = '{}.jpg'.format(epoch)
                torchvision.utils.save_image(grid, image_save_name)

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_acc_100 = val_accuracy * 100
        bc = balanced_accuracy_score(y_true, y_pred)

        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {}), balanced accuracy:{:.6f}%'.format(val_loss, val_acc_100, val_correct_num,
                                                                       len(self.val_dataset), bc))

        return val_loss, val_accuracy, bc


    def train_model(self):

        self.model.to(self.device)

        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Total EPOCH:', self.epochs)
        print('Runing device:', self.device)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.start_epoch, self.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            self.train_one_epoch()
            val_loss, val_accuracy, balanced_acc = self.val_on_epoch_end(epoch)

            # 这里放训练epoch的callbacks
            self.early_stopping(epoch+1, self.model, val_accuracy, self.optimizer)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break

















