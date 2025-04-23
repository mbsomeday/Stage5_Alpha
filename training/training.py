# # 将上级目录加入 sys.path， 防止命令行运行时找不到包
# import os, sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(curPath)[0]
# sys.path.append(root_path)

import os.path
import torch, torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from data.dataset import my_dataset
from training.train_callbacks import EarlyStopping, Epoch_logger
from utils.utils import get_obj_from_str
from utils.utils import get_vgg_DSmodel, DotDict

from torch.optim import SGD
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class train_ped_model():
    '''
        已将val monitor替换为balanced accuracy
    '''
    def __init__(self, model_name, model, ds_name_list, batch_size=64, epochs=100, save_prefix=None, gen_img=False):
        self.model_name = model_name
        self.model = model
        self.model = self.model.to(DEVICE)
        print(f'model is on {DEVICE}')

        # -------------------- 训练参数设置开始 --------------------
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # self.optimizer = torch.optim.RMSprop([{'params': self.model.parameters(), 'initial_lr': 0.1}], weight_decay=1e-1, eps=0.001)
        # self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=3)   # 是分类任务，所以监控accuracy

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
            self.lr_schedule.step(metrics=val_loss)

            # # 这里放训练epoch的callbacks
            # self.early_stopping(epoch+1, self.model, val_accuracy, self.optimizer)

            # if self.early_stopping.early_stop:
            #     print(f'Early Stopping!')
            #     break



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
    def __init__(self, model_name, model, ds_name_list, camLoss_coefficient=0.1,
                 batch_size=4, epochs=100, save_prefix=None, gen_img=False, reload=None):
        # torch.manual_seed(13)

        self.camLoss_coefficient = camLoss_coefficient

        self.model_name = model_name
        self.model = model
        self.model = self.model.to(DEVICE)

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

        self.train_nonPed_num, self.train_ped_num = self.train_dataset.get_ped_cls_num()
        self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        # ------------ 增加ds model，目的是融入 cam loss ------------
        self.ds_model = get_vgg_DSmodel(device=DEVICE)
        self.ds_model.eval()
        self.ds_model.to(DEVICE)

        self.feed_forward_features = None
        self.backward_features = None

        self.grad_layer = 'features'

        self._register_hooks(self.ds_model, self.grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

        # ------------ 增加ds model代码结束 ------------

        if save_prefix is None:
            save_prefix = model_name
            for ds_name in ds_name_list:
                save_prefix += ds_name
        save_prefix += '_CAMLoss'

        # ------------ callbacks start ------------
        self.early_stopping = EarlyStopping(save_prefix, top_k=3)

        train_num_info = [len(self.train_dataset), self.train_nonPed_num, self.train_ped_num]
        val_num_info = [len(self.val_dataset), self.val_nonPed_num, self.val_ped_num]
        self.training_logger = Training_logger(save_prefix, model_name=model_name, ds_name_list=ds_name_list,
                                               train_num_info=train_num_info, val_num_info=val_num_info)
        # ------------ callbacks end ------------

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
            ckpt = torch.load(reload, map_location=DEVICE, weights_only=False)
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
        y_true = []
        y_pred = []

        for batch, data in enumerate(tqdm(self.train_loader)):
            # 将image和label放到GPU中
            images = data['image']
            labels = data['ped_label']

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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
            masked_images = masked_images.to(DEVICE)
            masked_images = masked_images.type(torch.float32)
            masked_out = self.model(masked_images)

            masked_loss = self.loss_fn(masked_out, labels)

            loss = loss_cls + self.camLoss_coefficient * masked_loss
            # print(f'loss: {loss}, masked_loss: {masked_loss}')

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

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
        training_bc = balanced_accuracy_score(y_true, y_pred)

        print('Training Loss:{:.6f}, Training accuracy:{:.6f}% ({} / {})'.format(training_loss, train_acc_100, training_correct_num,
                                                                       len(self.val_dataset)))
        return train_accuracy, training_loss, training_correct_num, training_bc

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
                masked_images = masked_images.to(DEVICE)
                masked_images = masked_images.type(torch.float32)
                masked_out = self.model(masked_images)

                masked_loss = self.loss_fn(masked_out, labels)

                # ------------  新代码结束 ------------

                loss_cls = self.loss_fn(out, labels)

                loss = loss_cls + self.camLoss_coefficient * masked_loss

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
        val_bc = balanced_accuracy_score(y_true, y_pred)

        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {}), balanced accuracy:{:.6f}%'.format(val_loss, val_acc_100, val_correct_num,
                                                                       len(self.val_dataset), bc))

        return val_loss, val_accuracy, val_correct_num, val_bc


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
            train_accuracy, training_loss, training_correct_num, training_bc = self.train_one_epoch()
            val_loss, val_accuracy, val_correct_num, val_bc = self.val_on_epoch_end(epoch)

            training_info = [train_accuracy, training_loss, training_correct_num, training_bc]
            val_info = [val_loss, val_accuracy, val_correct_num, val_bc]

            # ------------------------ 训练epoch的callbacks ------------------------
            self.early_stopping(epoch+1, self.model, val_accuracy, self.optimizer)
            self.training_logger(training_info=training_info, val_info=val_info)
            # ------------------------ 训练epoch的callbacks ------------------------


            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break


class train_ped_model_alpha():
    def __init__(self, model_obj: str,
                 ds_name_list,
                 batch_size,
                 reload=None,
                 epochs=50,
                 base_lr=0.01,
                 warmup_epochs=0,
                 lr_patience=5,
                 camLoss_coefficient=None,
                 save_best_cls=False):
        '''
        todo: 为什么 camLoss_coefficient 不设置为0？
        :param model_obj: 传入的例子: models.VGG.vgg16_bn
        :param ds_name_list:
        :param warmup_epoch: 若为0，则不用warm up策略
        :param camLoss_coefficient: 若为None，则不用cam loss训练
        '''
        # -------------------- 打印训练信息 --------------------
        print('-' * 20, 'Training Info', '-' * 20)
        print(f'warmup_epochs: {warmup_epochs}')
        print(f'batch_size: {batch_size}')
        print(f'Reload: {reload}')
        print('-' * 20)

        # -------------------- 成员变量 --------------------
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.lr_patience = lr_patience
        self.save_best_cls = save_best_cls
        self.camLoss_coefficient = camLoss_coefficient

        # -------------------- 获取 ped model for train --------------------
        self.model = get_obj_from_str(model_obj)(num_class=2)
        self.model = self.model.to(DEVICE)

        # -------------------- 获取数据 --------------------
        self.ds_name_list = ds_name_list
        self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        self.train_nonPed_num, self.train_ped_num = self.train_dataset.get_ped_cls_num()
        self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        # -------------------- 训练配置 --------------------
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.base_lr, weight_decay=1e-5)
        # self.optimizer = torch.optim.RMSprop([{'params': self.model.parameters(), 'initial_lr': 1e-5}], weight_decay=1e-5, eps=0.001)
        # self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=lr_patience)   # 是分类任务，所以监控accuracy
        # self.lr_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=warmup_epochs, gamma=0.963, last_epoch=self.warmup_epochs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # -------------------- Callbacks --------------------
        # 设置保存训练信息的文件夹的名字
        save_prefix = model_obj.rsplit('.')[-1]
        for ds_name in ds_name_list:
            info = '_' + ds_name
            save_prefix += info

        if self.camLoss_coefficient is not None:
            save_prefix += '_CAMLoss'

        callback_savd_dir = save_prefix

        self.early_stopping = EarlyStopping(save_prefix, top_k=2)

        train_num_info = [len(self.train_dataset), self.train_nonPed_num, self.train_ped_num]
        val_num_info = [len(self.val_dataset), self.val_nonPed_num, self.val_ped_num]

        self.epoch_logger = Epoch_logger(save_dir=callback_savd_dir, model_name=model_obj.split('.')[-1],
                                         ds_name_list=ds_name_list, train_num_info=train_num_info, val_num_info=val_num_info,
                                         task='ped_cls'
                                         )

        # -------------------- 获取ds model，目的是融入 cam loss --------------------
        if self.camLoss_coefficient is not None:
            self.ds_model = get_vgg_DSmodel()
            self.ds_model.eval()
            self.ds_model = self.ds_model.to(DEVICE)

            self.feed_forward_features = None
            self.backward_features = None

            self.grad_layer = 'features'
            self._register_hooks(self.ds_model, self.grad_layer)

            # sigma, omega for making the soft-mask
            self.sigma = 0.25
            self.omega = 100

        # -------------------- 如果reload，optmizer，start_epoch等也要重新设置 --------------------
        if reload is not None:
            print(f'Reloading weights from {reload}')
            ckpt = torch.load(reload, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch']
            self.early_stopping.best_val_acc = ckpt['best_val_acc']
        else:
            self.start_epoch = 0


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

    def train_one_epoch(self):
        '''
            train one batch
            返回 train_epoch_info 字典
        '''
        self.model.train()

        training_loss = 0.0
        training_correct_num = 0
        y_true = []
        y_pred = []

        nonPed_acc_num = 0
        ped_acc_num = 0

        for batch, data in enumerate(tqdm(self.train_loader)):
            images = data['image']
            labels = data['ped_label']
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = self.model(images)
            _, pred = torch.max(out, 1)

            loss_cls = self.loss_fn(out, labels)

            # # ------------ 计算 cam loss ------------
            # # 生成masked image，这里只对non ped进行mask，因为mask对ped的效果不好
            # nonPed_idx = labels == 0
            # nonPed_images = images[nonPed_idx]
            # if self.camLoss_coefficient is not None and nonPed_images.shape[0] > 0:
            #     masked_images = np.zeros(shape=nonPed_images.shape)
            #     for img_idx, image in enumerate(nonPed_images):
            #         image = torch.unsqueeze(image, dim=0)
            #         heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
            #         masked_images[img_idx] = masked_image.cpu().detach()
            #     masked_images = torch.tensor(masked_images)
            #     masked_images = masked_images.to(DEVICE)
            #     masked_images = masked_images.type(torch.float32)
            #     masked_out = self.model(masked_images)
            #
            #     masked_loss = self.loss_fn(masked_out, labels[nonPed_idx])
            #     loss = loss_cls + self.camLoss_coefficient * masked_loss
            # else:
            #     loss = loss_cls

            loss = loss_cls         # baseline时不计算camloss

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            # 用于loss记录
            training_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

            # ------------ 计算各个类别的正确个数 ------------
            nonPed_idx = labels == 0
            nonPed_acc = (labels[nonPed_idx] == pred[nonPed_idx]) * 1
            nonPed_acc_num += nonPed_acc.sum()

            ped_idx = labels == 1
            ped_acc = (labels[ped_idx] == pred[ped_idx]) * 1
            ped_acc_num += ped_acc.sum()

            # break

        train_accuracy = training_correct_num / len(self.train_dataset)
        training_bc = balanced_accuracy_score(y_true, y_pred)

        train_nonPed_acc = nonPed_acc_num / self.train_nonPed_num
        train_ped_acc = ped_acc_num / self.train_ped_num

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
        self.model.eval()

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

                out = self.model(images)
                _, pred = torch.max(out, 1)

                loss_cls = self.loss_fn(out, labels)

                # ------------ 计算 cam loss ------------
                # 生成masked image，这里只对non ped进行mask，因为mask对ped的效果不好
                nonPed_idx = labels == 0
                nonPed_images = images[nonPed_idx]
                if self.camLoss_coefficient is not None and nonPed_images.shape[0] > 0:
                    masked_images = np.zeros(shape=nonPed_images.shape)
                    for img_idx, image in enumerate(nonPed_images):
                        image = torch.unsqueeze(image, dim=0)
                        heatmap, mask, masked_image = self.calc_cam(self.ds_model, image)
                        masked_images[img_idx] = masked_image.cpu().detach()
                    masked_images = torch.tensor(masked_images)
                    masked_images = masked_images.to(DEVICE)
                    masked_images = masked_images.type(torch.float32)
                    masked_out = self.model(masked_images)

                    masked_loss = self.loss_fn(masked_out, labels[nonPed_idx])

                    loss = loss_cls + self.camLoss_coefficient * masked_loss
                else:
                    loss = loss_cls

                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                # ------------ 计算各个类别的正确个数 ------------
                nonPed_idx = labels == 0
                nonPed_acc = (labels[nonPed_idx] == pred[nonPed_idx]) * 1
                nonPed_acc_num += nonPed_acc.sum()

                ped_idx = labels == 1
                ped_acc = (labels[ped_idx] == pred[ped_idx]) * 1
                ped_acc_num += ped_acc.sum()

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        val_nonPed_acc = nonPed_acc_num / self.val_nonPed_num
        val_ped_acc = ped_acc_num / self.val_ped_num

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


    def lr_decay(self, epoch):
        # warm-up阶段
        if (epoch + 1) <= self.warmup_epochs:        # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.base_lr * 0.963 ** ((epoch + 1) / 3)        # gamma=0.963, lr decay epochs=3

        # else:       # monitored metric持续几个epoch不变，lr decay阶段,加入了ped和nonPed的count
        #     if self.early_stopping.counter > self.lr_patience:
        #         self.optimizer.param_groups[0]['lr'] *= 0.5
        #     elif self.early_stopping.save_best_cls_model and self.early_stopping.save_nonPed_info.counter > self.lr_patience:
        #         self.optimizer.param_groups[0]['lr'] *= 0.5
        #     elif self.early_stopping.save_best_cls_model and self.early_stopping.save_ped_info.counter > self.lr_patience:
        #         self.optimizer.param_groups[0]['lr'] *= 0.5


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
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            train_epoch_info = self.train_one_epoch()
            val_epoch_info = self.val_on_epoch_end(epoch)

            # ------------------------ 训练epoch的callbacks ------------------------
            self.early_stopping(epoch+1, self.model, self.optimizer, val_epoch_info)
            self.epoch_logger(epoch=epoch+1, training_info=train_epoch_info, val_info=val_epoch_info)

            # ------------------------ 学习率调整 ------------------------
            self.lr_decay(epoch)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break



class train_ds_model_alpha():
    def __init__(self, model_obj: str, ds_name_list, batch_size, epochs=50, reload=None, base_lr=0.256, warmup_epochs=0, lr_patience=5):
        super().__init__()

        # -------------------- 成员变量 --------------------
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.lr_patience = lr_patience

        # -------------------- 获取 ped model for train --------------------
        print(f'model_obj:{model_obj}')
        self.model = get_obj_from_str(model_obj)(num_class=4)
        self.model = self.model.to(DEVICE)

        # -------------------- 获取数据 --------------------
        self.ds_name_list = ds_name_list

        self.train_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, path_key='org_dataset', txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # -------------------- 训练配置 --------------------
        # # vgg16
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9)
        # self.loss_fn = torch.nn.CrossEntropyLoss()

        # efficientNet
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.base_lr, weight_decay=0.9, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)  # 学习率衰减策略
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # -------------------- Callbacks --------------------
        save_prefix = model_obj.split('.')[-1] + '_dsCls'
        callback_savd_dir = save_prefix

        self.early_stopping = EarlyStopping(save_prefix, top_k=3, model_save_dir=callback_savd_dir, save_best_cls=False)
        train_num_info = [len(self.train_dataset), -1, -1]
        val_num_info = [len(self.val_dataset), -1, -1]

        self.epoch_logger = Epoch_logger(save_dir=callback_savd_dir, model_name=model_obj.split('.')[-1],
                                         ds_name_list=ds_name_list, train_num_info=train_num_info, val_num_info=val_num_info,
                                         task='ds_cls'
                                         )

        # -------------------- 如果reload，optmizer，start_epoch等也要重新设置 --------------------
        if reload is not None:
            print(f'Reloading weights from {reload}')
            ckpt = torch.load(reload, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch']
            self.early_stopping.best_val_acc = ckpt['best_val_bc']
        else:
            self.start_epoch = 0

    # 自定义衰减函数
    def lr_lambda(self, epoch):
        decay_rate = 0.97
        decay_epochs = 2.4
        return decay_rate ** (epoch / decay_epochs)

    def train_one_epoch(self):
        self.model.train()

        training_loss = 0.0
        training_correct_num = 0
        y_true = []
        y_pred = []

        for batch, data in enumerate(tqdm(self.train_loader)):
            images = data['image']
            labels = data['ds_label']
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = self.model(images)
            _, pred = torch.max(out, 1)

            loss = self.loss_fn(out, labels)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            training_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

        train_accuracy = training_correct_num / len(self.train_dataset)
        training_bc = balanced_accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        print(f'training cm:\n {cm}')

        print(f'Training Loss:{training_loss:.6f}, Balanced accuracy: {training_bc:.6f}, accuracy: {train_accuracy:.6f}')

        train_epoch_info = {
            'train_accuracy': train_accuracy,
            'training_loss': training_loss,
            'training_correct_num': training_correct_num,
            'training_bc': training_bc,
        }

        train_epoch_info = DotDict(train_epoch_info)

        return train_epoch_info
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
                _, pred = torch.max(out, 1)

                loss = self.loss_fn(out, labels)

                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        print(f'val cm:\n {cm}')

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        print(f'Val Loss:{val_loss:.6f}, Balanced accuracy: {val_bc:.6f}, accuracy: {val_accuracy:.6f}')

        val_epoch_info = {
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_correct_num': val_correct_num,
            'val_bc': val_bc,
        }

        val_epoch_info = DotDict(val_epoch_info)

        return val_epoch_info

    def lr_decay(self, epoch):
        # warm-up阶段
        if (epoch + 1) <= self.warmup_epochs:
            self.optimizer.param_groups[0]['lr'] = self.base_lr * (epoch + 1) / self.warmup_epochs
        # monitored metric持续几个epoch不变，lr decay阶段
        else:
            # EfficientNet
            self.scheduler.step()
            # # VGG16
            # if self.early_stopping.counter > self.lr_patience:
            #     self.optimizer.param_groups[0]['lr'] *= 0.5

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
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            # ------------------------ 学习率调整 ------------------------
            self.lr_decay(epoch + 1)

            # ------------------------ 开始训练 ------------------------
            train_epoch_info = self.train_one_epoch()
            val_epoch_info = self.val_on_epoch_end(epoch)

            # ------------------------ 训练epoch的callbacks ------------------------
            self.early_stopping(epoch+1, self.model, self.optimizer, val_epoch_info)
            self.epoch_logger(epoch=epoch+1, training_info=train_epoch_info, val_info=val_epoch_info)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break








# if __name__ == '__main__':
#     # print('a')
#     from models.VGG import vgg16_bn
#     from utils.utils import get_obj_from_str
#     import math
#
#     test_alpha = train_ped_model_alpha(model_obj='models.VGG.vgg16_bn', ds_name_list=['D3'], batch_size=4, reload=None,
#                                        save_prefix=None,
#                                        )



















