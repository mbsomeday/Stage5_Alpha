import os.path
import torch, torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import my_dataset
from training.train_callbacks import EarlyStopping

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class train_model():
    def __init__(self, model_name, model, ds_name_list, batch_size=4, epochs=100, save_prefix=None, gen_img=False):
        self.model_name = model_name
        self.model = model
        self.model = self.model.to(DEVICE)
        print(f'model is on {DEVICE}')

        self.epochs = epochs
        self.ds_name_list = ds_name_list

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

            if self.gen_img:
                grid_images = torch.cat(list(images[:4]), dim=2)
                grid_recons = torch.cat(list(out[:4]), dim=2)
                grid_img = torch.cat((grid_images, grid_recons), dim=1)
                grid = torchvision.utils.make_grid(grid_img, nrow=4)
                image_save_name = '{}.jpg'.format(epoch)
                torchvision.utils.save_image(grid, image_save_name)

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_acc_100 = val_accuracy * 100
        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {})'.format(val_loss, val_acc_100, val_correct_num,
                                                                       len(self.val_dataset)))

        return val_loss, val_accuracy


    def train(self):

        self.model.to(DEVICE)

        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Total EPOCH:', self.epochs)
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            self.train_one_epoch()
            val_loss, val_accuracy = self.val_on_epoch_end(epoch)

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

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images = data['image']
                labels = data['ds_label']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                out = self.model(images)
                loss = self.loss_fn(out, labels)

                _, pred = torch.max(out, 1)
                val_correct_num += (pred == labels).sum()
                val_loss += loss.item()

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_acc_100 = val_accuracy * 100
        print('Val Loss:{:.6f}, Val accuracy:{:.6f}% ({} / {})'.format(val_loss, val_acc_100, val_correct_num,
                                                                       len(self.val_dataset)))

        return val_loss, val_accuracy


    def train(self):

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



















