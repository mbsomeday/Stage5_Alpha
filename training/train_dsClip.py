import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from data.dataset import dataset_clip
from utils.utils import DotDict
from training.train_callbacks import EarlyStopping, Epoch_logger

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class train_clipDS_model():
    def __init__(self,
                 batch_size,
                 epochs=200,
                 reload=None,
                 base_lr=0.01,
                 warmup_epochs=0,
                 lr_patience=5):

        super().__init__()
        # -------------------- 打印训练信息 --------------------
        print('-' * 20, 'Training Info', '-' * 20)
        print(f'Task: dataset classification')
        print(f'warmup_epochs: {warmup_epochs}')
        print(f'batch_size: {batch_size}')
        print(f'Reload: {reload}')
        print('-' * 20)

        # -------------------- 成员变量 --------------------
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.lr_patience = lr_patience

        # -------------------- 获取 ped model for train --------------------
        self.model = models.vgg16(num_classes=4, weights=None).to(DEVICE)

        # -------------------- 获取数据 --------------------
        # self.ds_name_list = ds_name_list
        self.ds_name_list = ['D1', 'D2', 'D3', 'D4']
        path_key = 'tiny_dataset'

        self.train_dataset = dataset_clip(self.ds_name_list, path_key, txt_name='train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = dataset_clip(self.ds_name_list, path_key, txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # -------------------- 训练配置 --------------------
        # # vgg16
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # -------------------- Callbacks --------------------
        save_prefix = 'vgg16_CropDSCls'
        callback_savd_dir = save_prefix

        self.early_stopping = EarlyStopping(save_prefix, top_k=2, patience=15)
        train_num_info = [len(self.train_dataset), -1, -1]
        val_num_info = [len(self.val_dataset), -1, -1]

        self.epoch_logger = Epoch_logger(save_dir=callback_savd_dir, model_name='vgg16',
                                         ds_name_list=self.ds_name_list, train_num_info=train_num_info, val_num_info=val_num_info,
                                         task='ds_cls'
                                         )
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
            images = data['clip'].to(DEVICE)
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

        cm = confusion_matrix(y_true, y_pred)
        print(f'training cm:\n {cm}')
        bc = balanced_accuracy_score(y_true, y_pred)

        print(f'Training Loss:{training_loss:.6f}, accuracy: {train_accuracy:.6f}')

        train_epoch_info = {
            'train_accuracy': train_accuracy,
            'training_loss': training_loss,
            'training_bc': bc,
            'training_correct_num': training_correct_num,
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
                images = data['clip'].to(DEVICE)
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
        bc = balanced_accuracy_score(y_true, y_pred)


        val_accuracy = val_correct_num / len(self.val_dataset)

        print(f'Val Loss:{val_loss:.6f}, accuracy: {val_accuracy:.6f}')

        val_epoch_info = {
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_bc': bc,
            'val_correct_num': val_correct_num,
        }

        val_epoch_info = DotDict(val_epoch_info)

        return val_epoch_info

    def lr_decay(self, epoch):
        # warm-up阶段
        if epoch <= self.warmup_epochs:        # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.base_lr * epoch / self.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.base_lr * 0.963 ** (epoch / 3)        # gamma=0.963, lr decay epochs=3

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
            self.early_stopping(epoch+1, self.model, self.optimizer, val_epoch_info)
            self.epoch_logger(epoch=epoch+1, training_info=train_epoch_info, val_info=val_epoch_info)

            # ------------------------ 学习率调整 ------------------------
            self.lr_decay(epoch + 1)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break



# if __name__ == '__main__':
#     tt = train_clipDS_model(batch_size=48)
#     tt.train_model()



























