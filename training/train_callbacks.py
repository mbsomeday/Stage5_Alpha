import os, torch, sys, math
import logging
import numpy as np
import torch.optim as optim


class EarlyStopping():
    '''
        保存当前为止最好的模型(loss最低)，
        当loss稳定不变patience个epoch时，结束训练
    '''

    def __init__(self, save_prefix, top_k=3, patience=10, delta=0.000001,
                 model_save_dir=None, warmup_epochs=0):
        '''
            这个 early stopping关注的是 accuracy
            model save name: prefix_{epoch}_{acc}.pth
        '''

        self.top_k = top_k
        self.save_prefix = save_prefix
        self.warmup_epochs = warmup_epochs

        if model_save_dir is not None:
            self.model_save_dir = model_save_dir
        else:
            self.model_save_dir = os.path.join(os.getcwd(), save_prefix+'_ckpt')

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        self.patience = patience
        self.counter = 0  # 记录loss不变的epoch数目
        self.early_stop = False # 是否停止训练
        self.best_val_acc = -np.inf
        self.delta = delta

        print('-' * 20 + 'Early Stopping Info' + '-' * 20)
        print('Create early stopping, monitoring [validation accuracy] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {save_prefix}_epoch_acc.pth')
        print(f'Early Stop with patience: {self.patience} ')

        msg = f'The best {self.top_k} models will be saved to {self.model_save_dir}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    def __call__(self, epoch, model, val_acc, optimizer, lr_schedule=None):
        # 表现没有超过best
        if val_acc < self.best_val_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # 比best表现好
        else:
            self.save_checkpoint(val_acc, model, optimizer, epoch)
            self.counter = 0

        if epoch > self.warmup_epochs:
            lr_schedule.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']

        msg = f'Epoch:{epoch}, lr:{current_lr}, counter:{self.counter}/{self.patience}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    # 删除多余的权重文件
    def del_redundant_weights(self):
        # 删除已经有的文件,只保留n+1个模型
        all_weights_temp = os.listdir(self.model_save_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        # 按存储格式来： save_name = prefix_{epoch}_{acc}.pth
        if len(all_weights) > self.top_k:
            sorted = []
            for weight in all_weights:
                val_acc = weight.split('-')[-1]
                sorted.append((weight, val_acc))

            sorted.sort(key=lambda w: w[1], reverse=False)
            print('After sorting:', sorted)

            del_path = os.path.join(self.model_save_dir, sorted[0][0])
            os.remove(del_path)
            print('Del file:', del_path)

    def save_checkpoint(self, val_acc, model, optimizer, epoch):
        '''Saves model when validation loss decrease.'''

        print(f'Validation accuracy increased ({self.best_val_acc:.6f} --> {val_acc:.6f}).  Saving model ...')

        self.del_redundant_weights()
        save_name = f"{self.save_prefix}-{epoch:03d}-{val_acc:.6f}.pth"
        self.best_val_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }

        # 存储权重
        save_path = os.path.join(self.model_save_dir, save_name)
        torch.save(checkpoint, save_path)


class ImageLogger():
    def __init__(self):
        pass


class Epoch_logger():
    '''
        用于记录训练过程中的loss，accuracy变化情况
    '''
    def __init__(self, save_dir, model_name, ds_name_list, train_num_info, val_num_info):
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.ds_name_list = ds_name_list

        # 获取数据集的总量，各个类别的量
        self.train_num, self.train_nonPed_num, self.train_ped_num = train_num_info
        self.val_num, self.val_nonPed_num, self.val_ped_num = val_num_info

        self.txt_path = os.path.join(self.save_dir, 'train_info.txt')

        # todo 训练时取消下列注释
        __stderr__ = sys.stderr  # 将当前默认的错误输出结果保存为__stderr__
        sys.stderr = open(os.path.join(self.save_dir, 'errorLog.txt'), 'a')  # 将后续的报错信息写入对应的文件中
        # assert not os.path.exists(self.txt_path), f'The {self.txt_path} already exists, please chcek!'

        # 在文件的开头写入训练的信息
        with open(self.txt_path, 'a') as f:
            msg = f'Model: {model_name}, Training on datasets: {self.ds_name_list}\n'
            f.write(msg)


    def __call__(self, epoch, training_info, val_info):

        train_nonPed_acc = training_info.nonPed_acc_num / self.train_nonPed_num
        train_ped_acc = training_info.ped_acc_num / self.train_ped_num
        val_nonPed_acc = val_info.nonPed_acc_num / self.val_nonPed_num
        val_ped_acc = val_info.ped_acc_num / self.val_ped_num

        train_msg = f'Training Loss:{training_info.training_loss:.6f}, Balanced accuracy: {training_info.training_bc:.6f}, accuracy: {training_info.train_accuracy:.6f}, [0: {train_nonPed_acc:.4f}({training_info.nonPed_acc_num}/{self.train_nonPed_num}), 1: {train_ped_acc:.4f}({training_info.ped_acc_num}/{self.train_ped_num}), all: ({training_info.training_correct_num}/{self.train_num})]\n'

        val_msg = f'Val Loss:{val_info.val_loss:.6f}, Balanced accuracy: {val_info.val_bc:.6f}, accuracy: {val_info.val_accuracy:.6f}, [0: {val_nonPed_acc:.4f}({val_info.nonPed_acc_num}/{self.val_nonPed_num}), 1: {val_ped_acc:.4f}({val_info.ped_acc_num}/{self.val_ped_num}), all: ({val_info.val_correct_num}/{self.val_num})]\n'

        with open(self.txt_path, 'a') as f:
            f.write(f'Epoch: {epoch + 1}\n')
            f.write(train_msg)
            f.write(val_msg)























