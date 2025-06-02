import os, torch, sys, math
import logging
import numpy as np
import torch.optim as optim

from utils.utils import DotDict


class EarlyStopping():
    def __init__(self, save_prefix,
                 top_k=2,
                 patience=10,
                 delta=0.00001):
        '''
        :param save_prefix: 存储前缀，例子：EfficientB0_D1，也用于创建save dir
        :param top_k: 保存几个最好模型
        :param patience: 当监控的 metric 连续 patience 个 epoch 不增加，则触发early stopping
        :param delta: 监控metric增加的最小值，当超过该值的时候表示模型有进步
        '''

        self.top_k = top_k
        self.save_prefix = save_prefix
        self.cur_epoch = 0

        self.model_save_dir = os.path.join(os.getcwd(), save_prefix)

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        self.patience = patience
        self.counter = 0            # 记录loss不变的epoch数目
        self.early_stop = False     # 是否停止训练
        self.best_val_acc = -np.inf
        self.delta = delta

        print('-' * 20 + 'Early Stopping Info' + '-' * 20)
        print('Create early stopping, monitoring [validation balanced accuracy] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {save_prefix}_epoch_acc.pth')
        print(f'Early Stop with patience: {self.patience}')

        msg = f'The best {self.top_k} models will be saved to {self.model_save_dir}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    def __call__(self, epoch, model, optimizer, val_epoch_info):
        '''
            目的是monitor总体及各类别的accuracy
        '''
        self.cur_epoch = epoch
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Current lr: {cur_lr}')

        if val_epoch_info.val_bc < self.best_val_acc + self.delta:       # 表现没有提升的情况
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
        else:       # 表现提升
            metrics = [self.best_val_acc, val_epoch_info.val_bc]
            self.save_checkpoint(model=model, metrics=metrics, optimizer=optimizer, ckpt_dir=self.model_save_dir)
            self.counter = 0

        # 根据counter判断是否设置停止flag
        if self.counter >= self.patience:
            self.early_stop = True

        # 记录earlystop信息
        msg = f"Epoch:{epoch}, overall counter:{self.counter}/{self.patience}, current lr: {cur_lr}\n"
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)


    def del_redundant_weights(self, ckpt_dir):
        all_weights_temp = os.listdir(ckpt_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        # 按存储格式来： save_name = prefix_{epoch}_{balanced_acc}.pth
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


    def save_checkpoint(self, model, metrics, optimizer, ckpt_dir):

        print(f'Performance increases ({metrics[0]} --> {metrics[1]}). Saving Model.')

        self.del_redundant_weights(ckpt_dir)
        save_name = f"{self.save_prefix}-{self.cur_epoch:02d}-{metrics[1]:.5f}.pth"     # 格式：prefix_{epoch}_{balanced_acc}.pth
        self.best_val_acc = metrics[1]

        checkpoint = {
            'epoch': self.cur_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_bc': self.best_val_acc,
        }

        save_path = os.path.join(ckpt_dir, save_name)
        torch.save(checkpoint, save_path)



class ImageLogger():
    def __init__(self):
        pass


class Epoch_logger():
    '''
        用于记录训练过程中的loss，accuracy变化情况
    '''
    def __init__(self, save_dir, model_name, ds_name_list, train_num_info, val_num_info, task='ped_cls'):
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.ds_name_list = ds_name_list
        self.task = task

        # 获取数据集的总量，各个类别的量
        self.train_num, self.train_nonPed_num, self.train_ped_num = train_num_info
        self.val_num, self.val_nonPed_num, self.val_ped_num = val_num_info

        self.txt_path = os.path.join(self.save_dir, 'train_info.txt')

        # 注：训练时取消下列注释
        __stderr__ = sys.stderr  # 将当前默认的错误输出结果保存为__stderr__
        # sys.stderr = open(os.path.join(self.save_dir, 'errorLog.txt'), 'a')  # 将后续的报错信息写入对应的文件中
        # assert not os.path.exists(self.txt_path), f'The {self.txt_path} already exists, please chcek!'

        # 在文件的开头写入训练的信息
        with open(self.txt_path, 'a') as f:
            msg = f'Model: {model_name}, Training on datasets: {self.ds_name_list}\n'
            f.write(msg)


    def __call__(self, epoch, training_info, val_info):

        if self.task == 'ped_cls':
            train_nonPed_acc = training_info.nonPed_acc_num / self.train_nonPed_num
            train_ped_acc = training_info.ped_acc_num / self.train_ped_num
            val_nonPed_acc = val_info.nonPed_acc_num / self.val_nonPed_num
            val_ped_acc = val_info.ped_acc_num / self.val_ped_num

            train_msg = f'Training Loss:{training_info.training_loss:.6f}, Balanced accuracy: {training_info.training_bc:.6f}, accuracy: {training_info.train_accuracy:.6f}, [0: {train_nonPed_acc:.4f}({training_info.nonPed_acc_num}/{self.train_nonPed_num}), 1: {train_ped_acc:.4f}({training_info.ped_acc_num}/{self.train_ped_num}), all: ({training_info.training_correct_num}/{self.train_num})]\n'

            val_msg = f'Val Loss:{val_info.val_loss:.6f}, Balanced accuracy: {val_info.val_bc:.6f}, accuracy: {val_info.val_accuracy:.6f}, [0: {val_nonPed_acc:.4f}({val_info.nonPed_acc_num}/{self.val_nonPed_num}), 1: {val_ped_acc:.4f}({val_info.ped_acc_num}/{self.val_ped_num}), all: ({val_info.val_correct_num}/{self.val_num})]\n'

            with open(self.txt_path, 'a') as f:
                f.write(f'Epoch: {epoch}\n')
                f.write(train_msg)
                f.write(val_msg)

        elif self.task == 'ds_cls':
            train_msg = f'Training Loss:{training_info.training_loss:.6f}, Balanced accuracy: {training_info.training_bc:.6f}, accuracy: {training_info.train_accuracy:.6f}\n'
            val_msg = f'Val Loss:{val_info.val_loss:.6f}, Balanced accuracy: {val_info.val_bc:.6f}, accuracy: {val_info.val_accuracy:.6f}\n'

            with open(self.txt_path, 'a') as f:
                f.write(f'Epoch: {epoch}\n')
                f.write(train_msg)
                f.write(val_msg)

        else:
            raise RuntimeError('Epoch_logger的task错误，请检查！')


# class BestClsModel():
#     '''
#         根据模型在某一个类别上的
#     '''
#     def __init__(self):
#         super().__init__()




















