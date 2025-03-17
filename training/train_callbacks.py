import os, torch
import numpy as np


class EarlyStopping():
    '''
        保存当前为止最好的模型(loss最低)，
        当loss稳定不变patience个epoch时，结束训练
    '''

    def __init__(self, save_prefix, top_k=3,
                 patience=10, delta=0.000001,
                 model_save_dir=None):
        '''
            这个 early stopping关注的是 accuracy
            model save name: prefix_{epoch}_{acc}.pth
        '''

        self.top_k = top_k
        self.save_prefix = save_prefix

        if model_save_dir is not None:
            self.model_save_dir = model_save_dir
        else:
            self.model_save_dir = os.path.join(os.getcwd(), 'ckpt')

        if not os.path.exists(self.model_save_dir):
                os.mkdir(self.model_save_dir)

        self.patience = patience
        self.counter = 0  # 记录loss不变的epoch数目
        self.early_stop = False # 是否停止训练
        self.best_val_acc = -np.Inf
        self.delta = delta

        print('-' * 20 + 'Early Stopping Info' + '-' * 20)
        print('Create early stopping, monitoring [validation accuracy] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {save_prefix}_epoch_acc.pth')
        print(f'Early Stop with patience: {self.patience} ')

    def __call__(self, epoch, model, val_acc, optimizer):
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

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }

        save_path = os.path.join(self.model_save_dir, save_name)

        # 存储权重
        torch.save(checkpoint, save_path)
        self.best_val_acc = val_acc



class ImageLogger():
    def __init__(self):
        pass
















