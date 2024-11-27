import numpy as np
import torch, os
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from dataset import CIFARGroup_DSCls
from model import Net


def train():
    # 超参数设置
    batch_size = 16
    lr = 1e-5
    weight_decay = 1e-5
    group_dir = r'D:\my_phd\dataset\CIFAR10\groups'

    model = Net()
    cifar_dataset = CIFARGroup_DSCls(group_dir)
    cifar_loader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = np.Inf
    patience = 5
    waited = 0

    for epoch in range(50):
        print(f'Starting epoch {epoch}')

        epoch_loss = 0
        training_correct_num = 0
        for batch_idx, data in enumerate(tqdm(cifar_loader)):
            images, labels, _ = data
            out = model(images)
            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            # 计算accuracy
            _, pred = torch.max(out, 1)
            training_correct_num += (pred == labels).sum()

        training_accuracy = training_correct_num / len(cifar_dataset)
        print(f'Epoch {epoch} accuracy: {training_accuracy}')

        if epoch_loss < best_loss:
            waited += 1
            print(f'Best loss: {epoch_loss:.6f}')

        if waited > patience:
            print(f'loss保持{waited}个epoch没有降低，结束训练')

            save_model_path = r'./cifar10_group_cls.pth'
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     }
            torch.save(state, save_model_path)


if __name__ == '__main__':
    train()
























