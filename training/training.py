import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from data.dataset import my_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class train_model():
    def __init__(self, model_name, model, ds_name_list, batch_size=4, epochs=10):
        self.model_name = model_name
        self.model = model
        self.epochs = epochs
        self.ds_name_list = ds_name_list

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        self.train_dataset = my_dataset(ds_name_list, 'augmentation_train.txt')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list, 'val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def train_one_epoch(self, epoch):

        self.model.train()

        training_loss = 0.0
        training_correct_num = 0
        # start_time = time()

        for batch, data in enumerate(tqdm(self.train_loader)):
            # 将image和label放到GPU中
            images = data['image']
            labels = data['label']

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

        # print(f'Training time for epoch:{epoch + 1}: {(time() - start_time):.2f}s')
        print('Training Loss:{:.6f}, Training accuracy:{:.6f}% ({} / {})'.format(training_loss, train_acc_100, training_correct_num,
                                                                       len(self.val_dataset)))

    def val_on_epoch_end(self):
        self.model.eval()
        val_loss = 0.0
        val_correct_num = 0

        with torch.no_grad():
            for data in (self.val_loader):
                images = data['image']
                labels = data['label']
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

        print('-' * 20 + 'training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print(f'From dataset: {self.ds_name_list}')
        print('Total Batch:', len(self.train_loader))
        print('Total EPOCH:', self.epochs)
        print('Runing device:', DEVICE)

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for epoch in range(self.epochs):
            print('=' * 30 + ' begin EPOCH ' + str(epoch + 1) + '=' * 30)
            self.train_one_epoch(epoch)
            self.val_on_epoch_end()

            # 这里放训练epoch的callbacks

            # break





















