import torch, random, os, math
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Noise adding functions
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=1):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img
#添加高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

# Dataset
class My_dataset(Dataset):
    def __init__(self, ds_base, txt_path):
        self.ds_base = ds_base
        self.txt_path = txt_path
        self.image_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImgLab()

    def init_ImgLab(self):
        images, labels = [], []

        with open(self.txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            item = item.strip().split()
            image_path = item[0].replace('\\', os.sep)

            images.append(os.path.join(self.ds_base, image_path))
            labels.append(item[1])

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
            label-0:原始图片
            label-1:椒盐噪声
            label-2:高斯噪声
        '''
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = np.array(self.labels[idx]).astype(np.int64)
        if self.labels[idx] == '0':
            image = image
        elif self.labels[idx] == '1':
            image = AddPepperNoise(0.99, 1.0)(image)
        else:
            image = AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.8, amplitude=random.uniform(0, 35))(
                image)
        image = self.image_transformer(image)
        return image, label

# Model
class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_class=2):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_class),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg16_bn(num_class=3):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class)


# train
def train():
    ds_base = r'/kaggle/input/stage4-d1-ecpdaytime-7augs/Stage4_D1_ECPDaytime_7Augs'

    train_txt_path = r'/kaggle/input/stage4-d1-ecpdaytime-7augs/group/train_group.txt'
    train_dataset = My_dataset(ds_base, train_txt_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_txt_path = r'/kaggle/input/stage4-d1-ecpdaytime-7augs/group/val_group.txt'
    val_dataset = My_dataset(ds_base, val_txt_path)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_txt_path = r'/kaggle/input/stage4-d1-ecpdaytime-7augs/group/test_group.txt'
    test_dataset = My_dataset(ds_base, test_txt_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = vgg16_bn()
    model = model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    msg = ''

    for epoch in range(20):
        print(f'start epoch {epoch + 1}')
        training_correct_num = 0
        train_epoch_loss = 0
        model.train()
        for img, label in tqdm(train_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            out = model(img)
            cur_loss = loss_fn(out, label)
            train_epoch_loss += cur_loss

            cur_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 用于计算accuracy
            _, pred = torch.max(out, 1)
            training_correct_num += (pred == label).sum()

            # break
        training_accuracy = training_correct_num / len(train_dataset)
        train_epoch_loss = train_epoch_loss / len(train_dataset)

        print('training accuracy :%.6f' % training_accuracy)
        print('training loss :%.6f' % train_epoch_loss)

        # 每个epoch后要 validate
        val_correct_num = 0
        val_epoch_loss = 0
        model.eval()
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                out = model(img)
                cur_loss = loss_fn(out, label)
                val_epoch_loss += cur_loss

                # 用于计算accuracy
                _, pred = torch.max(out, 1)
                val_correct_num += (pred == label).sum()

        val_accuracy = val_correct_num / len(val_dataset)
        val_epoch_loss = val_epoch_loss / len(val_dataset)

        print('val accuracy :%.6f' % val_accuracy)
        print('val loss :%.6f' % val_epoch_loss)

        train_info = str(f'{training_accuracy:.4f}') + ' ' + str(f'{train_epoch_loss:.4f}')
        val_info = str(f'{val_accuracy:.4f}') + ' ' + str(f'{val_epoch_loss:.4f}')
        msg += str(epoch + 1) + ' ' + train_info + ' ' + val_info + '\n'

    # 在整体训练完成后进行测试
    test_epoch_loss = 0
    test_correct_num = 0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            out = model(img)
            cur_loss = loss_fn(out, label)
            test_epoch_loss += cur_loss

            # 用于计算accuracy
            _, pred = torch.max(out, 1)
            test_correct_num += (pred == label).sum()

    test_accuracy = test_correct_num / len(test_dataset)
    test_epoch_loss = test_epoch_loss / len(test_dataset)

    print('test accuracy :%.6f' % test_accuracy)
    print('test loss :%.6f' % test_epoch_loss)


    with open(r'/kaggle/working/group_results.txt', 'a') as f:
        #     with open(r'/kaggle/working/groupNoise_results.txt', 'a') as f:
        for item in msg:
            f.write(item)

    return model

if __name__ == '__main__':
    BATCH_SIZE = 4


















