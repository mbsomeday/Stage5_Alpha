import os, torch, math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10

class my_dataset(Dataset):
    '''
        This is a test class for getting D2.
    '''
    def __init__(self, ds_dir=r'/veracruz/home/j/jwang/data/D2_CityPersons', txt_name='test.txt'):
    # def __init__(self, ds_dir=r'D:\my_phd\dataset\D2_CityPersons', txt_name='test.txt'):

        super().__init__()
        self.ds_dir = ds_dir
        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.get_imgLab()

    def get_imgLab(self):
        images, labels = [], []

        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            item = item.replace('\\', os.sep)
            item = item.strip().split()
            image_path = os.path.join(self.ds_dir, item[0])
            label = item[-1]

            images.append(image_path)
            labels.append(label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.img_transforms(image)
        label = np.array(self.labels[idx]).astype(np.int64)
        return image, label

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_class=2):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(512*7*7, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, num_class),

            nn.Linear(512*7*7, 512),
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


def vgg16_bn(num_class=2):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class)


print('Now start training')
model = vgg16_bn()
model = model.to(DEVICE)

test_dataset = my_dataset()
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS+1):
    print(f'Training epoch loop {epoch}')
    epoch_loss = 0
    training_correct_num = 0

    for batch_idx, data in enumerate(test_loader):
        images, labels = data
#         images = images.to(DEVICE)
#         labels = labels.to(DEVICE)
#
#         out = model(images)
#         loss = loss_fn(out, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss
#
#         # 计算accuracy
#         _, pred = torch.max(out, 1)
#         training_correct_num += (pred == labels).sum()
#
#     training_accuracy = training_correct_num / len(test_loader)
#     print(f'Epoch {epoch} accuracy: {training_accuracy}, training_correct_num:{training_correct_num}')
#
#     # save model when finished
#     if epoch == EPOCHS:
#         print(f'Save model on {epoch}')
#
#         save_model_path = r'./test_vgg16bn.pth'
#         state = {'model': model.state_dict()}
#         torch.save(state, save_model_path)






















