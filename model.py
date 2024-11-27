# https://blog.csdn.net/weixin_44026604/article/details/113799154

from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    # init定义网络中的结构
    def __init__(self):
        super(Net, self).__init__()
        # 3输入，16输出，卷积核(7, 7)，膨胀系数为2
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, dilation=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # dropout
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
        self.fc1 = nn.Linear(288, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 4)

    # forward定义数据在网络中的流向
    def forward(self, x, a):
        print('a:', a)
        # 卷积之后做一个最大池化，然后RELU激活
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 整形
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}






if __name__ == '__main__':
    from torchsummary import summary

    model = Net()

    summary(model, [(3, 224, 224), (1, 1, 1)], batch_size=3)

    # print(model)
    # msg = get_parameter_number(model)











