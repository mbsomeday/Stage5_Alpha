'''
    efficient net
'''

from efficientnet_pytorch import EfficientNet
import torch
from torchsummary import summary


rand_input = torch.rand((1, 3, 224, 224))
model = EfficientNet.from_name('efficientnet-b4', num_classes=2)

# out = model(rand_input)
# print(model)

summary(model, (3, 224, 224))




















