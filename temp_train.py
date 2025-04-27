import torch
from torch import nn


# pool of square window of size=3, stride=2
m = nn.AvgPool2d(3, stride=2)
# # pool of non-square window
# m = nn.AvgPool2d((3, 2), stride=(2, 1))
input = torch.randn(1, 32, 7, 7)
output = m(input)
print(f'output: {output.shape}')















