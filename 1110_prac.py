# https://www.cnblogs.com/miners/p/15101283.html
# https://github.com/datawhalechina/thorough-pytorch/blob/main/source/%E7%AC%AC%E5%8D%81%E7%AB%A0/Transformer%20%E8%A7%A3%E8%AF%BB.md
# https://www.cnblogs.com/wevolf/p/12484972.html

import torch
from torch import nn


class selfattention:
    def __init__(self, X):
        self.X = X  # 词向量

    def attention(self, d):
        n = self.X.size()[1]
        WQ = torch.nn.Linear(n, d)
        WK = torch.nn.Linear(n, d)
        WV = torch.nn.Linear(n, d)
        Q = WQ(self.X)
        K = WK(self.X)
        V = WV(self.X)
        att = torch.matmul(torch.softmax(torch.matmul(Q, K.T) / Q.size()[1], dim=1), V)
        return att


if __name__ == '__main__':
    print('Start')
    # 输入, 10个词，dim=500
    att = selfattention(torch.rand(10, 500))
    score = att.attention(9)
    print(score.shape)


























