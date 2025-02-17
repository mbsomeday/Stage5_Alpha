'''
    测试自定义loss和model params
'''

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_layer = nn.Sequential(
            nn.Linear(4, 3),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        out = self.test_layer(x)
        return out

if __name__ == '__main__':
    torch.manual_seed(13)
    rand_input = torch.rand((1, 4))
    print('input:', rand_input)
    model = Net()
    out = model(rand_input)

    # for name, module in model.test_layer._modules.items():
    #     print(f'{name} - {module}')

    print('*' * 30 + ' Init ' + '*' * 30)
    for name, params in model.named_parameters():
        print(f'{name} - {params} - {params.shape}- {params.grad}')
    print('*' * 80)

    print('out:', out)

    from training.custom_losses import test_loss

    loss_fn = test_loss()

    label_tensor = torch.tensor([1])
    loss_val = loss_fn(label_tensor, out)

    loss_val.backward()

    print('=' * 30 + ' Before opt '+ '=' * 30 )
    for name, params in model.named_parameters():
        print(f'{name} - {params} - {params.shape} - grad:{params.grad}')
    print('=' * 80)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
    optimizer.step()

    print('loss_val:', loss_val)

    print('=' * 30 + ' After opt '+ '=' * 30 )
    for name, params in model.named_parameters():
        print(f'{name} - {params} - {params.shape} - grad:{params.grad}')
    print('=' * 80)

    out = model(rand_input)
    loss_val = loss_fn(label_tensor, out)

    loss_val.backward()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
    optimizer.step()

    print('=' * 30 + ' After opt '+ '=' * 30 )
    for name, params in model.named_parameters():
        print(f'{name} - {params} - {params.shape} - grad:{params.grad}')
    print('=' * 80)

















