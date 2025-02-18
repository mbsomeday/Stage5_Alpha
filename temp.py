'''
    在训练ped cls的时候加入ds model的CAM作为loss
'''

import torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.paths_dict import PATHS
from models.VGG import vgg16_bn
from data.dataset import my_dataset
from training.custom_losses import test_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ds_name = 'D3'
print(f'ds name: {ds_name}')
def reload_model(model, weights_path):
    checkpoints = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoints['model_state_dict'])
    return model

# ped_model = reload_model(vgg16_bn(num_class=2), PATHS['ped_cls_ckpt'][ds_name])
ped_model = vgg16_bn(2).to(DEVICE)
ds_model = reload_model(vgg16_bn(num_class=4), PATHS['ds_cls_ckpt']).to(DEVICE)

train_dataset = my_dataset([ds_name], txt_name='augmentation_train.txt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = my_dataset([ds_name], txt_name='val.txt')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

loss_fn = test_loss(ds_model=ds_model)
optimizer = torch.optim.SGD(ped_model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    print(f'Epoch: {epoch}')

    # train
    for idx, data_dict in enumerate(tqdm(train_loader)):
        images = data_dict['image'].to(DEVICE)
        ped_labels = data_dict['ped_label'].to(DEVICE)
        ds_labels = data_dict['ds_label'].to(DEVICE)

        out = ped_model(images)

        loss_val = loss_fn(preds=out, targets=ped_labels, images=images, ped_model=ped_model, ds_labels=ds_labels)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


    # val
    val_loss = 0
    val_correct_num = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(val_loader):
            images = data_dict['image'].to(DEVICE)
            ped_labels = data_dict['ped_label'].to(DEVICE)
            ds_labels = data_dict['ds_label'].to(DEVICE)

            out = ped_model(images)
            _, pred = torch.max(out, 1)

            val_correct_num += (pred == ped_labels).sum()

            # loss_val += loss_fn(preds=out, targets=ped_labels, images=images, ped_model=ped_model, ds_labels=ds_labels)
            # val_loss += loss_val.item()

        val_accuracy = val_correct_num / len(val_dataset)
        val_acc_100 = val_accuracy * 100
        print(f'Epoch {epoch} ped classification val accuracy: {val_acc_100}')





















