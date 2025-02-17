import torch
from torch import nn
from torch import autograd
import numpy as np

class test_loss(nn.Module):
    def __init__(self, ds_model):
        super().__init__()
        self.ds_model = ds_model
        self.ped_loss_fn = torch.nn.CrossEntropyLoss()
        self.visual_layer_name = '40'
        self.ds_model.eval()

    def get_CAM(self, image, model):
        features = torch.unsqueeze(image, dim=0)
        features_flatten = None
        visual_flag = False
        for index, (name, module) in enumerate(model.features._modules.items()):
            if name != self.visual_layer_name and not visual_flag:
                features = module(features)
            elif name == self.visual_layer_name:
                features = module(features)
                visual_flag = True
            else:
                features_flatten = module(features if features_flatten is None else features_flatten)

        features_flatten = torch.flatten(features_flatten, 1)

        out = model.classifier(features_flatten)

        pred = torch.argmax(out, dim=1).item()
        pred_class = out[:, pred]

        features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

        grads = features_grad
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        pooled_grads = pooled_grads[0]
        features = features[0]

        for i in range(features.shape[0]):
            features[i, ...] *= pooled_grads[i, ...]
        heatmap = features.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)

        heatmap /= np.max(heatmap)
        return heatmap


    def forward(self, preds, targets, images=None, ped_model=None, ds_labels=None):
        CAM_loss = 0

        for image in images:

            # 获取ped的CAM
            ped_CAM = self.get_CAM(image=image, model=ped_model)

            # 获取ds的CAM
            ds_CAM = self.get_CAM(image=image, model=self.ds_model)

            # 获取 CAM_diff
            CAM_diff = torch.tensor(abs(ped_CAM-ds_CAM).sum())

            # 取log
            cur_CAM_loss = torch.log(CAM_diff)

            CAM_loss += cur_CAM_loss


        CAM_loss = 0.1 * CAM_loss.item()

        ped_loss_val = self.ped_loss_fn(preds, targets)

        total_loss = CAM_loss + ped_loss_val

        return total_loss


if __name__ == '__main__':

    import torch, os
    from torch.utils.data import DataLoader

    from configs.paths_dict import PATHS
    from models.VGG import vgg16_bn
    from data.dataset import my_dataset

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_name = 'D3'
    print(f'ds name: {ds_name}')


    def reload_model(model, weights_path):
        checkpoints = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoints['model_state_dict'])
        return model


    # ped_model = reload_model(vgg16_bn(num_class=2), PATHS['ped_cls_ckpt'][ds_name])
    ped_model = vgg16_bn(2)
    ds_model = reload_model(vgg16_bn(num_class=4), PATHS['ds_cls_ckpt'])

    train_dataset = my_dataset([ds_name], txt_name='augmentation_train.txt')
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)

    loss_fn = test_loss(ds_model=ds_model)

    for idx, data_dict in enumerate(train_loader):
        images = data_dict['image'].to(DEVICE)
        ped_labels = data_dict['ped_label'].to(DEVICE)
        ds_labels = data_dict['ds_label'].to(DEVICE)

        print('ped_labels:', ped_labels)
        # print('ds_labels:', ds_labels)

        out = ped_model(images)

        loss_val = loss_fn(preds=out, targets=ped_labels, images=images, ped_model=ped_model, ds_labels=ds_labels)

        break


















