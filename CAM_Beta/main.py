import os, sys

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from CAM_Beta.vgg import vgg16_bn
from CAM_Beta.dataset import my_Dataset, dsCls_Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ds_dir = r'/kaggle/input/stage4-d1-ecpdaytime-7augs/Stage4_D1_ECPDaytime_7Augs'
    txt_path = 'test.txt'

    test_dataset = my_Dataset(ds_dir, txt_path, cls_label=0)
    test_loader = DataLoader(test_dataset, batch_size=1)

    dsCls_model = vgg16_bn(num_class=4)
    pedCls_model = vgg16_bn(num_class=2)

    dsCls_model.eval()
    pedCls_model.eval()

    pedCls_model = pedCls_model.to(DEVICE)
    dsCls_model = dsCls_model.to(DEVICE)

    all_msg = ''
    with torch.no_grad():
        for image, label, ds_label, image_name in tqdm(test_loader):

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            # 行人检测
            pedCls_out = pedCls_model(image)
            _, pedCls_pred = torch.max(pedCls_out, dim=1)

            pedCls_prob = torch.softmax(pedCls_out, dim=1)

            if pedCls_pred[0] == (label):
                pedCls_flag = True
            else:
                pedCls_flag = False

            # 数据集检测
            dsCls_out = dsCls_model(image)

            print(pedCls_out)
            print('dsCls_out:', dsCls_out)

            label_0 = np.array(pedCls_prob[0][0])
            label_1 = np.array(pedCls_prob[0][1])

            # 数据集检测
            dsCls_out = dsCls_model(image)
            _, dsCls_pred = torch.max(dsCls_out, dim=1)

            if dsCls_pred == ds_label:
                dsCls_flag = True
            else:
                dsCls_flag = False

            ds_pred = 'D' + str(np.array(dsCls_pred)[0])

            msg = str(image_name[0]) + ' ' + str('%.6f' % label_0) + ' ' + str('%.6f' % label_1) + ' ' + str(pedCls_flag) + ' ' + str(ds_pred) + ' ' + str(dsCls_flag) + '\n'

            all_msg += msg
            print(msg)


            break


    with open(r'/kaggle/working/M1onD1_res.txt', 'a') as f:
        for item in all_msg:
            f.write(item)





# TODO 1: 对 D1 的 test 进行 dataset classification
def dsCls_eval():
    ds_dir = r'D:\my_phd\dataset\Stage3\D1_ECPDaytime'
    txt_path = 'test.txt'

    train_dataset = dsCls_Dataset(ds_dir, txt_path, label=0)
    train_loader = DataLoader(train_dataset, batch_size=4)

    model = vgg16_bn(num_class=4)

    model.eval()

    # with torch.no_grad():
    #     for image, label in train_loader



# TODO 2: 对归类为D1的图片，用M1进行分类




# TODO 3：对归类为其他数据集的图片，进行CAM, (ped, ds)两个model的热力图，




# TODO 4：用热力图和原图进行操作，输入到M1中检测


if __name__ == '__main__':
    main()
























