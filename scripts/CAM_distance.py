import os.path
from tqdm import tqdm
import torch
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import load_model
from configs.paths_dict import PATHS
from data.dataset import get_data, dataset_from_list


# 1. 根据ds_name 获取 在eval状态的 ds model 和 ped model
def get_ds_ped_models(ds_name):
    ds_model = models.efficientnet_b0(weights=None, num_classes=4)
    ds_weights = PATHS['EfficientNet_ds_cls']
    ds_model = load_model(ds_model, ds_weights)
    ds_model.eval()

    ped_model = models.efficientnet_b0(weights=None, num_classes=2)
    ped_weights = PATHS['EfficientNet_ped_cls'][ds_name]
    ped_model = load_model(ped_model, ped_weights)
    ped_model.eval()

    return ds_model, ped_model


# 2.获取CAM
def get_cam(model, finalconv_name, image):
    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        # print(f'hook feature_blobs: {feature_blobs}')
        feature_blobs.append(output.cpu().data.numpy())

    def return_cam(feature_map, weight_softmax, visual_idx):
        '''
        :param feature_map: (batch_size, 1280, 7, 7)
        :param weight_softmax: (num_classes, 1280)
        :param visual_idx: (batch_size, )
        '''
        bc, nc, h, w = feature_map.shape
        batch_classifier_weights = weight_softmax[visual_idx]
        if len(batch_classifier_weights.shape) == 1:
            batch_classifier_weights = batch_classifier_weights[np.newaxis, :]
        reshaped_feature = feature_map.reshape((-1, nc, h * w))

        cam_array = None
        for i in range(bc):
            cam = batch_classifier_weights[i].dot(reshaped_feature[i])

            # 对cam进行处理
            if cam_array is None:
                cam_array = cam
            else:
                cam_array = np.vstack((cam_array, cam))

        if len(cam_array.shape) == 1:
            cam_array = cam_array[np.newaxis, :]

        # 对cam进行一系列处理
        cam_array = cam_array.reshape((-1, h, w))
        cam_array = cam_array - np.min(cam_array, axis=(1, 2), keepdims=True)  # cam = cam - np.min(cam)
        try:
            cam_array = cam_array / np.max(cam_array, axis=(1, 2), keepdims=True)
        except ZeroDivisionError:
            print('error')
        cam_visual_array = np.uint8(255 * cam_array)

        return cam_array, cam_visual_array

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(model.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())  # 最后一个conv层的weights, shape=(num_classes, 1280)

    logit = model(image)
    h_x = F.softmax(logit, dim=1)
    probs, idx = h_x.sort(1, descending=True)

    batch_cam, batch_cam_vis = return_cam(feature_blobs[0], weight_softmax, idx[:, 0])

    return batch_cam, batch_cam_vis


# 3. 计算CAM 距离
def cos_distance(cam1, cam2):
    '''
        将二维矩阵flatten后计算cos距离
        cos distance范围： [-1, 1]，数值越小说明越相似
    '''
    cam1 = cam1.flatten()
    cam2 = cam2.flatten()

    cos_sim = cam1.dot(cam2) / (np.linalg.norm(cam1) * np.linalg.norm(cam2))
    cos_dis = 1 - cos_sim
    return cos_dis

def batch_cam_distance(cam_batch1, cam_batch2):
    assert cam_batch1.shape == cam_batch2.shape
    bc, h, w = cam_batch1.shape

    distance_list = []
    for i in range(bc):
        cam1 = cam_batch1[i]
        cam2 = cam_batch2[i]

        cam_distance = cos_distance(cam1, cam2)
        distance_list.append(cam_distance)
    return distance_list



# 4. 合并上述功能
def main():
    # 通过arg 参数传递
    batch_size = 4
    finalconv_name = 'features'
    txt_base = r'D:\my_phd\on_git\Stage5_Alpha\scripts\EfficientNet'
    ds_name = 'D3'
    txt_name = 'test'
    selected_file = 'pedR_dsW.txt'

    txt_path = os.path.join(txt_base, str(ds_name+'_'+txt_name), selected_file)

    # model and data
    ds_model, ped_model = get_ds_ped_models(ds_name)

    # from torchsummary import summary
    # summary(ds_model, (3, 224, 224))

    get_dataset = dataset_from_list(txt_path)
    get_loader = DataLoader(get_dataset, batch_size=batch_size)

    # get_dataset, get_loader = get_data(ds_name_list=['D3'], path_key='org_dataset', txt_name='val.txt', batch_size=batch_size, shuffle=False)

    total_cam_dis_list = []
    for data_idx, image in enumerate(tqdm(get_loader)):
        batch_ds_cam, batch_ds_cam_vis = get_cam(model=ds_model, finalconv_name=finalconv_name, image=image)
        batc_ped_cam, batch_ped_cam_vis = get_cam(model=ped_model, finalconv_name=finalconv_name, image=image)

        batch_cam_dis_list = batch_cam_distance(batch_ds_cam, batc_ped_cam)
        total_cam_dis_list.extend(batch_cam_dis_list)

    print(f'共 {len(total_cam_dis_list)} 个样本')
    print(f'cos距离的均值：{sum(total_cam_dis_list) / len(total_cam_dis_list)}')

if __name__ == '__main__':
    main()



















