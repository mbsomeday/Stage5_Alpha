import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from data.dataset import my_dataset
from utils.utils import load_model, get_obj_from_str, DEVICE


def test_ds_classifier(model_obj, weights_path, batch_size):
    '''
        测试数据集分类模型
    :param model_obj:
    :param weights_path:
    :param batch_size:
    :return:
    '''
    ds_model = get_obj_from_str(model_obj)(num_class=4)
    ds_model = load_model(ds_model, weights_path).to(DEVICE)
    ds_model.eval()

    ds_dataset = my_dataset(ds_name_list=['D1', 'D2', 'D3', 'D4'], path_key='org_dataset', txt_name='test.txt')
    ds_loader = DataLoader(ds_dataset, batch_size=batch_size, shuffle=False)

    correct_num = 0
    y_pred = []
    y_label = []

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(ds_loader)):
            images = data_dict['image'].to(DEVICE)
            ds_label = data_dict['ds_label'].to(DEVICE)

            logits = ds_model(images)
            probs = torch.argmax(logits, 1)

            y_label.extend(ds_label.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

            correct_num += (ds_label == probs).sum()

        cm = confusion_matrix(y_label, y_pred)
        print(f'Testing cm:\n {cm}')

        ds_accuracy = correct_num / len(ds_dataset)
        print(f'准确率为：{ds_accuracy}')





# if __name__ == '__main__':
#     model_obj = 'models.EfficientNet.efficientNetB0'
#     # weights_path = r'C:\Users\wangj\Desktop\efficientB0\efficientB0_dsCls\efficientNetB0_dsCls-10-0.97636.pth'
#     weights_path = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'
#
#     tt = test_ds_classifier(model_obj=model_obj, weights_path=weights_path, batch_size=96)
#















