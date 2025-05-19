import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from data.dataset import my_dataset
from utils.utils import load_model, get_obj_from_str, DEVICE


def test_ped_classifier(model_obj, weights_path, ds_name_list, batch_size, txt_name):
    '''
        测试行人分类模型
    '''
    ped_model = get_obj_from_str(model_obj)(num_class=2)
    ped_model = load_model(ped_model, weights_path).to(DEVICE)
    ped_model.eval()

    ped_dataset = my_dataset(ds_name_list=ds_name_list, path_key='org_dataset', txt_name=txt_name)
    ped_loader = DataLoader(ped_dataset, batch_size=batch_size, shuffle=False)

    correct_num = 0
    y_pred = []
    y_label = []

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(ped_loader)):
            images = data_dict['image'].to(DEVICE)
            ped_label = data_dict['ped_label'].to(DEVICE)

            logits = ped_model(images)
            preds = torch.argmax(logits, 1)

            y_label.extend(ped_label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            correct_num += (ped_label == preds).sum()


        cm = confusion_matrix(y_label, y_pred)
        print(f'Testing cm:\n {cm}')

        bc = balanced_accuracy_score(y_label, y_pred)
        ds_accuracy = correct_num / len(ped_dataset)
        print(f'行人分类模型准确率为：{ds_accuracy}, bc:{bc}')


# if __name__ == '__main__':
#     model_obj = 'models.EfficientNet.efficientNetB0'
#     weights_path = r'C:\Users\wangj\Desktop\efficientB0\efficientB0_dsCls\efficientNetB0_dsCls-10-0.97636.pth'
#     # weights_path = r'/data/jcampos/jiawei_data/code/efficientNetB0_dsCls/efficientNetB0_dsCls-10-0.97636.pth'

    # tt = test_ds_classifier(model_obj=model_obj, weights_path=weights_path, batch_size=8)

























