import torch, os, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.VGG import vgg16_bn
from data.dataset import my_dataset
from configs.paths_dict import PATHS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ds_name', type=str)

    args = parser.parse_args()
    return args


def ped_test(model, test_dataset, test_loader):
    model.eval()

    correct_num = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict['image'].to(DEVICE)
            ped_labels = data_dict['ped_label'].to(DEVICE)

            ped_out = model(images)
            ped_pred = torch.max(ped_out, dim=1)

            correct_num += (ped_pred == ped_labels).sum()

        test_accuracy = round(correct_num / len(test_dataset), 6)
        print(test_accuracy)



if __name__ == '__main__':

    args = get_args()

    test_dataset = my_dataset(ds_name_list=[ds_name], txt_name='test.txt', key_name='dataset_dict')
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = vgg16_bn(num_class=2)
    ckpt = torch.load(PATHS['ped_cls_ckpt'][ds_name], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    ped_test(model, test_dataset=test_dataset, test_loader=test_loader)



























