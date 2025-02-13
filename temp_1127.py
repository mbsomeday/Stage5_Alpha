import os

with open(r'D:\chrom_download\M3onD3_res.txt', 'r') as f:
    data = f.readlines()

pedCls_wrong_num = 0
dsCls_wrong_num = 0
both_wrong = 0

# for idx, item in enumerate(data):
#     item = item.strip().split()
#
#     pedCls_res = item[-3]
#     dsCls_res = item[-1]
#
#     if pedCls_res == 'False':
#         pedCls_wrong_num += 1
#
#     if pedCls_res == 'False' and dsCls_res == 'False':
#         both_wrong += 1
#     #     print(item)
#
#     if dsCls_res == 'False':
#         print(item)
#         dsCls_wrong_num += 1
#
# print('pedCls_wrong_num:', pedCls_wrong_num)
# print('dsCls_wrong_num:', dsCls_wrong_num)
# print('both_wrong:', both_wrong)

total_num = 0
right_of_total = 0

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for idx, item in enumerate(data):
    item = item.strip().split()

    image_cls = item[0].split('/')[1]

    if image_cls == 'nonPedestrian':
        image_label = 0
    else:
        image_label = 1

    pred_0 = float(item[1])
    pred_1 = float(item[2])

    if pred_0 > pred_1:
        prd_label = 0
    else:
        prd_label = 1
    #
    # print('image_label:', image_label)
    # print('prd_label:', prd_label)

    dsCls_flag = item[-1]

    if dsCls_flag == 'True':
        total_num += 1

        pedCls_flag = item[-3]
        if pedCls_flag == 'True':
            right_of_total += 1
            if image_label == 0:
                true_neg += 1
            else:
                true_pos += 1
        else:
            if image_label == 0 and prd_label == 1:
                false_pos += 1
            elif image_label == 1 and prd_label == 0:
                false_neg += 1


acc = right_of_total / total_num
print('total_num:', total_num)
print('right_of_total:', right_of_total)
print('acc:', acc)

print('true_neg:', true_neg)
print('true_pos:', true_pos)
print('false_pos:', false_pos)
print('false_neg:', false_neg)

# =====================================================

# import torch
# from PIL import Image
# from torchvision import transforms
#
# from CAM_Beta.vgg import vgg16_bn
#
# DEVICE = 'cpu'
# def reload_model(model, weights_path):
#     checkpoints = torch.load(weights_path, map_location=DEVICE)
#     model.load_state_dict(checkpoints['model_state_dict'])
#
#     return model
#
# M1_weights = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D1-014-0.9740.pth'
# M3_weights = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D3-025-0.9303.pth'
# M4_weights = r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D4-013-0.9502.pth'
#
# D1_path = r'D:\my_phd\dataset\Stage3\D1_ECPDaytime'
#
# model = vgg16_bn(num_class=2)
# model = reload_model(model, M3_weights)
#
# image_path = os.path.join(D1_path, 'nonPedestrian', 'stuttgart_00190_1.jpg')
# image = Image.open(image_path)
# image_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# image = image_transform(image)
#
# image = torch.unsqueeze(image, dim=0)
#
# out = model(image)
# probs = torch.softmax(out, dim=1)
# print(probs)


















