import os, random


dataset_base = r'D:\my_phd\dataset\Stage3\D1_ECPDaytime'
aug_txt = os.path.join(dataset_base, 'dataset_txt', 'train.txt')

with open(aug_txt, 'r') as f:
    data = f.readlines()

num_each = int(len(data) / 3)
print(f'Total num:{len(data)}, each file contains:{num_each}')

train_each_group = int(num_each * 0.8)
val_each_group = int(num_each * 0.1)
test_each_group = int(num_each * 0.1)

print(f'Train: {train_each_group} / {train_each_group * 3}')
print(f'Val: {val_each_group} / {val_each_group * 3}')
print(f'Test: {test_each_group} / {test_each_group * 3}')

# 随机打乱顺序
random.seed(13)
random.shuffle(data)

# group_msg = ''
# group_txt_path = os.path.join(dataset_base, 'dataset_txt', 'group', 'group.txt')

train_msg = ''
train_group_path = os.path.join(dataset_base, 'dataset_txt', 'group', 'train_group.txt')

val_msg = ''
val_group_path = os.path.join(dataset_base, 'dataset_txt', 'group', 'val_group.txt')

test_msg = ''
test_group_path = os.path.join(dataset_base, 'dataset_txt', 'group', 'test_group.txt')

for i in range(3):
    for idx, item in enumerate(data[i * num_each: (i + 1) * num_each]):
        item = item.strip().split()
        image_path = item[0]
        msg = image_path + ' ' + str(i) + '\n'
        # group_msg += image_path + ' ' + str(i) + '\n'
        if idx < train_each_group:
            train_msg += msg
        elif idx < train_each_group + val_each_group:
            val_msg += msg
        else:
            test_msg += msg

with open(train_group_path, 'a') as f:
    for item in train_msg:
        f.write(item)

with open(val_group_path, 'a') as f:
    for item in val_msg:
        f.write(item)

with open(test_group_path, 'a') as f:
    for item in test_msg:
        f.write(item)































