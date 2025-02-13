import os
from PIL import Image


ds_base_dir = r'D:\my_phd\dataset\Stage3\D3_ECPNight'
# ped_dir = os.path.join(ds_base_dir, 'pedestrian')
nonPed_dir = os.path.join(ds_base_dir, 'nonPedestrian')

# ped_images = os.listdir(ped_dir)
nonPed_images = os.listdir(nonPed_dir)

# ped_dir
for idx, cur_image in enumerate(nonPed_images):
    if idx < 50:
        image_path = os.path.join(nonPed_dir, cur_image)
        image = Image.open(image_path)
        image = image.resize((64, 64))

        new_name = 'nonPed_' + str(idx+1) + '.jpg'
        new_path = os.path.join(r'D:\my_phd\dataset\Stage3\D3_ECPNight\temp_64', new_name)

        image.save(new_path)



























