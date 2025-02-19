import os

LCA = {
    'dataset_dict': {
        'D1': r'/tormenta/s/ssesaai/data/Stage4_D1_ECPDaytime_7Augs',
        'D2': r'/tormenta/s/ssesaai/data/Stage4_D2_CityPersons_7Augs',
        'D3': r'/tormenta/s/ssesaai/data/Stage4_D3_ECPNight_7Augs',
        'D4': r'/tormenta/s/ssesaai/data/Stage4_D4_BDD100K_7Augs'
    },

    'autoencoder_ckpt_dict': {
        'D1': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D1_epo26_00894.ckpt',
        'D2': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D2_epo59_01239.ckpt',
        'D3': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D3_epo49_01236.ckpt',
        'D4': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D4_epo34_01236.ckpt'
    }
}

LOCAL = {
    'dataset_dict': {
        'D1': r'D:\my_phd\dataset\Stage3\D1_ECPDaytime',
        'D2': r'D:\my_phd\dataset\Stage3\D2_CityPersons',
        'D3': r'D:\my_phd\dataset\Stage3\D3_ECPNight',
        'D4': r'D:\my_phd\dataset\Stage3\D4_BDD100K',
    },

    'ped_cls_ckpt': {
        'D1': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D1-014-0.9740.pth',
        'D2': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D2-025-0.9124.pth',
        'D3': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D3-025-0.9303.pth',
        'D4': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D4-013-0.9502.pth',
    },

    'ds_cls_ckpt': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth',

    'autoencoder_ckpt_dict': {
        'D1': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D1_epo26_00894.ckpt',
        'D2': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D2_epo59_01239.ckpt',
        'D3': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D3_epo49_01236.ckpt',
        'D4': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D4_epo34_01236.ckpt',
    }
}

KAGGLE = {
    'dataset_dict': {
        'D1': r'/kaggle/input/stage4-d1-ecpdaytime-7augs/Stage4_D1_ECPDaytime_7Augs',
        'D2': r'/kaggle/input/stage4-d2-citypersons-7augs/Stage4_D2_CityPersons_7Augs',
        'D3': r'/kaggle/input/stage4-d3-ecpnight-7augs',
        'D4': r'/kaggle/input/stage4-d4-7augs'
    },

    'AEOrg_dataset': {
        'D1': r'/kaggle/input/stage5-dataset-orgrecons/D1_Recon/D1',
        'D2': r'/kaggle/input/stage5-dataset-orgrecons/D2_Recon/D2',
        'D3': r'/kaggle/input/stage5-dataset-orgrecons/D3_Recon/D3',
        'D4': r'/kaggle/input/stage5-dataset-orgrecons/D4_Recon/D4',
    },

    'AE1_dataset': {
        'D1': r'/kaggle/input/stage5-dataset-orgrecons/D1_Recon/D1',
        'D2': r'/kaggle/input/stage5-dataset-ae1recons/AE1D2_test/D2',
        'D3': r'/kaggle/input/stage5-dataset-ae1recons/AE1D3_test/D3',
        'D4': r'/kaggle/input/stage5-dataset-ae1recons/AE1D4_test/D4',
    },

    'ds_cls_ckpt': r'/kaggle/input/stage4-dscls-weights/vgg16bn-dsCls-029-0.9777.pth',

    'ped_cls_ckpt': {
        'D1': r'/kaggle/input/stage3-baselineweights/vgg-D1_ECPDaytime-020-0.97007483.pth',
        'D2': r'/kaggle/input/stage3-baselineweights/vgg-D2_CityPersons-019-0.95496535.pth',
        'D3': r'/kaggle/input/stage3-baselineweights/vgg-D3_ECPNight-028-0.94842410.pth',
        'D4': r'/kaggle/input/stage3-baselineweights/vgg-D4_BDD100K-020-0.91304350.pth'
    },

    'autoencoder_ckpt_dict': {
        'D1': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D1-014-0.9740.pth',
        'D2': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D2-025-0.9124.pth',
        'D3': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D3-025-0.9303.pth',
        'D4': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D4-013-0.9502.pth'
    },

}


cwd = os.getcwd()

print('-' * 50)

if 'my_phd' in cwd:
    print(f'Run on Local -- working dir: {cwd}')
    PATHS = LOCAL
elif 'kaggle' in cwd:
    print(f'Run on kaggle -- working dir: {cwd}')
    PATHS = KAGGLE
elif 'veracruz' in cwd:
    print(f'Run on lca -- working dir: {cwd}')
    PATHS = LCA
else:
    raise Exception('运行平台未知，需配置路径!')

print()













