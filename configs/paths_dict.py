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

    'autoencoder_ckpt_dict': {
        'D1': r'/kaggle/input/stage5-weights-ldm-d1/D1_epo26_00894.ckpt',
        'D2': r'/kaggle/input/stage5-weights-ldm-d2/D2_epo59_01239.ckpt',
        'D3': r'/kaggle/input/stage5-weights-ldm-d3/D3_epo49_01236.ckpt',
        'D4': r'/kaggle/input/stage5-weights-ldm-d4/D4_epo34_01236.ckpt'
    }
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













