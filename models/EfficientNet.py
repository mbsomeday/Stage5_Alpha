from torchvision import models


def get_efficient_ds_model():
    model = models.efficientnet_b0(weights=None, progress=True, num_classes=4)
    return model


def get_efficient_ped_model():
    model = models.efficientnet_b0(weights=None, progress=True, num_classes=2)
    return model












