from torchvision import models


def efficientNetB0(num_class):
    model = models.efficientnet_b0(weights=None, progress=True, num_classes=num_class)
    return model














