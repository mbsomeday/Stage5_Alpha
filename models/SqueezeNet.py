from torchvision import models


def squeezeNet(num_class):
    model = models.squeezenet1_0(weights=None, progress=True, num_classes=num_class)
    return model