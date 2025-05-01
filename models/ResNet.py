from torchvision import models


def ResNet34(num_class):
    model = models.resnet34(weights=None, )
    return model


