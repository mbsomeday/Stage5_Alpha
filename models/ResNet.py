from torchvision import models


def ResNet34(num_class):
    model = models.resnet34(weights=None, num_class=num_class)
    return model


