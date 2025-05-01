from torchvision import models


def ResNet34(num_class):
    model = models.resnet34(weights=None, num_classes=num_class)
    return model
