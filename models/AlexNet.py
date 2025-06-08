from torchvision import models


def AlexNet(num_class):
    model = models.AlexNet(num_classes=num_class)
    return model