from torchvision.models import vgg16, VGG16_Weights

# Best available weights (currently alias for IMAGENET1K_V2)
# Note that these weights may change across versions
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
print(model)




























