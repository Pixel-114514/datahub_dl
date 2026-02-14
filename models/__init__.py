from models.cnn import SimpleCNN
from models.resnet import ResNet

MODEL_REGISTRY = {
    "resnet": ResNet,
    "cnn": SimpleCNN, 
}