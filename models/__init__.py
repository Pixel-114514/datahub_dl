from models.cnn import SimpleCNN
from models.resnet import ResNet
from models.sr import SimpleSRResNet
from models.resshift import ResShiftUNet
from models.vae import ConvVAE
from models.ddpm.unet import UNetModel


MODEL_REGISTRY = {
    "vae": ConvVAE,
    "cnn": SimpleCNN,
    "resnet": ResNet,
    "ddpm": UNetModel,
    "sr_resnet": SimpleSRResNet,
    "resshift": ResShiftUNet,
}
