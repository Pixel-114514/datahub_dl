from models.cnn import SimpleCNN
from models.vae import ConvVAE
from models.ddpm.unet import UNetModel


MODEL_REGISTRY = {
    
    "vae": ConvVAE,
    "cnn": SimpleCNN, 
    "ddpm": UNetModel

}