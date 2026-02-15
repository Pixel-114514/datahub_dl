from models.cnn import SimpleCNN
from models.vae import ConvVAE



MODEL_REGISTRY = {
    "vae": ConvVAE,
    "cnn": SimpleCNN, 
}