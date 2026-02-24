from trainer.base import BaseTrainer
from trainer.vae import VAETrainer
from trainer.diffusion import DiffusionTrainer

TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "vae": VAETrainer,
    "ddpm": DiffusionTrainer,
}