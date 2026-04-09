from trainer.base import BaseTrainer
from trainer.vae import VAETrainer
from trainer.diffusion import DiffusionTrainer
from trainer.sr import SuperResolutionTrainer
from trainer.resshift import ResShiftTrainer

TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "vae": VAETrainer,
    "ddpm": DiffusionTrainer,
    "sr": SuperResolutionTrainer,
    "resshift": ResShiftTrainer,
}
