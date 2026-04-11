from trainer.base import BaseTrainer
from trainer.vae import VAETrainer
from trainer.diffusion import DiffusionTrainer
from trainer.sr import SuperResolutionTrainer
from trainer.sr3 import SR3Trainer
from trainer.resshift import ResShiftTrainer

TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "vae": VAETrainer,
    "ddpm": DiffusionTrainer,
    "sr": SuperResolutionTrainer,
    "sr3": SR3Trainer,
    "resshift": ResShiftTrainer,
}
