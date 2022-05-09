from enum import Enum

import typer

from generative_models_pytorch.gan import GAN, GANTrainingOption
from generative_models_pytorch.vae import VAE, VAETrainingOption
from generative_models_pytorch.wgan import WassersteinGAN, WassersteinGANTrainingOption

app = typer.Typer(help="Train generative models and generate images.")


class GenerativeModel(str, Enum):
    GAN = "gan"
    WGAN = "wgan"
    VAE = "vae"


def create_model(model: GenerativeModel, device: str):
    if model == GenerativeModel.GAN:
        return GAN(), GANTrainingOption(
            device=device, g_path="generator.pth", d_path="discriminator.pth"
        )
    if model == GenerativeModel.WGAN:
        return WassersteinGAN(), WassersteinGANTrainingOption(
            device=device, g_path="wgenerator.pth", d_path="wdiscriminator.pth"
        )
    return VAE(), VAETrainingOption(device=device, path="vae.pth")


@app.command()
def train(model: GenerativeModel, device="cpu"):
    """
    Train a generative model.
    """
    gmodel, option = create_model(model, device)
    gmodel.train(option)


@app.command()
def generate(device="cpu"):
    """
    Generate images from a trained model.
    """
    pass


if __name__ == "__main__":
    app()
