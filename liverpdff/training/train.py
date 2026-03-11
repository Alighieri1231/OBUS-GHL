"""
train.py

LightningCLI entrypoint for liver PDFF experiments.
"""
import torch

try:
    from lightning.pytorch.cli import LightningCLI
except ImportError:
    from pytorch_lightning.cli import LightningCLI

torch.set_float32_matmul_precision("medium")

cli = LightningCLI(save_config_callback=None)
