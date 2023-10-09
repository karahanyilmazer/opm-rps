#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of OPM data using EEGNET.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2
import os
import sys

sys.path.insert(0, r'C:\Files\Coding\Python\Neuro\eeg_classes')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.analysis.EEGNET import EEGNET
from torch import Tensor, nn, optim, rand
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# %%
# Initialize the WandB logger
wandb_logger = WandbLogger(project="MNIST")

# Initialize the autoencoder
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
autoencoder = EEGNET(encoder, decoder)

# Setup data
transform = ToTensor()
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
train_loader = DataLoader(train_set)
test_loader = DataLoader(test_set)

# Train the model
trainer = pl.Trainer(limit_train_batches=100, max_epochs=3, logger=wandb_logger)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)

# %%
# Load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = EEGNET.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# Choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# Embed 4 fake images!
fake_image_batch = rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

# %%
# train on 4 GPUs
trainer = pl.Trainer(
    devices=4,
    accelerator="gpu",
)

# train 1TB+ parameter models with Deepspeed/fsdp
trainer = pl.Trainer(
    devices=4, accelerator="gpu", strategy="deepspeed_stage_2", precision=16
)

# 20+ helpful flags for rapid idea iteration
trainer = pl.Trainer(max_epochs=10, min_epochs=5, overfit_batches=1)

# access the latest state of the art techniques
trainer = pl.Trainer(callbacks=[StochasticWeightAveraging(...)])
