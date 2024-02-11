#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of the DEAP dataset using TorchEEG.

@author: Karahan Yilmazer
"""
# %% Import packages
# !%matplotlib qt
# !%load_ext autoreload
# !%autoreload 2

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LOCATION_DICT,
)
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
from torcheeg.trainers import ClassifierTrainer

# %%
dataset = DEAPDataset(
    io_path=f'./examples_pipeline/deap',
    root_path='./data_preprocessed_python',
    offline_transform=transforms.Compose(
        [
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True),
        ]
    ),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(), transforms.ToTensor()]
    ),
    label_transform=transforms.Compose(
        [
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]
    ),
    num_worker=8,
)

k_fold = KFoldGroupbyTrial(
    n_splits=10, split_path='./examples_pipeline/split', shuffle=True, random_state=42
)
# %%
for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    trainer = ClassifierTrainer(
        model=model, num_classes=2, lr=1e-4, weight_decay=1e-4, accelerator="gpu"
    )
    trainer.fit(
        train_loader,
        val_loader,
        max_epochs=50,
        default_root_dir=f'./examples_pipeline/model/{i}',
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_val_batches=0.0,
    )
    score = trainer.test(
        val_loader, enable_progress_bar=True, enable_model_summary=True
    )[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')

# %%
