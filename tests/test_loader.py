import numpy as np
from loaders import get_dataset, get_dataloader
from torch.utils import data
import yaml
import pytest
from skimage import io
import pickle
import torch
import os


def test_get_dataloader():
    cfg = {'dataset': 'FashionMNISTpad_OOD',
           'path': 'datasets',
           'shuffle': True,
           'n_workers': 0,
           'batch_size': 1,
           'split': 'training'}
    dl = get_dataloader(cfg)


def test_concat_dataset():
    data_cfg = {'concat1':
                    {'dataset': 'FashionMNISTpad_OOD',
                     'path': 'datasets',
                     'shuffle': True,
                     'split': 'training'},
                'concat2':
                    {'dataset': 'MNISTpad_OOD',
                     'path': 'datasets',
                     'shuffle': True,
                     'n_workers': 0,
                     'batch_size': 1,
                     'split': 'training'},
                 'n_workers': 0,
                 'batch_size': 1,
                }
    get_dataset(data_cfg)

