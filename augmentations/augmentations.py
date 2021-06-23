import math
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torchvision.transforms import (
        RandomApply,
        ColorJitter,
        )

from PIL import Image, ImageOps


class RandomRotate90:
    def __init__(self):
        self.t = RandomChoice([RandomRotation(0), RandomRotation(90),
                               RandomRotation(180), RandomRotation(270)])
    def __call__(self, img):
        return self.t(img)


class ColorJitterSimCLR:
    def __init__(self, jitter_d=0.5, jitter_p=0.8):
        self.t = RandomApply([ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)],
                             p=jitter_p)

    def __call__(self, img):
        return self.t(img)


class Invert(object):
    """Transform that inverts pixel intensities"""
    def __init__(self):
        pass

    def __call__(self, img):
        return 1 - img


class ToGray:
    """ Transform color image to gray scale"""
    def __init__(self):
        pass

    def __call__(self, img):
        img = torch.sqrt(torch.sum(img ** 2, dim=0, keepdim=True))
        return torch.cat([img, img, img], dim=0).detach()


class Fragment:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sample):
        '''sample: numpy array for image (after ToTensor)'''
        if self.mode == 'horizontal':
            half = (sample.shape[2]) // 2
            if np.random.rand() > 0.5:  # uppder half
                sample[:,:,:half] = 0
            else:
                sample[:,:,half:] = 0

        elif self.mode == 'vertical':
            half = (sample.shape[1]) // 2
            if np.random.rand() > 0.5:  # uppder half
                sample[:,:half,:] = 0
            else:
                sample[:,half:,:] = 0
        elif self.mode == '1/4':
            choice = (np.random.rand() * 4) % 4
            half1 = (sample.shape[1]) // 2
            half2 = (sample.shape[2]) // 2
            img = torch.zeros_like(sample)
            if choice == 0:
                img[:,:half1,:half2] = sample[:,:half1,:half2]
            elif choice == 1:
                img[:,half1:,:half2] = sample[:,half1:,:half2]
            elif choice == 2:
                img[:,half1:,half2:] = sample[:,half1:,half2:]
            else:
                img[:,:half1,half2:] = sample[:,:half1,half2:]
            sample = img
        else:
            raise ValueError
        return sample


class UniformDequantize:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img / 256. * 255. + torch.rand_like(img) / 256.
        return img


class GaussianDequantize:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img + torch.randn_like(img) * 0.01
        return img

