"""
modified_dataset.py
===================
Inherited and modified pytorch datasets
"""
import os
import re
import sys
import numpy as np
from scipy.io import loadmat
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.utils import verify_str_arg
from utils import get_shuffled_idx


class Gray2RGB:
    """change grayscale PIL image to RGB format.
    channel values are copied"""
    def __call__(self, x):
        return x.convert('RGB')


class MNIST_OOD(MNIST):
    """
    See also the original MNIST class: 
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    """
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=False, seed=1):
        super(MNIST_OOD, self).__init__(root, transform=transform,
                                        target_transform=target_transform, download=download)
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training' or split == 'validation':
            self.train = True
            shuffle_idx = get_shuffled_idx(60000, seed)
        else:
            self.train = False
        self.split = split

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data, targets = self.data, self.targets

        if split == 'training':
            self.data = data[shuffle_idx][:54000]
            self.targets = targets[shuffle_idx][:54000]
        elif split == 'validation':
            self.data = data[shuffle_idx][54000:]
            self.targets = targets[shuffle_idx][54000:]
        elif split == 'evaluation':
            self.data = data
            self.targets = targets


class FashionMNIST_OOD(FashionMNIST):
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=False, seed=1):
        super(FashionMNIST_OOD, self).__init__(root, transform=transform,
                                        target_transform=target_transform, download=download)
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training' or split == 'validation':
            self.train = True
            shuffle_idx = get_shuffled_idx(60000, seed)
        else:
            self.train = False
        self.split = split

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data, targets = self.data, self.targets

        if split == 'training':
            self.data = data[shuffle_idx][:54000]
            self.targets = targets[shuffle_idx][:54000]
        elif split == 'validation':
            self.data = data[shuffle_idx][54000:]
            self.targets = targets[shuffle_idx][54000:]
        elif split == 'evaluation':
            self.data = data
            self.targets = targets


class CIFAR10_OOD(CIFAR10):
    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=True):

        super(CIFAR10_OOD, self).__init__(root, transform=transform,
                                          target_transform=target_transform, download=True)
        assert split in ('training', 'validation', 'evaluation', 'training_full')

        if split == 'training' or split == 'validation' or split == 'training_full':
            self.train = True
            shuffle_idx = np.load(os.path.join(root, 'cifar10_trainval_idx.npy'))
        else:
            self.train = False
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        if split == 'training':
            self.data = self.data[shuffle_idx][:45000]
            self.targets = self.targets[shuffle_idx][:45000]
        elif split == 'validation':
            self.data = self.data[shuffle_idx][45000:]
            self.targets = self.targets[shuffle_idx][45000:]
        elif split == 'training_full':
            pass

        self._load_meta()


class CIFAR100_OOD(CIFAR10_OOD):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class SVHN_OOD(SVHN):

    def __init__(self, root, split='training', transform=None, target_transform=None,
                 download=False):
        super(SVHN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training' or split == 'validation':
            svhn_split = 'train'
            shuffle_idx = np.load(os.path.join(root, 'svhn_trainval_idx.npy'))
        else:
            svhn_split = 'test'

        # self.split = verify_str_arg(svhn_split, "split", tuple(self.split_list.keys()))
        self.split = svhn_split  # special treatment
        self.url = self.split_list[svhn_split][0]
        self.filename = self.split_list[svhn_split][1]
        self.file_md5 = self.split_list[svhn_split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if split == 'training':
            self.data = self.data[shuffle_idx][:65930]
            self.labels = self.labels[shuffle_idx][:65930]
        elif split == 'validation':
            self.data = self.data[shuffle_idx][65930:]
            self.labels = self.labels[shuffle_idx][65930:]


class Constant_OOD(Dataset):
    def __init__(self, root, split='training', size=(32, 32), transform=None, channel=3):
        super(Constant_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.transform = transform
        self.root = root
        self.img_size = size
        self.channel = channel
        self.const_vals = np.load(os.path.join(root, 'const_img.npy'))  # (40,000, 3) array

        if split == 'training':
            self.const_vals = self.const_vals[:32000]
        elif split == 'validation':
            self.const_vals = self.const_vals[32000:36000]
        elif split == 'evaluation':
            self.const_vals = self.const_vals[36000:]

    def __getitem__(self, index):
        img = np.ones(self.img_size + (self.channel,), dtype=np.float32) * self.const_vals[index] / 255  # (H, W, C)
        img = img.astype(np.float32)
        if self.channel == 1:
            img = img[:, :, 0:1]

        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.const_vals)


class ConstantGray_OOD(Dataset):
    def __init__(self, root, split='training', size=(32, 32), transform=None, channel=3):
        super(ConstantGray_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.transform = transform
        self.root = root
        self.img_size = size
        self.channel = channel
        self.const_vals = np.load(os.path.join(root, 'const_img_gray.npy'))  # (40,000,) array

        if split == 'training':
            self.const_vals = self.const_vals[:32000]
        elif split == 'validation':
            self.const_vals = self.const_vals[32000:36000]
        elif split == 'evaluation':
            self.const_vals = self.const_vals[36000:]

    def __getitem__(self, index):
        img = np.ones(self.img_size + (self.channel,), dtype=np.float32) * self.const_vals[index] / 255  # (H, W, C)
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.const_vals)


class Noise_OOD(Dataset):
    def __init__(self, root, split='training', transform=None, channel=3, size=(32,32)):
        super(Noise_OOD, self).__init__()
        assert split in ('training', 'validation', 'evaluation')
        self.transform = transform
        self.root = root
        self.vals = np.load(os.path.join(root, 'noise_img.npy'))  # (40000, 32, 32, 3) array
        self.channel = channel
        self.size = size

        if split == 'training':
            self.vals = self.vals[:32000]
        elif split == 'validation':
            self.vals = self.vals[32000:36000]
        elif split == 'evaluation':
            self.vals = self.vals[36000:]

    def __getitem__(self, index):
        img = self.vals[index] / 255

        img = img.astype(np.float32)
        if self.channel == 1:
            img = img[:, :, 0:1]
        if self.size != (32, 32):
            img = img[:self.size[0], :self.size[1], :]
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.vals)


#
# CelebA
#
IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "list_attr_celeba.txt"


def _is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def _find_images_and_annotation(root_dir):
    images = {}
    attr = None
    assert os.path.exists(root_dir), "{} not exists".format(root_dir)
    img_dir = os.path.join(root_dir, 'Img/img_align_celeba')
    for fname in os.listdir(img_dir):
        if _is_image(fname):
            path = os.path.join(img_dir, fname)
            images[os.path.splitext(fname)[0]] = path

    attr = os.path.join(root_dir, 'Anno', ATTR_ANNO)
    assert attr is not None, "Failed to find `list_attr_celeba.txt`"

    # begin to parse all image
    final = []
    with open(attr, "r") as fin:
        image_total = 0
        attrs = []
        for i_line, line in enumerate(fin):
            line = line.strip()
            if i_line == 0:
                image_total = int(line)
            elif i_line == 1:
                attrs = line.split(" ")
            else:
                line = re.sub("[ ]+", " ", line)
                line = line.split(" ")
                fname = os.path.splitext(line[0])[0]
                onehot = [int(int(d) > 0) for d in line[1:]]
                assert len(onehot) == len(attrs), "{} only has {} attrs < {}".format(
                    fname, len(onehot), len(attrs))
                final.append({
                    "path": images[fname],
                    "attr": onehot
                })
    print("Find {} images, with {} attrs".format(len(final), len(attrs)))
    return final, attrs


class CelebA_OOD(Dataset):
    def __init__(self, root_dir, split='training', transform=None, seed=1):
        """attributes are not implemented"""
        super().__init__()
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training':
            setnum = 0
        elif split == 'validation':
            setnum = 1
        elif split == 'evaluation':
            setnum = 2
        else:
            raise ValueError(f'Unexpected split {split}')

        d_split = self.read_split_file(root_dir)
        self.data = d_split[setnum]
        self.transform = transform
        self.split = split
        self.root_dir = os.path.join(root_dir, 'CelebA', 'Img', 'img_align_celeba')


    def __getitem__(self, index):
        filename = self.data[index]
        path = os.path.join(self.root_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0. 

    def __len__(self):
        return len(self.data)

    def read_split_file(self, root_dir):
        split_path = os.path.join(root_dir, 'CelebA', 'Eval', 'list_eval_partition.txt')
        d_split = {0:[], 1:[], 2:[]}
        with open(split_path) as f:
            for line in f:
                fname, setnum = line.strip().split()
                d_split[int(setnum)].append(fname)
        return d_split


class NotMNIST(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        super().__init__()
        self.transform = transform
        shuffle_idx = np.load(os.path.join(root_dir, 'notmnist_trainval_idx.npy'))
        datadict = loadmat(os.path.join(root_dir, 'NotMNIST/notMNIST_small.mat'))
        data = datadict['images'].transpose(2, 0, 1).astype('float32')
        data = data[shuffle_idx]
        targets = datadict['labels'].astype('float32')
        targets = targets[shuffle_idx]
        if split == 'training':
            self.data = data[:14979]
            self.targets = targets[:14979]
        elif split == 'validation':
            self.data = data[14979:16851]
            self.targets = targets[14979:16851]
        elif split == 'evaluation':
            self.data = data[16851:]
            self.targets = targets[16851:]
        else:
            raise ValueError

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.data)


class ImageNet32(Dataset):
    def __init__(self, root_dir, split='training', transform=None, seed=1, train_split_ratio=0.8):
        """
        split: 'training' - the whole train split (1281149)
               'evaluation' - the whole val split (49999)
               'train_train' - (train_split_ratio) portion of train split
               'train_val' - (1 - train_split_ratio) portion of train split
        """
        super().__init__()
        self.root_dir = os.path.join(root_dir, 'ImageNet32')
        self.split = split
        self.transform = transform
        self.shuffle_idx = get_shuffled_idx(1281149, seed)
        n_train = int(len(self.shuffle_idx) * train_split_ratio)

        if split == 'training':  # whole train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
        elif split == 'evaluation':  # whole val split
            self.imgdir = os.path.join(self.root_dir, 'valid_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
        elif split == 'train_train':  # 80 % of train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
            self.l_img_file = [self.l_img_file[i] for i in self.shuffle_idx[:n_train]]
        elif split == 'train_val':  # 20 % of train split
            self.imgdir = os.path.join(self.root_dir, 'train_32x32')
            self.l_img_file = sorted(os.listdir(self.imgdir))
            self.l_img_file = [self.l_img_file[i] for i in self.shuffle_idx[n_train:]]
        else:
            raise ValueError(f'{split}')

    def __getitem__(self, index):
        imgpath = os.path.join(self.imgdir, self.l_img_file[index])
        im = Image.open(imgpath)
        if self.transform is not None:
            im = self.transform(im)
        return im, 0 

    def __len__(self):
        return len(self.l_img_file)
