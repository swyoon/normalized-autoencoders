"""
Misc Utility functions
"""
import os
import sys
import logging
import datetime
import numpy as np
import torch 
import yaml
from torch.autograd import Variable
from collections import OrderedDict


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def convert_state_dict_remove_main(state_dict):
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    elif not next(iter(state_dict)).startswith("module.main"):
        return convert_state_dict(state_dict)  # abort if dict is not a DataParallel model_state

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[12:]  # remove `module.main`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path, mode='w')
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def check_config_validity(cfg):
    """checks validity of config"""
    assert 'model' in cfg
    assert 'training' in cfg
    assert 'validation' in cfg
    assert 'evaluation' in cfg
    assert 'data' in cfg

    data = cfg['data']
    assert 'dataset' in data
    assert 'path' in data
    assert 'n_classes' in data
    assert 'split' in data
    assert 'resize_factor' in data
    assert 'label' in data

    d = cfg['training']
    s = 'training'
    assert 'train_iters' in d, s
    assert 'val_interval' in d, s
    assert 'print_interval' in d, s
    assert 'optimizer' in d, s
    assert 'loss' in d, s
    assert 'batch_size' in d, s
    assert 'n_workers' in d, s

    d = cfg['validation']
    s = 'validation'
    assert 'batch_size' in d, s
    assert 'n_workers' in d, s

    d = cfg['evaluation']
    s = 'evaluation'
    assert 'batch_size' in d, s
    assert 'n_workers' in d, s
    assert 'num_crop_width' in d, s
    assert 'num_crop_height' in d, s


    print('config file validation passed')


def add_uniform_noise(x):
    """
    x: torch.Tensor
    """
    return x * 255. / 256. + torch.rand_like(x) / 256.


import errno
import os


# Recursive mkdir
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


from sklearn.metrics import roc_auc_score
def roc_btw_arr(arr1, arr2):
    true_label = np.concatenate([np.ones_like(arr1),
                                 np.zeros_like(arr2)])
    score = np.concatenate([arr1, arr2])
    return roc_auc_score(true_label, score)


def batch_run(m, dl, device, flatten=False, method='predict', input_type='first', no_grad=True, **kwargs):
    """
    m: model
    dl: dataloader
    device: device
    method: the name of a function to be called
    no_grad: use torch.no_grad if True.
    kwargs: additional argument for the method being called
    """
    method = getattr(m, method)
    l_result = []
    for batch in dl:
        if input_type == 'first':
            x = batch[0]

        if no_grad:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = method(x.cuda(device), **kwargs).detach().cpu()
        else:
            if flatten:
                x = x.view(len(x), -1)
            pred = method(x.cuda(device), **kwargs).detach().cpu()

        l_result.append(pred)
    return torch.cat(l_result)



def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, 'w') as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)


def get_shuffled_idx(N, seed):
    """get randomly permuted indices"""
    rng = np.random.default_rng(seed)
    return rng.permutation(N)


def search_params_intp(params):
    ret = {}
    for param in params.keys():
        # param : "train.batch"
        spl = param.split(".")
        if len(spl) == 2:
            if spl[0] in ret:
                ret[spl[0]][spl[1]] = params[param]
            else:
                ret[spl[0]] = {spl[1]: params[param]}
            # temp = {}
            # temp[spl[1]] = params[param]
            # ret[spl[0]] = temp
        elif len(spl) == 3:
            if spl[0] in ret:
                if spl[1] in ret[spl[0]]:
                    ret[spl[0]][spl[1]][spl[2]] = params[param]
                else:
                    ret[spl[0]][spl[1]] = {spl[2]: params[param]}
            else:
                ret[spl[0]] = {spl[1]: {spl[2]: params[param]}}
        elif len(spl) == 1:
            ret[spl[0]] = params[param]
        else:
            raise ValueError
    return ret


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def weight_norm(net):
    """computes L2 norm of weights of parameters"""
    norm = 0
    for param in net.parameters():
        norm += (param ** 2).sum()
    return norm



## additional command-line argument parsing
def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if val == 'True':
        return True
    try:
        return float(val)
    except ValueError:
        return val


def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs


def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg
