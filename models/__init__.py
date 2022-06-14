import os
from omegaconf import OmegaConf
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from augmentations import get_composed_augmentations

from models.ae import (
    AE,
    VAE,
    DAE,
    WAE,
)
from models.nae import NAE, NAE_L2_OMI
from models.mcmc import get_sampler
from models.modules import (
    DeConvNet2,
    FCNet,
    ConvNet2FC,
    ConvMLP,
    IGEBMEncoder,
    ConvNet64,
    DeConvNet64,
)
from models.modules_sngan import Generator as SNGANGeneratorBN
from models.modules_sngan import GeneratorNoBN as SNGANGeneratorNoBN
from models.modules_sngan import GeneratorNoBN64 as SNGANGeneratorNoBN64
from models.modules_sngan import GeneratorGN as SNGANGeneratorGN
from models.energybased import EnergyBasedModel



def get_net(in_dim, out_dim, **kwargs):
    nh = kwargs.get("nh", 8)
    out_activation = kwargs.get("out_activation", "linear")

    if kwargs["arch"] == "conv2fc":
        nh_mlp = kwargs["nh_mlp"]
        net = ConvNet2FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
        )

    elif kwargs["arch"] == "deconv2":
        net = DeConvNet2(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "conv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = ConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif kwargs["arch"] == "deconv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = DeConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif kwargs["arch"] == "fc":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        net = FCNet(
            in_dim=in_dim,
            out_dim=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "convmlp":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        net = ConvMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "IGEBMEncoder":
        use_spectral_norm = kwargs.get("use_spectral_norm", False)
        keepdim = kwargs.get("keepdim", True)
        net = IGEBMEncoder(
            in_chan=in_dim,
            out_chan=out_dim,
            n_class=None,
            use_spectral_norm=use_spectral_norm,
            keepdim=keepdim,
        )
    elif kwargs["arch"] == "sngan_generator_bn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_nobn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_nobn64":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN64(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_gn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        num_groups = kwargs["num_groups"]
        net = SNGANGeneratorGN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
            num_groups=num_groups,
        )

    return net


def get_ae(**model_cfg):
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop("x_dim")
    z_dim = model_cfg.pop("z_dim")
    enc_cfg = model_cfg.pop('encoder')
    dec_cfg = model_cfg.pop('decoder')

    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = AE(encoder, decoder)
    elif arch == "dae":
        sig = model_cfg["sig"]
        noise_type = model_cfg["noise_type"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = DAE(encoder, decoder, sig=sig, noise_type=noise_type)
    elif arch == "wae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = WAE(encoder, decoder, **model_cfg)
    elif arch == "vae":
        sigma_trainable = model_cfg.get("sigma_trainable", False)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = VAE(encoder, decoder, **model_cfg)
    return ae



def get_vae(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    encoder_out_dim = z_dim * 2

    encoder = get_net(in_dim=x_dim, out_dim=encoder_out_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    n_sample = model_cfg.get("n_sample", 1)
    pred_method = model_cfg.get("pred_method", "recon")

    if model_cfg["arch"] == "vae":
        ae = VAE(encoder, decoder, n_sample=n_sample, pred_method=pred_method)
    return ae


def get_ebm(**model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    in_dim = model_cfg["x_dim"]
    model_cfg.pop("x_dim")
    net = get_net(in_dim=in_dim, out_dim=1, **model_cfg["net"])
    model_cfg.pop("net")
    return EnergyBasedModel(net, **model_cfg)


def get_nae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])

    if arch == "nae":
        ae = NAE(encoder, decoder, **model_cfg["nae"])
    else:
        raise ValueError(f"{arch}")
    return ae


def get_nae_v2(**model_cfg):
    arch = model_cfg.pop('arch')
    sampling = model_cfg.pop('sampling')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    if arch == 'nae_l2' and sampling == 'omi':
        sampler_z = get_sampler(**model_cfg['sampler_z'])
        sampler_x = get_sampler(**model_cfg['sampler_x'])
        nae = NAE_L2_OMI(encoder, decoder, sampler_z, sampler_x, **model_cfg['nae'])
    else:
        raise ValueError(f'Invalid sampling: {sampling}')
    return nae


def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "nae": get_nae,
            "nae_l2": get_nae_v2,
        }[name]
    except:
        raise ("Model {} not available".format(name))


def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    model_name = cfg['model']['arch']

    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    model.eval()
    return model, cfg

