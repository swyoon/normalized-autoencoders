"""
generate samples
"""
import os
import numpy as np
import torch
from models import get_model
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('result_dir', type=str, help='path to the directory where yaml file and checkpoint is stored.')
parser.add_argument('config', type=str, help='the name of configs file.')
parser.add_argument('ckpt', type=str, help='the name of the checkpoint file.')
parser.add_argument('--device', default=0, type=int, help='the id of cuda device to use')
parser.add_argument('--n_sample', default=1000, type=int, help='the number of samples to be generated')
parser.add_argument('--zstep', default=None, type=int, help='the number of the latent Langevin MC steps')
parser.add_argument('--xstep', default=None, type=int, help='the number of the visible Langevin MC steps')
parser.add_argument('--x_shape', default=32, type=int, help='the size of a side of an image.')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--name', default=None, type=str, help='additional identifier for the result file.')
parser.add_argument('--replay', default=False, action='store_true', help='to use the sample replay buffer')
args = parser.parse_args()


config_path = os.path.join(args.result_dir, args.config)
ckpt_path = os.path.join(args.result_dir, args.ckpt)
device = f'cuda:{args.device}'

# load model
print(f'building a model from {config_path}')
cfg = OmegaConf.load(config_path)
model = get_model(cfg)
model_class = cfg['model']['arch']

print(f'loading a model from {ckpt_path}')
state = torch.load(ckpt_path)
if 'model_state' in state:
    state = state['model_state']
model.load_state_dict(state)
model.to(device)
model.eval()

if model_class == 'nae':
    print(f'replay {args.replay}')
    model.replay = args.replay

    dummy_x = torch.rand(1, 3, args.x_shape, args.x_shape, dtype=torch.float).to(device)
    model._set_x_shape(dummy_x)
    model._set_z_shape(dummy_x)

    if args.zstep is not None:
        model.z_step = args.zstep
        print(f'z_step: {model.z_step}')
    if args.xstep is not None:
        model.x_step = args.xstep
        print(f'x_step: {model.x_step}')
elif model_class == 'vae':
    dummy_x = torch.rand(1, 3, args.x_shape, args.x_shape, dtype=torch.float).to(device)
    model._set_z_shape(dummy_x)


# run sampling
batch_size = args.batch_size
n_batch = int(np.ceil(args.n_sample / batch_size))
l_sample = []
for i_batch in tqdm(range(n_batch)):
    if i_batch == n_batch - 1:
        n_sample = args.n_sample % batch_size if args.n_sample % batch_size else batch_size 
    else:
        n_sample = batch_size
    d_sample = model.sample(n_sample=n_sample, device=device)
    sample = d_sample['sample_x'].detach()

    # re-quantization
    # sample = (sample * 255 + 0.5).clamp(0, 255).cpu().to(torch.uint8)
    sample = (sample * 256).clamp(0, 255).cpu().to(torch.uint8)

    sample = sample.permute(0, 2, 3, 1).numpy()


    l_sample.append(sample)
sample = np.concatenate(l_sample)
print(f'sample shape: {sample.shape}')

# save result
if args.name is not None:
    out_name = os.path.join(args.result_dir, f'{args.ckpt.strip(".pkl")}_sample_{args.name}.npy')
else:
    out_name = os.path.join(args.result_dir, f'{args.ckpt.strip(".pkl")}_sample.npy')
np.save(out_name, sample)
print(f'sample saved at {out_name}')


