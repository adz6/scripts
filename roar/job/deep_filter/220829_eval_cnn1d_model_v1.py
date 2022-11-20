
from damselfly.models import cnn1d
from damselfly.data import loaders, augmentation
from damselfly.utils import train
from pathlib import Path
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', action='store', default=1, type=int)
parser.add_argument('--batchsize', dest='batchsize', action='store', default=500, type=int)
parser.add_argument('--circular-shift', nargs='+', dest='circular_shift', default=[0], action='store', type=int)

parser.add_argument('--pitch-min', dest='pitch_min', action='store', type=float)
parser.add_argument('--pitch-max', dest='pitch_max', action='store', type=float)
parser.add_argument('--energy-min', dest='energy_min', action='store', default=18575, type=int)
parser.add_argument('--energy-max', dest='energy_max', action='store', default=18580, type=int)
parser.add_argument('--radius-min', dest='radius_min', action='store', default=0.005, type=int)
parser.add_argument('--radius-max', dest='radius_max', action='store', default=0.005, type=int)

parser.add_argument('--data', dest='data', action='store',)
parser.add_argument('--checkpoint', dest='checkpoint', action='store')
parser.add_argument('--name', dest='name', action='store',)

args = parser.parse_args()

# model configuration

checkpoint_path = Path(args.checkpoint)
output_path = f'{args.name}'
datapath = Path(args.data)

print('Loading Data')
data, _, train_inds, _ = loaders.LoadH5ParamRange(
    path=datapath,
    target_energy_range=(args.energy_min, args.energy_max),
    target_pitch_range=(args.pitch_min, args.pitch_max),
    target_radius_range=(args.radius_min, args.radius_max),
    val_split=False,
    val_ratio=0.2,
    randomize_phase=True,
    copies=3,
)

np.save('train_inds.npy', train_inds)

data = augmentation.FFT(data)
data = torch.stack((data.real, data.imag), dim=1)
data = torch.cat(
    (data, torch.zeros(data.shape, dtype=torch.float)), dim=0
)

labels = torch.zeros(data.shape[0], dtype=torch.long)
labels[0:data.shape[0]//2] = 1

data = [data, labels]

config = {
    'batchsize': args.batchsize,
    'epochs': args.epochs,
    'output': output_path,
    'circular_shift':args.circular_shift
}
noise_var = 1.38e-23*10*50*60*205e6/8192

print('Training starting')

train.EvalModel(0, cnn1d.Cnn1d, checkpoint_path, data, config,
noise_gen=augmentation.AddNoise,
noise_gen_args=[noise_var],
data_norm=augmentation.NormBatch,
circular_shift=args.circular_shift
)


