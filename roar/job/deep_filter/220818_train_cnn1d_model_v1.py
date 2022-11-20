
from damselfly.models import cnn1d
from damselfly.data import loaders, augmentation
from damselfly.utils import train
from pathlib import Path
import torch.nn as nn
import torch

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', action='store', default=1, type=int)
parser.add_argument('--batchsize', dest='batchsize', action='store', default=500, type=int)
parser.add_argument('--lr', dest='lr', action='store', default=1e-3, type=float)


parser.add_argument('--pitch-min', dest='pitch_min', action='store', type=float)
parser.add_argument('--pitch-max', dest='pitch_max', action='store', type=float)
parser.add_argument('--energy-min', dest='energy_min', action='store', default=18575, type=int)
parser.add_argument('--energy-max', dest='energy_max', action='store', default=18580, type=int)
parser.add_argument('--radius-min', dest='radius_min', action='store', default=0.005, type=int)
parser.add_argument('--radius-max', dest='radius_max', action='store', default=0.005, type=int)

parser.add_argument('--data', dest='data', action='store',)
parser.add_argument('--name', dest='name', action='store',)

# model configuration
parser.add_argument('--kernels', nargs='+', dest='kernels', action='store', type=int)
parser.add_argument('--channels', nargs='+', dest='channels', action='store', type=int)
parser.add_argument('--strides', nargs='+', dest='strides', action='store', type=int)
parser.add_argument('--linear', nargs='+', dest='linear', action='store', type=int)

args = parser.parse_args()

checkpoint_path = f'{args.name}.tar'

datapath = Path(args.data)

conv_dict = {
    'channels':args.channels,
    'kernels':args.kernels,
    'strides':args.strides,
    'act': nn.LeakyReLU
}
linear_dict = {
    'sizes': [cnn1d.output_size(conv_dict, 8192), *args.linear],
    'act': nn.LeakyReLU,
}
model_args = (conv_dict, linear_dict)
model = cnn1d.Cnn1d(*model_args)

lr = 1e-3
opt_args = {
    'args':model.parameters(),
    'kwargs':{'lr':lr}
}

optimizer = torch.optim.Adam(opt_args['args'], **(opt_args['kwargs']))
loss_fcn = nn.CrossEntropyLoss()

print('Loading Data')
train_data, val_data = loaders.LoadH5ParamRange(
    path=datapath,
    target_energy_range=(args.energy_min, args.energy_max),
    target_pitch_range=(args.pitch_min, args.pitch_max),
    target_radius_range=(args.radius_min, args.radius_max),
    val_split=True,
    val_ratio=0.2,
    randomize_phase=True,
    copies=3,
)

train_data = augmentation.FFT(train_data)
val_data = augmentation.FFT(val_data)

train_data = torch.stack((train_data.real, train_data.imag), dim=1)
val_data = torch.stack((val_data.real, val_data.imag), dim=1)

train_data = torch.cat(
    (train_data, torch.zeros(train_data.shape, dtype=torch.float)), dim=0
)
val_data = torch.cat(
    (val_data, torch.zeros(val_data.shape, dtype=torch.float)), dim=0
)
train_labels = torch.zeros(train_data.shape[0], dtype=torch.long)
train_labels[0:train_data.shape[0]//2] = 1
val_labels = torch.zeros(val_data.shape[0], dtype=torch.long)
val_labels[0:val_data.shape[0]//2] = 1

train_data = (train_data, train_labels)
val_data = (val_data, val_labels)

config = {
    'batchsize': args.batchsize,
    'epochs': args.epochs,
    'checkpoint_epochs': 25,
    'checkpoint': checkpoint_path,
    'initial_epoch': 0,
    'loss': [],
    'acc': [],
    'val_acc': [],
    'model_args': model_args,
    'opt_args': opt_args,
}
noise_var = 1.38e-23 * 10 * 50 * 60 * 205e6/8192

print('Training starting')
train.TrainModel(0, model, optimizer, loss_fcn,\
 train_data, val_data, config, noise_gen=augmentation.AddNoise,\
 noise_gen_args=[noise_var], data_norm=augmentation.NormBatch
)
