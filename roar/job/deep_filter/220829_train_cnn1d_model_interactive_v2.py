
from damselfly.models import cnn1d
from damselfly.data import loaders, augmentation
from damselfly.utils import train
from pathlib import Path
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

checkpoint_path = Path.home()/'group'/'project'/'scripting'/'output'/\
'220830_test_cnn1d_training'
checkpoint_path.mkdir(exist_ok=True, parents=True)
checkpoint_name = 'run0.tar'

datapath = Path.home()/'group'/'project'/'datasets'/'data'/\
'220609_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'

conv_dict = {
    'channels':[2,15,20,20],
    'kernels':[3,3,3],
    'strides':[1,1,1],
    'pool':[5,5,5],
    'act': nn.LeakyReLU,
}
linear_dict = {
    'sizes': [cnn1d.output_size(conv_dict, 8192),512,64,2],
    'act': nn.LeakyReLU,
}
print(cnn1d.output_size(conv_dict, 8192))
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
    target_energy_range=(18575, 18580),
    target_pitch_range=(85.5, 88.0),
    target_radius_range=(0.005, 0.005),
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
    'batchsize': 2500,
    'epochs': 2000,
    'checkpoint_epochs': 25,
    'checkpoint': checkpoint_path/checkpoint_name,
    'initial_epoch': 0,
    'loss': [],
    'acc': [],
    'val_acc': [],
    'model_args': model_args,
    'opt_args': opt_args,
    'circular_shift':[-50,-40,-30,-20,-10,0,10,20,30,40]
}
noise_var = 1.38e-23*10*50*60*205e6/8192

print('Training starting')
train.TrainModel(0, model, optimizer, loss_fcn,\
 train_data, val_data, config, noise_gen=augmentation.AddNoise,\
 noise_gen_args=[noise_var], data_norm=augmentation.NormBatch
)
