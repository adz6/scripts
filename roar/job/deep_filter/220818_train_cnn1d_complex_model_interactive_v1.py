
from damselfly.models import cnn1d_complex
from damselfly.data import loaders, augmentation
from damselfly.utils import train
from pathlib import Path
import torch.nn as nn
import torch

import numpy as np
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', dest='epoch', action='store', default=1, type=int)
#parser.add_argument('--batchsize', dest='batchsize', action='store', default=500, type=int)
#parser.add_argument('--lr', dest='lr', action='store', default=1e-3, type=float)

#parser.add_argument('--pitch-min', dest='pitch_min', action='store', type=float)
#parser.add_argument('--pitch-max', dest='pitch_max', action='store', type=float)
#parser.add_argument('--energy-min', dest='energy_min', action='store', default=18575, type=int)
#parser.add_argument('--energy-max', dest='energy_max', action='store', default=18580, type=int)
#parser.add_argument('--radius-min', dest='radius_min', action='store', default=0.005, type=int)
#parser.add_argument('--radius-max', dest='radius_max', action='store', default=0.005, type=int)

#parser.add_argument('--train-data', dest='train_data', action='store',)
#parser.add_argument('--test-data', dest='test_data', action='store',)
#parser.add_argument('--name', dest='name', action='store',)

# model configuration
#parser.add_argument('--kernels', nargs='+', dest='kernels', action='store', type=int)
#parser.add_argument('--channels', nargs='+', dest='channels', action='store', type=int)
#parser.add_argument('--strides', nargs='+', dest='strides', action='store', type=int)
#parser.add_argument('--linear', nargs='+', dest='linear', action='store', type=int)


#args = parser.parse_args()


checkpoint_path = Path.home()/'group'/'project'/'scripting'/'output'/\
'220825_test_cnn1d_complex_training'
checkpoint_path.mkdir(exist_ok=True, parents=True)
checkpoint_name = 'run0.tar'

datapath = Path.home()/'group'/'project'/'datasets'/'data'/\
'220609_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'

conv_dict = {
    'channels':[1,10,15,25,35],
    'kernels':[4,4,2,2],
    'strides':[4,4,2,2],
    'act': cnn1d_complex.ComplexLeakyRelu
}
linear_dict = {
    'sizes': [cnn1d_complex.output_size(conv_dict, 8192),1024,256,2],
    'act': cnn1d_complex.ComplexLeakyRelu,
}
model_args = (conv_dict, linear_dict)
model = cnn1d_complex.ComplexCNN(*model_args)

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
    target_pitch_range=(86.7, 88.0),
    target_radius_range=(0.005, 0.005),
    val_split=True,
    val_ratio=0.2,
    randomize_phase=True,
    copies=2,
)

train_data = train_data.unsqueeze(dim=1)
val_data = val_data.unsqueeze(dim=1)

train_data = augmentation.FFT(train_data)
val_data = augmentation.FFT(val_data)

norm_train_data = augmentation.NormBatchComplex(train_data)

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.plot(abs(norm_train_data[0, 0, :].numpy()))
#plt.savefig('test0')

train_data = torch.cat(
    (train_data, torch.zeros(train_data.shape, dtype=torch.cfloat)), dim=0
)
val_data = torch.cat(
    (val_data, torch.zeros(val_data.shape, dtype=torch.cfloat)), dim=0
)
train_labels = torch.zeros(train_data.shape[0], dtype=torch.long)
train_labels[0:train_data.shape[0]//2] = 1
val_labels = torch.zeros(val_data.shape[0], dtype=torch.long)
val_labels[0:val_data.shape[0]//2] = 1

train_data = (train_data, train_labels)
val_data = (val_data, val_labels)

config = {
    'batchsize': 2500,
    'epochs': 250,
    'checkpoint_epochs': 25,
    'checkpoint': checkpoint_path/checkpoint_name,
    'initial_epoch': 0,
    'loss': [],
    'acc': [],
    'val_acc': [],
    'model_args': model_args,
    'opt_args': opt_args,
}
noise_var = 1.38e-23*10*50*60*205e6/8192

print('Training starting')
train.TrainModel(0, model, optimizer, loss_fcn,\
 train_data, val_data, config, noise_gen=augmentation.AddNoiseComplex,\
 noise_gen_args=[noise_var], data_norm=augmentation.NormBatchComplex
)
