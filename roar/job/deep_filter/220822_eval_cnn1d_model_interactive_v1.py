
from damselfly.models import cnn1d
from damselfly.data import loaders, augmentation
from damselfly.utils import train
from pathlib import Path
import torch.nn as nn
import torch

import numpy as np
import matplotlib.pyplot as plt

checkpoint_path = Path.home()/'group'/'project'/'scripting'/'output'/\
'220826_train_cnn1d'/'model12'/'model12.tar'

output_path = Path.home()/'group'/'project'/'scripting'/'output'/\
'220827_cnn1d_eval_results/'/'model12'
output_path.mkdir(exist_ok=True, parents=True)
output_path = output_path/'model12.npz'

datapath = Path.home()/'group'/'project'/'datasets'/'data'/\
'220619_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'

print('Loading Data')
data, _ = loaders.LoadH5ParamRange(
    path=datapath,
    target_energy_range=(18575, 18580),
    target_pitch_range=(85.5, 88.0),
    target_radius_range=(0.005, 0.005),
    val_split=False,
    val_ratio=0.2,
    randomize_phase=True,
    copies=3,
)

data = augmentation.FFT(data)
data = torch.stack((data.real, data.imag), dim=1)
data = torch.cat(
    (data, torch.zeros(data.shape, dtype=torch.float)), dim=0
)

labels = torch.zeros(data.shape[0], dtype=torch.long)
labels[0:data.shape[0]//2] = 1

data = [data, labels]

config = {
    'batchsize': 3000,
    'epochs': 100,
    'output': output_path,
}
noise_var = 1.38e-23*10*50*60*205e6/8192

print('Training starting')

train.EvalModel(0, cnn1d.Cnn1d, checkpoint_path, data, config,
noise_gen=augmentation.AddNoise,
noise_gen_args=[noise_var],
data_norm=augmentation.NormBatch,
)

output_path = Path.home()/'group'/'project'/'scripting'/'output'/\
'220827_cnn1d_eval_results'/'model12'
output_path.mkdir(exist_ok=True, parents=True)
output_path = output_path/'model12_shifted50.npz'

train.EvalModel(0, cnn1d.Cnn1d, checkpoint_path, data, config,
noise_gen=augmentation.AddNoise,
noise_gen_args=[noise_var],
data_norm=augmentation.NormBatch,
circular_shift=50
)
