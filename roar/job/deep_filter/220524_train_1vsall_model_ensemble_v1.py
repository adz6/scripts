import torch
from damselfly import models
import os
import math
from pathlib import Path
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('index')
args = parser.parse_args()

data_index = int(args.index)

def SaveCheckpoint(config, train_config, model_config, model_state, optimizer_state, epoch, loss):

    save_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'model_config': model_config,
        'epochs': epoch,
        'loss': loss, 
        #'optimizer_class': train_config['optimizer_class'],
        #'optimizer_config':{
        #    'lr': train_config['lr'],
        #    'params': optimizer_state['param_groups']
        #}
    }
    #config['checkpoint'].mkdir(parents=True, exist_ok=True)

    torch.save(save_dict, config['checkpoint'])

def LoadCheckpoint(config):

    checkpoint = torch.load(
        config['checkpoint'], 
        map_location=torch.device('cpu')
        )

    model = InitModel(checkpoint['model_config'])

    model.load_state_dict(
        checkpoint['model_state_dict'],
        )

    optimizer = checkpoint['optimizer_state_dict']

    epoch = checkpoint['epochs']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

def InitModel(config):

    if config['model_class'] == models.DFCNN:
        model = config['model_class'](
            config['config']['nclass'],
            config['config']['nch'],
            config['config']['conv'], 
            config['config']['lin']
        )

    elif config['model_class'] == models.MLP:
        model = config['model_class'](
            config['nclass'],
            config['lin'],
        )
    elif config['model_class'] == models.DilationBankCNN or \
            config['model_class'] == models.DilationBankCNN_v2 or \
            config['model_class'] == models.SparseCNN:
        model = config['model_class'](
            config['config']
        )
    elif config['model_class'] == models.DFCNN_dropout:
        model = config['model_class'](
            config['config']['nclass'],
            config['config']['nch'],
            config['config']['activation'],
            config['config']['conv'], 
            config['config']['lin']
        )
    
    return model

def Cleanup():
    torch.distributed.destroy_process_group()

def LoadSignal(config, samples=8192):

    if config['data_path'].name.endswith('.npy'):
        signal = np.load(config['data_path'])
    elif config['data_path'].name.endswith('.npz'):
        signal = np.load(config['data_path'])['x']
    elif config['data_path'].name.endswith('h5'):

        file = h5py.File(config['data_path'], 'r')

        signal_energy = file['energy'][:]
        signal_pitch = file['pitch'][:]
        signal_radius = file['radius'][:]

        energy_grid, pitch_grid, radius_grid = np.meshgrid(
            config['data_energy'],
            config['data_pitch'],
            config['data_radius']
             )

        data_inds = []
        for i, pair in enumerate(zip(energy_grid.flatten(), pitch_grid.flatten(), radius_grid.flatten())):

            #print(pair)
            try:
                index = np.argwhere(
                    np.logical_and(abs(pair[0] - signal_energy) < 1e-5, 
                    np.logical_and(abs(pair[1] - signal_pitch) < 1e-5, abs(pair[2] - signal_radius) < 1e-5))
                ).squeeze()
                if index.size == 1:
                    data_inds.append(index)
            except:
                pass

        data_inds = np.sort(np.array(data_inds, dtype=np.int32))
        print(data_inds.size)
        #print(data_inds)
        
        signal = file['x'][data_inds, 0:samples]
        #print(signal.shape)

        ###

        energy_grid, pitch_grid, radius_grid = np.meshgrid(
            config['target_energy'],
            config['target_pitch'],
            config['target_radius']
             )

        target_inds = []
        for i, pair in enumerate(zip(energy_grid.flatten(), pitch_grid.flatten(), radius_grid.flatten())):

            #print(pair)
            try:
                index = np.argwhere(
                    np.logical_and(abs(pair[0] - signal_energy) < 1e-5, 
                    np.logical_and(abs(pair[1] - signal_pitch) < 1e-5, abs(pair[2] - signal_radius) < 1e-5))
                ).squeeze()
                if index.size == 1:
                    target_inds.append(index)
            except:
                pass

        target_inds = np.sort(np.array(target_inds, dtype=np.int32))
        #print(target_inds)

        new_target_inds = []
        for i, ind in enumerate(target_inds):
            new_target_inds.append(np.argwhere(data_inds==ind).squeeze())

        target_inds = np.array(new_target_inds)
        print(target_inds.size)
        #print(target_inds)

        file.close()

    where_not_zero = np.argwhere(abs(signal).sum(-1)>0)
    return torch.tensor(signal[where_not_zero.squeeze(), :],), target_inds

def FFT(signal):

    #print(signal.shape)
    return torch.fft.fftshift(torch.fft.fft(signal, dim=-1, norm='forward'))

def GenerateDataTensor(signal, target_inds, config):

    #l_inds = config['inds'].size
    #inds = config['inds']
    #print(signal.shape[0] * config['number_signal_copies'])

    n_target = len(target_inds)
    n_total = signal.shape[0]

    n_copies_target = n_total // n_target

    n_all = n_total + n_target * (n_copies_target-1+8) # add 4 extra copies of the
    # target which we'll balance out with an equal amount of noise

    shape = (
        n_all ,
        config['number_channel'],
        signal.shape[1],
        )

    print(shape)

    target_signal = signal[target_inds, :]

    data = torch.zeros(shape, dtype=torch.float)

    data[0:signal.shape[0], 0, :] = signal.real.float()
    data[0:signal.shape[0], 1, :] = signal.imag.float()

    data[
        signal.shape[0]:signal.shape[0]+(n_copies_target+3)*n_target,
        0, :] = target_signal.tile((n_copies_target+3, 1)).real.float()
    data[
        signal.shape[0]:signal.shape[0]+(n_copies_target+3)*n_target,
         1, :] = target_signal.tile((n_copies_target+3, 1)).imag.float()

    return data

def GenerateTargetTensor(signal, target_inds, config):

    n_target = len(target_inds)
    n_total = signal.shape[0]

    n_copies_target = n_total // n_target

    n_all = n_total + n_target * (n_copies_target-1+8) # add 4 extra copies of the
    # target which we'll balance out with an equal amount of noise

    labels = torch.zeros(
        n_all,
        dtype=torch.long
        )

    labels[target_inds] = 1
    labels[signal.shape[0]:signal.shape[0]+(n_copies_target+3)*n_target] = 1

    return labels

def AddNoise(data, config):

    shape = data.shape
    
    noise = torch.normal(
        mean = 0,
        std = math.sqrt(config['var'] / 2),
        size = shape
        )

    return data + noise

def NormBatch(batch):
    
    batch = batch / torch.max(torch.max(abs(batch), dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]

    return batch

def BatchAcc(output, labels):

    torch_max = torch.max(output.cpu(), dim=-1)

    accuracy = (torch_max[1] == labels.cpu()).sum() / len(labels.cpu())

    return accuracy

def TrainModel(rank, config, train_config, model_config):

    torch.cuda.set_device(rank)
    print(rank)

    model = train_config['model'].cuda(rank)

    optimizer = train_config['optimizer_class'](
            model.parameters(),
            lr=train_config['lr']
        )

    if config['checkpoint'].exists():

        optimizer.load_state_dict(train_config['optimizer_state'])

    with torch.no_grad():
        data, target_inds = LoadSignal(config, samples = train_config['samples'])
        data = FFT(data)
        data_tensor = GenerateDataTensor(data, target_inds, train_config)
        target = GenerateTargetTensor(data, target_inds, train_config)

    dataset = torch.utils.data.TensorDataset(data_tensor, target)

    loader = torch.utils.data.DataLoader(
        dataset, 
        train_config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    objective = train_config['objective'].cuda(rank)
    ep_count = 0

    for ep in range(train_config['epochs']):

        batch_count = 0
        ep_acc = []
        ep_loss = []
        for batch, labels in loader:
            
            batch = AddNoise(batch, train_config)

            batch = NormBatch(batch)

            optimizer.zero_grad()
            output = model(batch.cuda(rank))
            loss = objective(output, labels.cuda(rank))
            loss.backward()

            optimizer.step()

            ep_acc.append(BatchAcc(output, labels))
            ep_loss.append(loss.item())

            train_config['loss'].append([ep_count, batch_count, loss.item()])
            batch_count += 1
        print(f'|  {ep + 1}  |  loss = {round(float(np.mean(ep_loss)), 5)}  |  acc = {round(float(np.mean(ep_acc)), 5)}  ')
        ep_count += 1

        if ep_count % 10 == 9:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            SaveCheckpoint(
                config,
                train_config,
                model_config,
                model_state_dict,
                optimizer_state_dict,
                train_config['initial_epoch'] + ep_count,
                train_config['loss'],
                )

if __name__ == "__main__":

    input_size = 8192

    conv_output_size = input_size // (5 ** 2)
    conv_layer_config = [
                            [
                                [2 * 1,60,60],
                                [60,60,60],
                                [5,5,5],
                                [1,1,1], # dilation
                                [.0, .0, .0], # Prob. dropout
                                5, # maxpool 
                            ],
                            [
                                [60,120,120],
                                [120,120,120],
                                [5,5,5],
                                [1,1,1], # dilation
                                [.0, .0, .0], # Prob. dropout
                                5, # maxpool 
                            ],

                        ]
                    
    linear_layer_config = [
            [conv_output_size * 120, 1024, 256], # input dense layer sizes
            [1024, 256, 128], # output dense layer sizes
            [0.0, 0.0, 0.0] # dropout
            ]

    model_config = {
    'model_class': models.DFCNN_dropout,
    'config': {
            'conv': conv_layer_config,
            'nclass': 2,
            'nch': 2,
            'lin': linear_layer_config,
            'activation': torch.nn.ELU()
        },
    }

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    print(f'Found {world_size} GPU(S)!')

    world_pitch_angles = np.linspace(85.0, 88.0, 61)

    config = {
        ############ DATA #################
        'data_path': Path.home()/'group'/'project'/'results'/'beamforming'/\
        'time_dependent'/'beamform_signals'\
        /'220505_dl_grid_84to90deg_1to45mm.h5',

        
        ########### CHECKPOINT ################
        'checkpoint': Path.home()/'group'/'project'/'results'/\
        'machine_learning'/'dnn'/'triggering'/'checkpoints'/'models'
        /f'220525_{world_pitch_angles[data_index]}to{world_pitch_angles[data_index+2]}deg_1to3mm_model1_1vsall_ensemble.tar',

        'rank':0,

        'data_energy':np.linspace(18545, 18555, 11),
        'data_pitch':world_pitch_angles,
        'data_radius':np.array([0.001, 0.002, 0.003]),

        # data index can run from 0 to the number of world pitch angles minus 3

        'target_energy': np.linspace(18545, 18555, 11),
        'target_pitch': world_pitch_angles[data_index:data_index+3],
        'target_radius': np.array([0.001, 0.002, 0.003]),

    }

    train_config = {
        'var': 1.38e-23 * 50 * 1 * 60 * 205e6 * 10 / 8192,
        'samples': 8192,
        'sample_freq': 205e6,
        'number_channel': 2, # real, imag.
        'number_signal_copies': 1,
        
        'world_size': world_size,
        #'inds': downsample_inds,

        'batchsize': 500,
        'lr': 1e-3,
        'optimizer_class': torch.optim.Adam,
        'objective': torch.nn.CrossEntropyLoss(),
        'epochs': 2500,
        'initial_epoch': 0,
        'loss':[],
    }

    # look for checkpoint
    if config['checkpoint'].exists():
        print('Loading from checkpoint.')
        model, optimizer_state, initial_epoch, loss_list = LoadCheckpoint(config)

        train_config['model'] = model
        train_config['optimizer_state'] = optimizer_state
        train_config['initial_epoch'] = initial_epoch
        train_config['loss'] = loss_list

        print(f'Epochs trained: {initial_epoch + 1}')
    else:
        print('Training from Scratch')

        model = InitModel(model_config)
        #print(model)
        train_config['model'] = model

    TrainModel(config['rank'], config, train_config, model_config)

