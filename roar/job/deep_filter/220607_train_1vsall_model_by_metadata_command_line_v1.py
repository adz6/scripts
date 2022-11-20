import torch
from damselfly import models
import os
import math
from pathlib import Path
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
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
    }

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

def FindInds(
    data_energy,
    data_pitch,
    data_radius,
    target_energy,
    target_pitch,
    target_radius
    ):

    energy_grid, pitch_grid, radius_grid = np.meshgrid(
            target_energy,
            target_pitch,
            target_radius
             )

    target_inds = []
    for i, pair in enumerate(zip(energy_grid.flatten(), pitch_grid.flatten(), radius_grid.flatten())):

        #print(pair)
        try:
            index = np.argwhere(
                np.logical_and(abs(pair[0] - data_energy) < 1e-5, 
                np.logical_and(abs(pair[1] - data_pitch) < 1e-5, abs(pair[2] - data_radius) < 1e-5))
            ).squeeze()
            if index.size == 1:
                target_inds.append(index)
        except:
            pass

    target_inds = np.sort(np.array(target_inds, dtype=np.int32))
    print(target_inds.size)

    return target_inds

def LoadSignal(config, path, samples=8192):

    if path.name.endswith('.npy'):
        signal = np.load(path)
    elif path.name.endswith('.npz'):
        signal = np.load(path)['x']
    elif path.name.endswith('h5'):

        file = h5py.File(path, 'r')

        signal_energy = file['energy'][:]
        signal_pitch = file['pitch'][:]
        signal_radius = file['radius'][:]

        train_data_inds = FindInds(
            signal_energy,
            signal_pitch,
            signal_radius,
            config[f'train_energy'],
            config[f'train_pitch'],
            config[f'train_radius'],
            )

        val_data_inds = FindInds(
            signal_energy,
            signal_pitch,
            signal_radius,
            config[f'val_energy'],
            config[f'val_pitch'],
            config[f'val_radius'],
            )

        background_data_inds = FindInds(
            signal_energy,
            signal_pitch,
            signal_radius,
            config[f'background_energy'],
            config[f'background_pitch'],
            config[f'background_radius'],
            )
        
        train_data = file['x'][train_data_inds, 0:samples]
        val_data = file['x'][val_data_inds, 0:samples]

        n_train = train_data.shape[0]
        n_val = val_data.shape[0]

        # randomly sample the background signals
        randints = np.random.choice(
            background_data_inds,
            size=n_train//2 + n_val//2
            )

        train_background = np.sort(randints[0:n_train//2])
        val_background = np.sort(randints[n_train//2:])

        train_data = np.concatenate(
            (train_data, np.zeros(train_data.shape, dtype=np.complex128)),
            axis=0
            )
        for i, n in enumerate(train_background): 
            train_data[n_train+i] = file['x'][n, 0:samples]
        
        val_data = np.concatenate(
            (val_data, np.zeros(val_data.shape, dtype=np.complex128)),
            axis=0
            )
        for i, n in enumerate(val_background): 
            val_data[n_val+i] = file['x'][n, 0:samples]

        #train_tensor = torch.zeros(
        #    (train_data.shape[0], 2, samples),
        #    dtype=torch.float,
        #    )
        #train_tensor[:, 0, :] = torch.tensor(train_data).real.float()
        #train_tensor[:, 1, :] = torch.tensor(train_data).imag.float()

        #val_tensor = torch.zeros(
        #    (val_data.shape[0], 2, samples),
        #    dtype=torch.float,
        #    )
        #val_tensor[:, 0, :] = torch.tensor(val_data).real.float()
        #val_tensor[:, 1, :] = torch.tensor(val_data).imag.float()

        
    return torch.tensor(train_data), torch.tensor(val_data)

def FFT(signal):

    #print(signal.shape)
    return torch.fft.fftshift(torch.fft.fft(signal, dim=-1, norm='forward'))

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

def TrainModel(config, train_config, model_config):


    model = train_config['model'].cuda()

    optimizer = train_config['optimizer_class'](
            model.parameters(),
            lr=train_config['lr']
        )

    if config['checkpoint'].exists():

        optimizer.load_state_dict(train_config['optimizer_state'])

    with torch.no_grad():
        train_data, val_data = LoadSignal(config, config['data_path'], samples = train_config['samples'])
    
        train_data, val_data = FFT(train_data), FFT(val_data)

        train_tensor = torch.zeros(
            (train_data.shape[0], 2, train_config['samples']),
            dtype=torch.float,
            )
        train_tensor[:, 0, :] = train_data.real.float()
        train_tensor[:, 1, :] = train_data.imag.float()
        train_data = train_tensor

        val_tensor = torch.zeros(
            (val_data.shape[0], 2, train_config['samples']),
            dtype=torch.float,
            )
        val_tensor[:, 0, :] = val_data.real.float()
        val_tensor[:, 1, :] = val_data.imag.float()
        val_data=val_tensor

        train_target = torch.zeros(train_data.shape[0], dtype=torch.long)
        train_target[0:train_target.shape[0]//2] = 1

        val_target = torch.zeros(val_data.shape[0], dtype=torch.long)
        val_target[0:val_target.shape[0]//2] = 1

    train_dataset = torch.utils.data.TensorDataset(train_data, train_target)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_target)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        train_config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        train_config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    objective = train_config['objective'].cuda()
    ep_count = 0

    for ep in range(train_config['epochs']):

        batch_count = 0
        ep_acc = []
        ep_loss = []
        for batch, labels in train_loader:
            
            batch = AddNoise(batch, train_config)

            batch = NormBatch(batch)

            optimizer.zero_grad()
            output = model(batch.cuda())
            loss = objective(output, labels.cuda())
            loss.backward()

            optimizer.step()

            ep_acc.append(BatchAcc(output, labels))
            ep_loss.append(loss.item())

            train_config['loss'].append([ep_count, batch_count, loss.item()])
            batch_count += 1

        # validation check
        with torch.no_grad():
            val_acc = []
            for batch, labels in val_loader:
                batch = AddNoise(batch, train_config)
                batch = NormBatch(batch)
                output = model(batch.cuda())
                val_acc.append(BatchAcc(output, labels))

        print(f'|  {ep + 1}  |  loss = {round(float(np.mean(ep_loss)), 5)}  |  acc = {round(float(np.mean(ep_acc)), 5)}  | val. acc = {round(float(np.mean(val_acc)), 5)}')
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

    dilations = np.arange(325, 350, 1)
    input_size = 8192

    # downsample applied to data tensor function
    #path2inds = Path.home()/'group'/'project'/'results'/\
    #'beamforming'/'time_dependent'/'beamform_signals'/\
    #'220506_dl_grid_87.5to90deg_1to3mm_fftind.npy'
    #downsample_inds = np.load(path2inds)

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

    #### Parameters #########################

    all_pitch = np.linspace(85.0, 88.0, 3001)
    all_energy = np.linspace(18575, 18580, 51)
    all_radius = np.linspace(0.005, 0.005, 1)

    #### Target ###############################

    ensemble_target_pitch = np.array_split(np.linspace(86, 88, 2001), 20)

    target_pitch = ensemble_target_pitch[data_index]
    target_energy = all_energy
    target_radius = all_radius

    target_pitch_mask = np.zeros(all_pitch.size, dtype=bool)
    target_pitch_mask[
        np.argwhere(all_pitch==target_pitch[0]).squeeze():
        np.argwhere(all_pitch==target_pitch[-1]).squeeze()+1
    ] = True
    target_energy_mask = np.ones(all_energy.size, dtype=bool)

    print(target_pitch_mask.sum(), target_energy_mask.sum())

    #### Background ##########################
    background_pitch_mask = ~target_pitch_mask
    background_energy_mask = target_energy_mask

    background_pitch = all_pitch[background_pitch_mask]
    background_energy = all_energy[background_energy_mask]
    background_radius = all_radius

    #### Train ##################################
    train_pitch_mask = np.copy(target_pitch_mask)
    train_pitch_mask[        
        np.argwhere(all_pitch==target_pitch[0]).squeeze():
        np.argwhere(all_pitch==target_pitch[-1]).squeeze()+1:
        4
    ] = False
    train_energy_mask = np.copy(target_energy_mask)
    #train_energy_mask[0::4] = False

    print(train_pitch_mask.sum(), train_energy_mask.sum())

    train_pitch = all_pitch[train_pitch_mask]
    train_energy = all_energy[train_energy_mask]
    train_radius = all_radius

    #### Validation ##########################
    val_pitch_mask = np.copy(train_pitch_mask)
    val_pitch_mask[
        np.argwhere(all_pitch==target_pitch[0]).squeeze():
        np.argwhere(all_pitch==target_pitch[-1]).squeeze()+1
    ] = ~val_pitch_mask[
        np.argwhere(all_pitch==target_pitch[0]).squeeze():
        np.argwhere(all_pitch==target_pitch[-1]).squeeze()+1
        ]
    val_energy_mask = np.copy(train_energy_mask)

    print(val_pitch_mask.sum(), val_energy_mask.sum())

    val_pitch = all_pitch[val_pitch_mask]
    val_energy = all_energy[val_energy_mask]
    val_radius = all_radius

    pitch_start = np.round(target_pitch[0], 1)
    pitch_end = np.round(target_pitch[-1], 1)

    config = {
        ############ DATA #################
        'data_path': Path.home()/'group'/'project'/'results'/'beamforming'/\
        'time_dependent'/'beamform_signals'\
        /'220606_dl_grid_85to88_5mm.h5',
        ############ VAL DATA ###############
        'val_data_path': Path.home()/'group'/'project'/'results'/'beamforming'/\
        'time_dependent'/'beamform_signals'\
        /'220606_dl_grid_85to88_5mm.h5',


        ########### CHECKPOINT ################
        'checkpoint': Path.home()/'group'/'project'/'results'/\
        'machine_learning'/'dnn'/'triggering'/'checkpoints'/'models'
        /f'220607_{pitch_start}to{pitch_end}deg_5mm_model1_1vsall_v1_ensemble.tar',


        #### RANK ########
        #'rank':0,

        ############ DATA #################
        'target_pitch':target_pitch,
        'target_energy':target_energy,
        'target_radius':target_radius,
        'background_pitch':background_pitch,
        'background_energy':background_energy,
        'background_radius':background_radius,
        'train_pitch':train_pitch,
        'train_energy':train_energy,
        'train_radius':train_radius,
        'val_pitch':val_pitch,
        'val_energy':val_energy,
        'val_radius':val_radius,
        'all_pitch': all_pitch,
        'all_energy': all_energy,
        'all_radius': all_radius,

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
        'epochs': 1500,
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

    TrainModel(config, train_config, model_config)

