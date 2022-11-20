import torch
from damselfly import models
import os
import math
from pathlib import Path
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import h5py

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

def FindIndsRange(
    data_energy,
    data_pitch,
    data_radius,
    target_energy,
    target_pitch,
    target_radius
    ):

    energy_range = (target_energy.min(), target_energy.max())
    pitch_range = (target_pitch.min(), target_pitch.max())
    radius_range = (target_radius.min(), target_radius.max())

    print(energy_range, pitch_range, radius_range)
    target_inds = []
    for i, pair in enumerate(zip(data_energy, data_pitch, data_radius)):

        #print(pair)

        if (energy_range[0]<=pair[0]<=energy_range[1])\
        and (pitch_range[0]<=pair[1]<=pitch_range[1])\
        and (radius_range[0]<=pair[2]<=radius_range[1]):
            target_inds.append(i)

    target_inds = np.sort(np.array(target_inds, dtype=np.int32))
    print(target_inds.size)

    return target_inds

def LoadSignal(config, path, samples=8192):

    if path.name.endswith('.npy'):
        signal = np.load(path)
    elif path.name.endswith('.npz'):
        signal = np.load(path)['x']
    elif path.name.endswith('h5'):

        try:

            file = h5py.File(path, 'r')

            print(list(file.keys()))

            signal_energy = file['meta']['energy'][:]
            signal_pitch = file['meta']['theta_min'][:]
            signal_radius = file['meta']['x_min'][:]

            target_data_inds = FindIndsRange(
                signal_energy,
                signal_pitch,
                signal_radius,
                config[f'target_energy'],
                config[f'target_pitch'],
                config[f'target_radius'],
                )

            #print(target_data_inds)

            background_data_inds = FindIndsRange(
                signal_energy,
                signal_pitch,
                signal_radius,
                config[f'background_energy'],
                config[f'background_pitch'],
                config[f'background_radius'],
                )

            target_array = file['x'][target_data_inds, :]
            background_array = file['x'][background_data_inds, :]
            #print(target_array.shape)

            # check for zero signals

            target_mean = np.mean(abs(target_array), axis=-1)
            target_not_zero = target_mean > 1e-17
            #print(target_not_zero)
            target_array = target_array[target_not_zero, :]

            #print(f'Found {~target_not_zero.sum()} zero signals in the target.')

            background_mean = np.mean(abs(background_array), axis=-1)
            background_not_zero = background_mean > 1e-17
            background_array = background_array[background_not_zero, :]

            #print(f'Found {~background_not_zero.sum()} zero signals in the background.')

            n_target = target_array.shape[0]
            n_background = background_array.shape[0]

            # randomly sample the background signals
            #randints = np.random.choice(
            #    background_data_inds,
            #    size=n_train//2 + n_val//2
            #    )

            #train_background = np.sort(randints[0:n_train//2])
            #val_background = np.sort(randints[n_train//2:])

            eval_data = np.concatenate(
                (target_array, background_array, np.zeros(background_array.shape, dtype=np.complex128)),
                axis=0
            )
        finally:
            file.close()
        
        return torch.tensor(eval_data), n_target, n_background

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

def EvalModel(rank, config, train_config, model_config):

    torch.cuda.set_device(rank)
    print(rank)

    model = train_config['model'].cuda(rank)

    with torch.no_grad():
        eval_data, n_target, n_background = LoadSignal(config, config['eval_data_path'], samples = train_config['samples'])

        print(eval_data.shape)
    
        eval_data = FFT(eval_data)

        eval_tensor = torch.zeros(
            (eval_data.shape[0], 2, train_config['samples']),
            dtype=torch.float,
            )
        eval_tensor[:, 0, :] = eval_data.real.float()
        eval_tensor[:, 1, :] = eval_data.imag.float()
        eval_data = eval_tensor

        eval_target = torch.zeros(eval_data.shape[0], dtype=torch.long)
        eval_target[0:n_target] = 1
        eval_target[n_target:n_target+n_background] = -1 # label background
        # separately here to distinguish between false alarm on noise vs
        # background signals

    eval_dataset = torch.utils.data.TensorDataset(eval_data, eval_target)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        train_config['batchsize'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    ep_count = 0
    eval_labels = []
    eval_outputs = []
    with torch.no_grad():
        for ep in range(train_config['epochs']):
            batch_count = 0
            batch_labels = []
            batch_outputs = []
            for batch, labels in eval_loader:
                batch = AddNoise(batch, train_config)
                batch = NormBatch(batch)
                output = model(batch.cuda(rank))
                batch_outputs.extend(output.cpu().numpy())
                batch_labels.extend(labels.numpy())
                batch_count += 1

            print(f'|  {ep + 1}  |')
            ep_count += 1
            eval_labels.extend(batch_labels)
            eval_outputs.extend(batch_outputs)

    np.savez(
        config['eval'],
        output=np.array(eval_outputs),
        labels=np.array(eval_labels)
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

    #### Parameters #########################

    all_pitch = np.linspace(85.0, 88.0, 3001)
    all_energy = np.linspace(18575, 18580, 51)
    all_radius = np.linspace(0.005, 0.005, 1)

    #### Target ###############################

    target_pitch = np.linspace(87.9, 88.0, 101)
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


    config = {
        ############ EVAL DATA ###############
        'eval_data_path': Path.home()/'group'/'project'/'datasets'/'data'/\
        '220609_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5',
        ########### CHECKPOINT ################
        'checkpoint': Path.home()/'group'/'project'/'results'/\
        'machine_learning'/'dnn'/'triggering'/'deep_filter_paper'/'checkpoints'\
        /'220607_87.9to88.0deg_5mm_model1_1vsall_v1_ensemble.tar',
        ########### EVAL RESULT ###############
        'eval': Path.home()/'group'/'project'/'results'/\
        'machine_learning'/'dnn'/'triggering'/'deep_filter_paper'/'eval'\
        /'220611_87.9to88.0deg_5mm_model1_1vsall_v1_test_output.npz',

        #### RANK ########
        'rank':0,

        ############ DATA #################
        'target_pitch':target_pitch,
        'target_energy':target_energy,
        'target_radius':target_radius,
        'background_pitch':background_pitch,
        'background_energy':background_energy,
        'background_radius':background_radius,
        'all_pitch': all_pitch,
        'all_energy': all_energy,
        'all_radius': all_radius,
    }

    train_config = {
        'var': 1.38e-23 * 50 * 1 * 60 * 205e6 * 10 / 8192,
        'samples': 8192,
        'sample_freq': 205e6,
        'number_channel': 2, # real, imag.
        
        'world_size': world_size,
        #'inds': downsample_inds,

        'batchsize': 500,
        'epochs': 25,
    }

    # look for checkpoint
    if config['checkpoint'].exists():
        print('Loading from checkpoint.')
        model, optimizer_state, initial_epoch, loss_list = LoadCheckpoint(config)

        train_config['model'] = model
        #train_config['optimizer_state'] = optimizer_state
        #train_config['initial_epoch'] = initial_epoch
        #train_config['loss'] = loss_list

        print(f'Epochs trained: {initial_epoch + 1}')
    else:
        print('Could not find checkpoint.')

        #model = InitModel(model_config)
        #print(model)
        #train_config['model'] = model

    EvalModel(config['rank'], config, train_config, model_config)

