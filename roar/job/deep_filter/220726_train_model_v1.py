import torch
from damselfly import models
import os
import math
from pathlib import Path
import numpy as np
import h5py
import argparse

#default_checkpoint_path = Path.home()/'group'/'project'/'results'/\
#'machine_learning'/'dnn'/'triggering'/'deep_filter_paper'/'checkpoints'
#default_test_path = Path.home()/'group'/'project'/'results'/\
#'machine_learning'/'dnn'/'triggering'/'deep_filter_paper'/'eval'
#default_data_path = Path.home()/'group'/'project'/'datasets'/'data'

parser = argparse.ArgumentParser()
parser.add_argument('--train-epoch', dest='train_epoch', action='store', default=1)
parser.add_argument('--eval-epoch', dest='eval_epoch', action='store',default=1)
parser.add_argument('--batchsize', dest='batchsize', action='store', default=500)

parser.add_argument('--pitch-min', dest='pitch_min', action='store', )
parser.add_argument('--pitch-max', dest='pitch_max', action='store', )
parser.add_argument('--energy-min', dest='energy_min', action='store', default=18575)
parser.add_argument('--energy-max', dest='energy_max', action='store', default=18580)
parser.add_argument('--radius-min', dest='radius_min', action='store', default=0.005)
parser.add_argument('--radius-max', dest='radius_max', action='store', default=0.005)

parser.add_argument('--train-data', dest='train_data', action='store',)
parser.add_argument('--test-data', dest='test_data', action='store',)
parser.add_argument('--name', dest='name', action='store',)

# model configuration
parser.add_argument('--kernel', dest='kernel', action='store', type=int)
parser.add_argument('--npool', dest='npool', action='store', type=int)
parser.add_argument('--nconv', nargs='+', dest='nconv', action='store', type=int)
parser.add_argument('--nfilter', dest='nfilter', action='store', type=int)
parser.add_argument('--nfilter-increase-rate', dest='filter_increase', action='store', type=int, default=2)

parser.add_argument('--nlinear', dest='nlinear', action='store', type=int)
parser.add_argument('--init_linear_size', dest='init_linear', action='store', type=int, default=1024)

args = parser.parse_args()

def SaveCheckpoint(config, train_config, model_config, model_state, optimizer_state, epoch, loss, acc, val_acc):

    save_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'model_config': model_config,
        'epochs': epoch,
        'loss': loss,
        'acc': acc,
        'val_acc': val_acc
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

def MakeConvList(args):
    conv_list = []
    for i in range(args.npool):
        nfilters = (args.filter_increase ** (i)) * args.nfilter
        conv_list.append([])
        # in filters
        if i == 0:
            filter_list = [nfilters for n in range(args.nconv[i])]
            filter_list[0] = 2
            conv_list[i].append(filter_list)
        else:
            filter_list = [nfilters for n in range(args.nconv[i])]
            filter_list[0] = conv_list[i-1][1][-1]
            conv_list[i].append(filter_list)
        # out filters
        filter_list = [nfilters for n in range(args.nconv[i])]
        conv_list[i].append(filter_list)
        # kernel size
        kernel_list = [args.kernel for n in range(args.nconv[i])]
        conv_list[i].append(kernel_list)
        # dilation
        conv_list[i].append([1 for n in range(args.nconv[i])])
        # dropout
        conv_list[i].append([0. for n in range(args.nconv[i])])
        # pool
        conv_list[i].append(args.kernel)

    return conv_list

def MakeLinearList(args):

    input_size = 8192
    conv_output_size = input_size // (args.kernel ** len(args.nconv))

    linear_list = []
    # input features
    lin_input_size = conv_output_size * args.nfilter * (args.filter_increase ** (len(args.nconv) - 1))
    temp = []
    for i in range(args.nlinear):
        if i == 0:
            temp.append(lin_input_size)
        else:
            temp.append(args.init_linear // (2 ** (i-1)))
    linear_list.append(temp)
    # output features
    temp = []
    for i in range(args.nlinear):
        temp.append(args.init_linear // (2 ** (i)))
    linear_list.append(temp)
    # dropout
    temp = [0. for n in range(args.nlinear)]
    linear_list.append(temp)

    return linear_list

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
    energy_range,
    pitch_range,
    radius_range,
    ):

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

def RandomPhase(signals):

    rng = np.random.default_rng()

    phase_shifts = (2 * rng.random(signals.shape[0]) - 1) * np.pi 

    return signals * np.exp(1j * phase_shifts)[:, np.newaxis]

def LoadSignalEval(config, path, samples=8192):

    if path.name.endswith('.npy'):
        signal = np.load(path)
    elif path.name.endswith('.npz'):
        signal = np.load(path)['x']
    elif path.name.endswith('h5'):

        file = h5py.File(path, 'r')
        rng = np.random.default_rng()

        signal_energy = file['meta']['energy'][:]
        signal_pitch = file['meta']['theta_min'][:]
        signal_radius = file['meta']['x_min'][:]

        all_inds = np.arange(0, signal_energy.size, 1)

        target_data_inds = FindIndsRange(
            signal_energy,
            signal_pitch,
            signal_radius,
            config[f'target_energy_range'],
            config[f'target_pitch_range'],
            config[f'target_radius_range'],
            )
        train_inds = target_data_inds
        n_train = train_inds.size

        background_mask = np.isin(all_inds, train_inds, invert=True)
        background_data_inds = all_inds[background_mask]

        #n_train = int(random_choice_target_inds.size * (1 - config['val_split']))
        n_background = background_data_inds.size

        print(f'The number of unique training signals is {n_train}')
        print(f'The number of unique background signals is {n_background}')

        train_background_inds = background_data_inds
        

        number_copies_train = (2 * train_background_inds.size) // train_inds.size

        #if config['rand_phase']:
        train_data = []
        for i in range(number_copies_train):
            train_data.append(
                RandomPhase(file['x'][train_inds, 0:samples])
                )

        n_diff = (2 * train_background_inds.size) - number_copies_train * train_inds.size
        train_data.append(file['x'][train_inds[0:n_diff], 0:samples])
        train_data = np.concatenate((*train_data,),)
        #else:
        #    train_data = file['x'][train_inds, 0:samples]

        n_train = train_data.shape[0]
        print(f'The number of training signals is {n_train}')
        background_data = file['x'][background_data_inds, 0:samples]
        n_background = background_data.shape[0]
        print(f'The number of background signals is {n_background}')

        train_data = np.concatenate(
            (
                train_data, 
                np.zeros(background_data.shape, dtype=np.complex128),
                np.zeros(background_data.shape, dtype=np.complex128)
            ),
            axis=0
            )
        for i in range(n_background): 
            train_data[n_train+i] = background_data[i, :]
        
    return torch.tensor(train_data), n_train, n_background

def LoadSignalTrain(config, path, samples=8192):

    if path.name.endswith('.npy'):
        signal = np.load(path)
    elif path.name.endswith('.npz'):
        signal = np.load(path)['x']
    elif path.name.endswith('h5'):

        file = h5py.File(path, 'r')
        rng = np.random.default_rng()

        signal_energy = file['meta']['energy'][:]
        signal_pitch = file['meta']['theta_min'][:]
        signal_radius = file['meta']['x_min'][:]

        all_inds = np.arange(0, signal_energy.size, 1)

        target_data_inds = FindIndsRange(
            signal_energy,
            signal_pitch,
            signal_radius,
            config[f'target_energy_range'],
            config[f'target_pitch_range'],
            config[f'target_radius_range'],
            )

        background_mask = np.isin(all_inds, target_data_inds, invert=True)
        background_data_inds = all_inds[background_mask]

        # randomly sample/split the train signals
        random_choice_target_inds = rng.choice(
            target_data_inds,
            size=target_data_inds.size,
            replace=False,
            )

        n_train = int(random_choice_target_inds.size * (1 - config['val_split']))
        n_background = background_data_inds.size

        # randomly sample/split the background signals
        rand_choice_background_inds = np.random.choice(
            background_data_inds,
            size=n_background,
            replace=False,
            )
        n_background_train = int(rand_choice_background_inds.size * (1 - config['val_split']))

        train_background_inds = np.sort(rand_choice_background_inds[0:n_background_train])
        val_background_inds = np.sort(rand_choice_background_inds[n_background_train:])

        train_inds = np.sort(random_choice_target_inds[0:n_train])
        val_inds = np.sort(random_choice_target_inds[n_train:])

        n_train = train_inds.size
        n_val = val_inds.size
        n_train_background = train_background_inds.size
        n_val_background = val_background_inds.size

        print(f'The number of unique training signals is {n_train}')
        print(f'The number of unique validation signals is {n_val}')
        print(f'The number of unique train background signals is {n_train_background}')
        print(f'The number of unique val background signals is {n_val_background}')

        #if config['rand_phase']:
        train_data = []
        val_data = []

        number_copies_train = (2 * train_background_inds.size) // train_inds.size
        number_copies_val = (2 * val_background_inds.size) // val_inds.size
        #print(number_copies_train * train_inds.size)
        #print((2 * train_background_inds.size) - number_copies_train * train_inds.size)

        for i in range(number_copies_train):
            train_data.append(
                RandomPhase(file['x'][train_inds, 0:samples])
                )
        n_diff = (2 * train_background_inds.size) - number_copies_train * train_inds.size
        train_data.append(file['x'][train_inds[0:n_diff], 0:samples])

        for i in range(number_copies_val):
            val_data.append(
                RandomPhase(file['x'][val_inds, 0:samples])
                )

        n_diff = (2 * val_background_inds.size) - number_copies_val * val_inds.size
        val_data.append(file['x'][val_inds[0:n_diff], 0:samples])

        train_data = np.concatenate((*train_data,),)
        val_data = np.concatenate((*val_data,),)
        #else:
        #    train_data = file['x'][train_inds, 0:samples]
        #    val_data = file['x'][val_inds, 0:samples]

        n_train = train_data.shape[0]
        print(f'The number of training signals is {n_train}')
        n_val = val_data.shape[0]
        print(f'The number of validation signals is {n_val}')


        # randomly sample the background signals
        #randints = np.random.choice(
        #    background_data_inds,
        #    size=n_train//2 + n_val//2
        #   )

        #train_background = np.sort(randints[0:n_train//2])
        #val_background = np.sort(randints[n_train//2:])

        print(f'The number of background training signals is {train_background_inds.size}')
        print(f'The number of background validation signals is {val_background_inds.size}')

        # concatenate an array of zeros of equal size to the training data. Half of this will be noise and half background.
        train_data = np.concatenate(
            (train_data, np.zeros((2*train_background_inds.size, samples), dtype=np.complex128)),
            axis=0
            )
        for i, n in enumerate(train_background_inds): 
            train_data[n_train+i] = file['x'][n, 0:samples]
        
        val_data = np.concatenate(
            (val_data, np.zeros((2*val_background_inds.size, samples), dtype=np.complex128)),
            axis=0
            )
        for i, n in enumerate(val_background_inds): 
            val_data[n_val+i] = file['x'][n, 0:samples]

    print(f'The train data shape is {train_data.shape}')
    print(f'The val data shape is {val_data.shape}')
        
    return torch.tensor(train_data), torch.tensor(val_data), n_train, n_val

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
        eval_data, n_target, n_background = LoadSignalEval(config, config['eval_data_path'], samples = train_config['samples'])

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
        # eval_target[n_target:n_target+n_background] = -1 # label background
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
        for ep in range(train_config['eval_epochs']):
            batch_count = 0
            batch_labels = []
            batch_outputs = []
            batch_acc = []
            for batch, labels in eval_loader:
                batch = AddNoise(batch, train_config)
                batch = NormBatch(batch)
                output = model(batch.cuda(rank))
                batch_outputs.extend(output.cpu().numpy())
                batch_labels.extend(labels.numpy())
                batch_count += 1
                batch_acc.append(BatchAcc(output, labels))



            print(f'|  {ep + 1}  |  {np.round(np.mean(batch_acc), 4)}  |')
            ep_count += 1
            eval_labels.extend(batch_labels)
            eval_outputs.extend(batch_outputs)

    np.savez(
        config['eval'],
        output=np.array(eval_outputs),
        labels=np.array(eval_labels)
        )

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
        train_data, val_data, n_train, n_val = LoadSignalTrain(config, config['data_path'], samples = train_config['samples'])
    
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
        train_target[0:n_train] = 1

        print(f'There are {n_train} of target 1 and {train_data.shape[0]} overall in the training data.')

        val_target = torch.zeros(val_data.shape[0], dtype=torch.long)
        val_target[0:n_val] = 1
        print(f'There are {n_val} of target 1 and {val_data.shape[0]} overall in the validatation data.')

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
    
    objective = train_config['objective'].cuda(rank)
    ep_count = 0

    for ep in range(train_config['train_epochs']):

        batch_count = 0
        ep_acc = []
        ep_loss = []
        for batch, labels in train_loader:
            
            batch = AddNoise(batch, train_config)

            batch = NormBatch(batch)

            optimizer.zero_grad()
            output = model(batch.cuda(rank))
            loss = objective(output, labels.cuda(rank))
            loss.backward()

            optimizer.step()
            batch_acc = BatchAcc(output, labels)

            ep_acc.append(batch_acc)
            ep_loss.append(loss.item())

            train_config['loss'].append([ep_count, batch_count, loss.item()])
            train_config['acc'].append([ep_count, batch_count, batch_acc])
            batch_count += 1

        # validation check
        with torch.no_grad():
            batch_count = 0
            val_acc = []
            for batch, labels in val_loader:
                batch = AddNoise(batch, train_config)
                batch = NormBatch(batch)
                output = model(batch.cuda(rank))
                val_acc.append(BatchAcc(output, labels))
                train_config['val_acc'].append([ep_count, batch_count, val_acc])
                batch_count += 1

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
                train_config['acc'],
                train_config['val_acc']
                )

    return model.cpu()

if __name__ == "__main__":

    conv_layer_config = MakeConvList(args)
    linear_layer_config = MakeLinearList(args)

    print(conv_layer_config, linear_layer_config)

    
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

    #### Target ###############################

    target_pitch_range = (float(args.pitch_min), float(args.pitch_max))
    target_energy_range = (int(args.energy_min), int(args.energy_max))
    target_radius_range = (float(args.radius_min), float(args.radius_max))

    train_epoch = int(args.train_epoch)
    eval_epoch = int(args.eval_epoch)

    config = {
        ############ TRAIN DATA #################
        'data_path': Path(args.train_data),
        'val_split':0.2,
        ############ TEST DATA ###############
        'eval_data_path': Path(args.test_data),
        ########### CHECKPOINT ################
        'checkpoint': Path(f'{args.name}_checkpoint.tar'),
        ########### EVAL RESULT ###############
        'eval': Path(f'{args.name}_test.npz'),

        #### RANK ########
        'rank':0,

        ############ DATA #################
        'target_pitch_range':target_pitch_range,
        'target_energy_range':target_energy_range,
        'target_radius_range':target_radius_range,
    }

    train_config = {
        'var': 1.38e-23 * 50 * 1 * 60 * 205e6 * 10 / 8192,
        'samples': 8192,
        'sample_freq': 205e6,
        'number_channel': 2, # real, imag.
        
        'world_size': world_size,
        #'inds': downsample_inds,

        'batchsize': int(args.batchsize),
        'lr': 1e-3,
        'optimizer_class': torch.optim.Adam,
        'objective': torch.nn.CrossEntropyLoss(),
        'train_epochs': train_epoch,
        'eval_epochs': eval_epoch,
        'initial_epoch': 0,
        'loss':[],
        'acc':[],
        'val_acc':[],
    }

    print('Training')

    model = InitModel(model_config)

    train_config['model'] = model

    trained_model = TrainModel(config['rank'], config, train_config, model_config)

    #train_config['model'] = trained_model.eval()

    #EvalModel(config['rank'], config, train_config, model_config)


