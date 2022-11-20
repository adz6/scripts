import h5py
import numpy as np
from pathlib import Path

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

def LoadSignal(config, samples=8192):

    file = h5py.File(config['path'], 'r')
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

    print(f'The number of unique training signals is {n_train}')
    random_choice = np.sort(rng.choice(train_inds, size=config['n_matched_filter'], replace=False))
    print(f'The number selected for matched filter estimation is {random_choice.size}')

    meta_data = {}
    train_data = file['x'][random_choice, 0:samples]
    train_pitch = file['meta']['theta_min'][random_choice]
    train_energy = file['meta']['energy'][random_choice]
    meta_data['train_pitch'] = train_pitch
    meta_data['train_energy'] = train_energy


    return train_data, meta_data

def Scores(data, samples=8192):

    var = 1.38e-23 * 200e6 * 60 * 50 * 10  # k_B * BW * N_ch * R * T - for time domain

    norm = 1 / np.sqrt(var * abs(data * data.conjugate()).sum(-1))

    scores = norm * abs(data * data.conjugate()).sum(-1)

    return scores

def Calculate(config):

    data, meta_data = LoadSignal(config)

    scores = Scores(data)

    np.savez(config['result'], scores=scores, **meta_data)

if __name__=='__main__':

    config = {}

    config['path'] = Path.home()/'group'/'project'/'datasets'/'data'/'220619_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'
    config['target_energy_range'] = (18575, 18580)
    config['target_pitch_range'] = (86.7, 88.0)
    config['target_radius_range'] = (0.005, 0.005)

    config['n_matched_filter'] = 10000

    config['result'] = Path.home()/'group'/'project'/'results'/'matched_filter'/'scores'/'220819_dl_test_data_mf_scores.npz'

    Calculate(config)

    
