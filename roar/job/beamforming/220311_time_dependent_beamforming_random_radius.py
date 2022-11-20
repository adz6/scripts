import numpy as np 
import math
from pathlib import Path
import mayfly as mf
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('index')
args = parser.parse_args()

index_bf = int(args.index)

def SortSignals(signals, metadata, radii, pitch):

    sort_signals = []
    for i, pair in enumerate(zip(radii, pitch)):
        try:
            index = np.int32(metadata[(metadata['x_min']==pair[0]) & (metadata['theta_min']==pair[1])].index[0])
            sort_signals.append(signals[index, :])
        except:
            sort_signals.append(np.zeros(signals[0, :].size))
    return np.array(sort_signals)

def ShiftAndSum(config, signal, gradb_freq, coordinates):
    
    grad_b_angles = -2 * math.pi * np.arange(0, config['n_sample'], 1) \
     * gradb_freq.reshape((signal.shape[0], 1)) / config['f_sample']

    x_antenna = config['r_array'] * np.cos(config['ant_angles'])
    y_antenna = config['r_array'] * np.sin(config['ant_angles'])

    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]
    r_coord = np.sqrt(x_coord ** 2 + y_coord ** 2)

    theta_grid = np.arctan2(y_coord, x_coord)
    theta_grid_grad_b = theta_grid.reshape(theta_grid.size, 1, 1) \
     + grad_b_angles.reshape((1, *grad_b_angles.shape))

    y_grad_b = r_coord.reshape((r_coord.size, 1, 1)) * np.cos(theta_grid_grad_b)
    x_grad_b = r_coord.reshape((r_coord.size, 1, 1)) * np.sin(theta_grid_grad_b)

    d_grad_b = np.sqrt(
        (x_antenna.reshape((x_antenna.size, 1, 1, 1))
        - x_grad_b.reshape((1, *x_grad_b.shape))) ** 2
        + (y_antenna.reshape((y_antenna.size, 1, 1, 1))
        - y_grad_b.reshape((1, *y_grad_b.shape))) ** 2
    )
    print(d_grad_b.shape)

    antispiral_angle = np.arctan2(
        y_antenna.reshape((y_antenna.size, 1, 1, 1))
        - y_grad_b.reshape((1, *y_grad_b.shape)),
        x_antenna.reshape((x_antenna.size, 1, 1, 1))
        - x_grad_b.reshape((1, *x_grad_b.shape))
    )

    phase_shift = np.exp(
        1j * (2 * math.pi * d_grad_b / config['wavelength_lo'] - antispiral_angle) 
        )

    summed_signal = (
        np.swapaxes(phase_shift, 0, 2)
         * signal.reshape((signal.shape[0], 1, signal.shape[1], signal.shape[2]))
         ).sum(2)
    return summed_signal

def LoadData(data_path):

    h5file = h5py.File(data_path)
    data = h5file['data']
    meta = {}

    for i,key in enumerate(h5file['meta'].keys()):
        meta[key] = h5file['meta'][key][:]


    return data, pd.DataFrame(meta)

def SortGradBFreq(data_radius, data_pitch, gradb_radii, gradb_pitch, gradb_freq):

    data_gradb_freq = np.zeros(data_radius.size)
    unique_gradb_pitch = np.unique(gradb_pitch)
    unique_gradb_radii = np.unique(gradb_radii)

    for i, pair in enumerate(zip(data_radius, data_pitch)):

        near_radius = unique_gradb_radii[
            np.argmin(abs(unique_gradb_radii - pair[0]))
        ]
        near_pitch = unique_gradb_pitch[
            np.argmin(abs(unique_gradb_pitch - pair[1]))
        ]

        ind = np.argwhere(
            np.logical_and(gradb_pitch == near_pitch, gradb_radii == near_radius)
            ).squeeze()
        data_gradb_freq[i] = gradb_freq[ind]

    return data_gradb_freq
    
if __name__ == '__main__':

    nch = 60

    data_path = Path.home()/'group'/'project'/'datasets'/\
    'data'/'220310_deep_learning_grid_87deg_100eV_3cm.h5'
    freq_path = Path.home()/'group'/'project'/'results'/\
    'mayfly'/'211129_grad_b_frequency_grid_radius_angle.npz'

    data, metadata = LoadData(data_path)

    data_radius = np.round(np.array(metadata['x_min'][:]), 5)
    data_pitch = np.round(np.array(metadata['theta_min'][:]), 5)

    gradb_freq = np.load(freq_path)['freq'].flatten()
    gradb_radii = np.round(np.load(freq_path)['radii'].flatten(), 5)
    gradb_pitch = np.load(freq_path)['angles'].flatten()
    gradb_radii, gradb_pitch = np.meshgrid(gradb_radii, gradb_pitch)
    gradb_radii, gradb_pitch = gradb_radii.flatten(), gradb_pitch.flatten()

    data_freq = SortGradBFreq(
        data_radius,
        data_pitch,
        gradb_radii,
        gradb_pitch,
        gradb_freq
        )

    # downselect data to a specific radius
    #data_radius_unique = np.sort(np.unique(data_radius))
    select_rad = data_radius[index_bf]
    select_pitch = data_pitch[index_bf]
    select_freq = data_freq[index_bf]
    select_data = data[index_bf, :].reshape((1, 60, data.shape[-1] // 60))

    #select_inds = np.argwhere(data_radius == select_rad).squeeze()

    # select data, and frequencies. Split
    #select_data = data[select_inds, :].reshape((select_inds.size, 60, 8192))
    #select_freq = data_freq[select_inds]

    #select_data_split = np.array_split(select_data, 21, axis=0)
    #select_freq_split = np.array_split(select_freq, 21, axis=0)

    #for i in range(len(select_data_split)):

    config = {
            'nch': nch,
            'ant_angles': np.radians(np.arange(0, nch, 1) * 360 / nch),
            'r_array': 0.10,
            'wavelength_lo': 2.998e8 / 25.86e9,
            'f_sample': 205e6,
            'n_sample': 8192,
            'grid_size': 0.05,
            'n_gridx': 7,
            'n_gridy': 7,
            'save_path': Path.home()/'group'/'project'/'results'/'beamforming'/\
            'time_dependent'/'beamforming_grids'/'220311_dl_87_test_signals',
            'name':f'{index_bf}'
        }

    #if (config['save_path']/config['name']).exists():
    #    pass 
    #else:

    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.005, 0.005, config['n_gridx']),
        np.linspace(select_rad-0.005,select_rad+0.005, config['n_gridy']),
        )

    coordinates = np.array([x_grid.flatten(), y_grid.flatten()]).T

    summed_signals = ShiftAndSum(config, select_data, 12 * select_freq, coordinates)
    print(summed_signals.shape)

    signal_max = np.argmax(np.sum(abs(summed_signals) ** 2, axis=-1), axis=-1).squeeze()
    print(signal_max)

    optimum_signal = summed_signals[0, signal_max, :]
    config['save_path'].mkdir(parents=True, exist_ok=True)
    np.save(config['save_path']/config['name'], optimum_signal)
    #print(i+1)



