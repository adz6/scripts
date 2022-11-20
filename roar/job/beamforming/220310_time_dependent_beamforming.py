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

    for i, radius in enumerate(data_radius):
        ind = np.argwhere(
            np.logical_and(data_pitch[i] == gradb_pitch, data_radius[i] == gradb_radii)
            ).squeeze()
        data_gradb_freq[i] = gradb_freq[ind]

    return data_gradb_freq

if __name__ == '__main__':

    nch = 60

    data_path = Path.home()/'group'/'project'/'datasets'/\
    'data'/'220414_dl_grid_84to90_1to45mm.h5'
    freq_path = Path.home()/'group'/'project'/'results'/\
    'mayfly'/'220303_grad_b_frequency_grid_radius_angle.npz'

    data, metadata = LoadData(data_path)

    #pitch_angle_meta = np.round(np.array(metadata['theta_min'][:]), 5)
    energy_meta = np.round(np.array(metadata['energy'][:]), 5)

    data_radius = np.round(np.array(metadata['x_min'][:]), 5)

    data_pitch = np.round(np.array(metadata['theta_min'][:]), 5)

    gradb_freq = np.load(freq_path)['freq'].flatten()
    gradb_radii = np.round(np.load(freq_path)['rad_grid'].flatten(), 5)
    gradb_pitch = np.load(freq_path)['angle_grid'].flatten()
    #gradb_radii, gradb_pitch = np.meshgrid(gradb_radii, gradb_pitch)
    #gradb_radii, gradb_pitch = gradb_radii.flatten(), gradb_pitch.flatten()

    data_freq = SortGradBFreq(
        data_radius,
        data_pitch,
        gradb_radii,
        gradb_pitch,
        gradb_freq
        )

    # downselect data to a specific radius
    data_radius_unique = np.sort(np.unique(data_radius))
    select_rad = data_radius_unique[index_bf]

    select_inds = np.argwhere(data_radius == select_rad).squeeze()

    # select data, and frequencies. Split
    select_data = data[select_inds, :].reshape((select_inds.size, 60, 2*8192))
    select_freq = data_freq[select_inds]
    select_pitch = data_pitch[select_inds]
    select_energy = energy_meta[select_inds]

    select_data_split = np.array_split(select_data, 21, axis=0)
    select_freq_split = np.array_split(select_freq, 21, axis=0)
    select_pitch_split = np.array_split(select_pitch, 21, axis=0)
    select_energy_split = np.array_split(select_energy, 21, axis=0)

    for i in range(len(select_data_split)):

        config = {
                'nch': nch,
                'ant_angles': np.radians(np.arange(0, nch, 1) * 360 / nch),
                'r_array': 0.10,
                'wavelength_lo': 2.998e8 / 25.86e9,
                'f_sample': 205e6,
                'n_sample': 2 * 8192,
                'n_gridx': 7,
                'n_gridy': 7,
                'save_path': Path.home()/'group'/'project'/'results'/\
                'beamforming'/'time_dependent'/'metadata'/\
                'deep_learning',
                'name':f'220416_dl_grid_84to90_gradb_sum_'\
                + f'rad{int(1000*select_rad)}mm_{i}'
            }

        if (config['save_path']/config['name']).exists():
            pass 
        else:

            x_grid, y_grid = np.meshgrid(
                np.linspace(-0.005, 0.005, config['n_gridx']),
                np.linspace(select_rad-0.005,select_rad+0.005, config['n_gridy']),
                )

            coordinates = np.array([x_grid.flatten(), y_grid.flatten()]).T

            summed_signals = ShiftAndSum(config, select_data_split[i], 1.2 * select_freq_split[i], coordinates)
            
            config['save_path'].mkdir(parents=True, exist_ok=True)
            
            np.savez(
                config['save_path']/config['name'],
                signals=summed_signals,
                radius=select_rad,
                pitch=select_pitch_split[i], 
                energy=select_energy_split[i],
                freq=select_freq_split[i],
                )
            
            #np.save(
            #   config['save_path']/config['name'],
            #   select_energy_split[i]
            #)
        print(i+1)



