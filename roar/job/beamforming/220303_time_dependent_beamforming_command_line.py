import numpy as np 
import math
from pathlib import Path
import mayfly as mf
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('index')
args = parser.parse_args()

index_bf = int(args.index)

def SortSignals(signals, metadata, radii, pitch):

    sort_signals = []
    for i, pair in enumerate(zip(radii, pitch)):
        #print(pair)
        #print(metadata)
        try:
            index = np.int32(metadata[(metadata['x_min']==pair[0]) & (metadata['theta_min']==pair[1])].index[0])
            sort_signals.append(signals[index, :])
        except:
            pass
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

if __name__ == '__main__':

    nch = 60

    data_path = Path.home()/'group'/'project'/'datasets'/'data'/'211116_grad_b_est.h5'
    freq_path = Path.home()/'group'/'project'/'results'/'mayfly'/'211129_grad_b_frequency_grid_radius_angle.npz'
    signals = mf.data.MFDataset(str(data_path)).data

    signal_metadata = pd.DataFrame(mf.data.MFDataset(str(data_path)).metadata)
    frequencies, radii, pitch = np.load(freq_path)['freq'].flatten(), np.load(freq_path)['radii'].flatten(), np.load(freq_path)['angles'].flatten()
    radii, pitch = np.meshgrid(radii, pitch)
    radii, pitch = radii.flatten(), pitch.flatten()

    #select_rad = [r]
    #select_pitch = [88.5]

    #select_rad, select_pitch = np.meshgrid(select_rad, select_pitch)
    #select_rad, select_pitch = select_rad.flatten(), select_pitch.flatten()
    select_freq = []

    for i, pair in enumerate(zip(radii, pitch)):

        ind = np.argwhere(np.logical_and(radii==pair[0], pitch==pair[1])).squeeze()
        select_freq.append(frequencies[ind])

    select_freq = np.array(select_freq)
    select_freq = select_freq[0:signals.shape[0]]

    signals = SortSignals(signals, signal_metadata, radii, pitch).reshape((signals.shape[0], nch, signals.shape[-1]//nch))

    signal_split = np.array_split(signals, 600, axis=0)
    select_freq_split = np.array_split(select_freq, 600)

    config = {
            'nch': nch,
            'ant_angles': np.radians(np.arange(0, nch, 1) * 360 / nch),
            'r_array': 0.10,
            'wavelength_lo': 2.998e8 / 25.86e9,
            'f_sample': 200e6,
            'n_sample': 8192,
            'grid_size': 0.05,
            'n_gridx': 31,
            'n_gridy': 31,
            'save_path': Path.home()/'group'/'project'/'results'/'beamforming'/'time_dependent'/'beamforming_grids',
            'name':f'220304_radius_pitch_grid_offaxis_beamforming_corrected_chunk{index_bf}'
        }

    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.06, 0.06, config['n_gridx']),
        np.linspace(-0.06, 0.06, config['n_gridy']),
        )

    coordinates = np.array([x_grid.flatten(), y_grid.flatten()]).T

    summed_signals = ShiftAndSum(config, signal_split[index_bf], 10 * select_freq_split[index_bf], coordinates)
    
    config['save_path'].mkdir(parents=True, exist_ok=True)
    np.save(config['save_path']/config['name'], summed_signals)




