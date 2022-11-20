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
            sort_signals.append(np.zeros(signals[0, :].size))
    return np.array(sort_signals)

def ShiftAndSum(config, signal, coordinates):

    x_antenna = config['r_array'] * np.cos(config['ant_angles'])
    y_antenna = config['r_array'] * np.sin(config['ant_angles'])

    x_coord, y_coord = coordinates[:, 0], coordinates[:, 1]

    d = np.sqrt(
        (x_antenna.reshape((x_antenna.size, 1))
        - x_coord.reshape((1, x_coord.size))) ** 2
        + (y_antenna.reshape((y_antenna.size, 1))
        - y_coord.reshape((1, y_coord.size))) ** 2
    )

    antispiral_angle = np.arctan2(
        y_antenna.reshape((y_antenna.size, 1))
        - y_coord.reshape((1, y_coord.size)),
        x_antenna.reshape((x_antenna.size, 1))
        - x_coord.reshape((1, x_coord.size))
    )

    phase_shift = np.exp(
        1j * (2 * math.pi * d / config['wavelength_lo'] - antispiral_angle) 
        )

    print(phase_shift.shape)
    
    summed_signal = (
        phase_shift.reshape((1, *phase_shift.shape, 1))
         * signal.reshape((signal.shape[0], signal.shape[1], 1, signal.shape[2]))
         ).sum(1)
    return summed_signal

if __name__ == '__main__':

    nch = 60

    data_path = Path.home()/'group'/'project'/'datasets'/'data'/'211116_grad_b_est.h5'
    freq_path = Path.home()/'group'/'project'/'results'/'mayfly'/'211129_grad_b_frequency_grid_radius_angle.npz'
    signals = mf.data.MFDataset(str(data_path)).data[:]

    signal_metadata = pd.DataFrame(mf.data.MFDataset(str(data_path)).metadata)
    radii, pitch = np.load(freq_path)['radii'].flatten(), np.load(freq_path)['angles'].flatten()
    radii, pitch = np.meshgrid(radii, pitch)
    radii, pitch = radii.flatten(), pitch.flatten()

    signals = SortSignals(signals, signal_metadata, radii, pitch).reshape((signals.shape[0], nch, signals.shape[-1]//nch))

    signals_split = np.array_split(signals, 300)

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
            'save_path': Path.home()/'group'/'project'/'results'/\
                'beamforming'/'static'/'beamforming_grids'/\
                '220308_radius_pitch_grid_offaxis_static_beamforming',
            'name':f'chunk{index_bf}'
        }

    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.05, 0.05, config['n_gridx']),
        np.linspace(-0.05, 0.05, config['n_gridy']),
        )

    coordinates = np.array([x_grid.flatten(), y_grid.flatten()]).T

    summed_signals = ShiftAndSum(config, signals_split[index_bf], coordinates)
    config['save_path'].mkdir(parents=True, exist_ok=True)
    np.save(config['save_path']/config['name'], summed_signals)






