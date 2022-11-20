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

#def Load(config, key, samples=8192):

    #file = h5py.File(config[key], 'r')
    #rng = np.random.default_rng()

    #signal_energy = file['meta']['energy'][:]
    #signal_pitch = file['meta']['theta_min'][:]
    #signal_radius = file['meta']['x_min'][:]

    #all_inds = np.arange(0, signal_energy.size, 1)

    #target_data_inds = FindIndsRange(
    #    signal_energy,
    #    signal_pitch,
    #    signal_radius,
    #    config[f'target_energy_range'],
    #    config[f'target_pitch_range'],
    #    config[f'target_radius_range'],
    #    )
    #train_inds = target_data_inds
    #n_train = train_inds.size

    #print(f'The number of unique training signals is {n_train}')
    #random_choice = np.sort(rng.choice(train_inds, size=config['n_matched_filter'], replace=False))
    #print(f'The number selected for matched filter estimation is {random_choice.size}')

    #meta_data = {}
    #train_data = file['x'][:, 0:samples]
    #train_pitch = file['meta']['theta_min'][:]
    #train_energy = file['meta']['energy'][:]
    #meta_data['pitch'] = train_pitch
    #meta_data['energy'] = train_energy

    #return  meta_data

def SelfScores(data):

    var = 1.38e-23 * 200e6 * 60 * 50 * 10  # k_B * BW * N_ch * R * T - for time domain

    try:
        h5data = h5py.File(data, 'r')
        ndata = h5data['x'].shape[0]
        scores = np.zeros(ndata)

        data_inds = np.array_split(np.arange(0, ndata, 1), 1 + ndata // 10000)

        for i, temp_data_inds in enumerate(data_inds):

            temp_data = h5data['x'][temp_data_inds, 0:8192]
            norm = 1 / np.sqrt(var * abs(temp_data * temp_data.conjugate()).sum(-1))
            temp_scores = abs(norm[:,np.newaxis] * temp_data * temp_data.conjugate()).sum(axis=-1)

            for k, temp_ind in enumerate(temp_data_inds):
                scores[temp_ind] =  temp_scores[k]

    finally:
        h5data.close()

    return scores

def Scores(data, template, samples=8192):

    var = 1.38e-23 * 200e6 * 60 * 50 * 10  # k_B * BW * N_ch * R * T - for time domain

    try:
        h5data = h5py.File(data, 'r')
        h5template = h5py.File(template, 'r')

        ndata = h5data['x'].shape[0]
        ntemplate = h5template['x'].shape[0]
        scores = np.zeros((ndata, ntemplate))

        data_inds = np.array_split(np.arange(0, ndata, 1), 1 + ndata // 10000)
        template_inds = np.array_split(np.arange(0, ntemplate, 1), 1 + ntemplate // 10000)

        for i, temp_data_inds in enumerate(data_inds):
            for j, temp_template_inds in enumerate(template_inds):

                temp_template = h5template['x'][temp_template_inds, 0:8192]
                temp_data = h5data['x'][temp_data_inds, 0:8192]
                norm = 1 / np.sqrt(var * abs(temp_template * temp_template.conjugate()).sum(-1))
                temp_scores = abs(np.matmul(norm[:,np.newaxis] * temp_template, temp_data.T.conjugate())).flatten()

                temp_grid_data, temp_grid_template = np.meshgrid(temp_data_inds, temp_template_inds)

                for k, pair in enumerate(zip(temp_grid_data.flatten(), temp_grid_template.flatten())):
                    scores[pair[0], pair[1]] =  temp_scores[k]

                print(f'{i+1} / {len(data_inds)}, {j+1} / {len(template_inds)}')

    finally:
        h5data.close()
        h5template.close()

    return scores

def Calculate(config):

    #signal, meta_signal = Load(config, 'signals')
    #template, meta_template = Load(config, 'templates')

    #meta_data = {}
    #meta_data['signal_pitch'] = meta_signal['pitch']
    #meta_data['signal_energy'] = meta_signal['energy']
    #meta_data['template_pitch'] = meta_template['pitch']
    #meta_data['template_energy'] = meta_template['energy']

    #print(signal.shape, template.shape)

    print('calculating scores')
    scores = Scores(config['signals'], config['templates'])
    print('calculating signal ideal scores')
    signal_ideal = SelfScores(config['signals'])
    print('calculating template ideal scores')
    template_ideal = SelfScores(config['templates'])

    meta_data = {}

    try:
        h5signal = h5py.File(config['signals'], 'r')
        h5template = h5py.File(config['templates'], 'r')

        meta_data['signal_pitch'] = h5signal['meta']['theta_min'][:]
        meta_data['signal_energy'] = h5signal['meta']['energy'][:]
        meta_data['template_pitch'] = h5template['meta']['theta_min'][:]
        meta_data['template_energy'] = h5template['meta']['energy'][:]

    finally:
        h5signal.close()
        h5template.close()

    np.savez(config['result'], scores=scores, signal_ideal_scores=signal_ideal, template_ideal_scores=template_ideal,  **meta_data)

if __name__=='__main__':

    config = {}

    config['signals'] = Path.home()/'group'/'project'/'datasets'/'data'/'220901_dl_test_data_85to90deg_5mm.h5'
    config['templates'] = Path.home()/'group'/'project'/'datasets'/'data'/'220922_dl_grid_data_85to90deg_18575to18580ev_5mm.h5'

    config['target_energy_range'] = (18575, 18580)
    config['target_pitch_range'] = (85.5, 88.6)
    config['target_radius_range'] = (0.005, 0.005)

    config['result'] = Path.home()/'group'/'project'/'results'/'matched_filter'/'scores'/'220922_dl_test_data_mf_scores.npz'

    Calculate(config)

    
