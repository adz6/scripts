import subprocess
import shutil
import json
from pathlib import Path
#import numpy as np

######################
# for interactive use
######################


def make_command(**kwargs):

    python_cmd = f'python {kwargs["job_dir"]}/script.py'

    train_opt = f'--train-epoch {kwargs["train_epoch"]}'\
    + f' --eval-epoch {kwargs["eval_epoch"]}'\
    + f' --batchsize {kwargs["batchsize"]}'

    data_opt = f'--pitch-min {kwargs["pitch_min"]}'\
    + f' --pitch-max {kwargs["pitch_max"]}'

    io_opt = f'--train-data {kwargs["train_data"]}'\
    + f' --test-data {kwargs["test_data"]}'\
    + f' --name {kwargs["name"]}'

    model_opt = f'--kernel {kwargs["kernel"]}' + f' --npool {kwargs["npool"]}'\
    + f' --nconv {"".join(str(n)+" "  for n in kwargs["nconv"])}'[0:-1] \
    + f' --nfilter {kwargs["nfilter"]}' + f' --nlinear {kwargs["nlinear"]}'

    return python_cmd+' '+train_opt+' '+data_opt+' '+io_opt+' '+model_opt

def write_job_config(**kwargs):

    job_hardware = f'-l nodes=1:ppn=1:gpus={kwargs["ngpu"]}:{kwargs["gpu_type"]}'
    job_mem = f'-l pmem={kwargs["mem"]}gb'
    job_time = f'-l walltime={kwargs["time_string"]}'
    job_queue = '-l qos=mgc_mri -A dfc13_mri'
    job_name = f'-N {kwargs["name"]}'
    join_output = f'-j oe'
    job_output = f'-o {kwargs["job_dir"]}/train.out'

    job_config = '#PBS'+' '+job_hardware+' '+job_mem+' '+job_time+' '\
    +job_time+' '+job_queue+' '+job_name+' '+join_output+' '+job_output

    return job_config

def write_job_script(job_dir, job_config, command):

    shebang = '#!/bin/bash'
    change_dir = f'cd {str(job_dir)}'

    with open(job_dir/'job.pbs', 'w') as outfile:
        outfile.write(shebang+'\n')
        outfile.write(job_config+'\n')
        outfile.write(change_dir+'\n')
        outfile.write(command)
        outfile.close()

def prepare_directory(config):

    # make the directory
    job_dir = Path(config['train_dir'])/config['job_name']
    job_dir.mkdir(parents=True, exist_ok=True)
    # copy the training script
    shutil.copyfile(config['script'], job_dir/'script.py')
    # write the job script
    job_config = write_job_config(
        ngpu=config['ngpu'],
        gpu_type=config['gpu_type'],
        mem=config['mem'],
        time_string=config['time_string'],
        name=config['job_name'],
        job_dir=str(job_dir)
    )
    job_command = make_command(
        train_epoch=config['train_epoch'],
        eval_epoch=config['eval_epoch'],
        batchsize=config['batchsize'],
        pitch_min=config['pitch_min'],
        pitch_max=config['pitch_max'],
        train_data=config['train_data'],
        test_data=config['test_data'],
        name=config['job_name'],

        kernel=config['kernel'],
        npool=config['npool'],
        nconv=config['nconv'],
        nfilter=config['nfilter'],
        nlinear=config['nlinear']
    )
    write_job_script(job_dir, job_config, job_command)
    # dump the config
    with open(job_dir/'config.json', 'w') as outfile:
        json.dump(config, outfile, indent=4)

def submit_job(config):
    pbs_script = Path(config['train_dir'])/config['job_name']/'job.pbs'

    submit_command = ['qsub', f'{str(pbs_script)}']
    subprocess.run(submit_command)

def run_interactive(config):
    # make the directory
    job_dir = Path(config['train_dir'])/config['job_name']
    job_dir.mkdir(parents=True, exist_ok=True)
    # copy the training script
    shutil.copyfile(config['script'], job_dir/'script.py')

    job_command = make_command(
        job_dir = str(job_dir),
        train_epoch=config['train_epoch'],
        eval_epoch=config['eval_epoch'],
        batchsize=config['batchsize'],
        pitch_min=config['pitch_min'],
        pitch_max=config['pitch_max'],
        train_data=config['train_data'],
        test_data=config['test_data'],
        name=config['job_name'],

        kernel=config['kernel'],
        npool=config['npool'],
        nconv=config['nconv'],
        nfilter=config['nfilter'],
        nlinear=config['nlinear']
    )

    print([*job_command.split(' ')])

    subprocess.run([*job_command.split(' ')])

if __name__=='__main__':

    project = Path.home()/'group'/'project'
    data_repo = project/'datasets'/'data'
    script_repo = project/'scripting'/'execution_scripts'
    output_repo = project/'scripting'/'output'

    config = {
        'train_dir': str(output_repo/'220816_low_pitch_train'),
        'job_name':'train_cnn_2',
        'script': str(script_repo/'machine_learning'/'220816_train_model_v2.py'),
        'time_string':'5:00:00',
        'ngpu': 1,
        'gpu_type': 'gc_v100s',
        'mem': 200,

        'train_data': str(data_repo/'220609_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'),
        'test_data': str(data_repo/'220619_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'),
        'pitch_min': 86.0,
        'pitch_max': 88.0,
        'batchsize': 500,
        'train_epoch':500,
        'eval_epoch': 1,

        'kernel': 3,
        'npool': 2,
        'nconv':[1, 2],
        'nfilter': 30,
        'nlinear': 3
    }


    run_interactive(config)



