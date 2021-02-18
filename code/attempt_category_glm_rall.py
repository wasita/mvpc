import sys
from os import chdir, makedirs
from os.path import exists, join
from subprocess import call
import numpy as np
from mvpa2.base.hdf5 import h5load
#from mvpa2.suite import *

participant = sys.argv[1]

data_dir = ('/dartfs-hpc/scratch/py132_mvpa/MVPA/group1_mvpa/derivatives')
base_dir = data_dir
glm_dir = join(data_dir, 'afni')
suma_dir = join(data_dir, 'freesurfer', 'fsaverage6', 'SUMA')
scripts_dir = join(base_dir, 'code')
fmri_dir = join(base_dir, 'derivatives')
prep_dir = join('/dartfs-hpc/scratch/py132_mvpa/MVPA/attention/derivatives/', 'fmriprep', 'sub-'+participant, 'func')
reg_dir = join(glm_dir, 'regressors', 'rall')
func_dir = join('/dartfs-hpc/scratch/py132_mvpa/MVPA/attention/', 'sub-'+participant, 'func')

if not exists(reg_dir):
    makedirs(reg_dir)

#!/usr/bin/env python

# Run from ~/attention/raw_bids/code with something like
# ./category_glm.py rid000001 |& tee ../logs/glm_rid000001_log.txt &

participant = sys.argv[1]
task = 'beh'

# Convert fmriprep's confounds.tsv for 3dDeconvolve -ortvec
ortvecs = []
for run in [1, 2, 3, 4, 5]:
    with open(join(prep_dir,
                   'sub-{0}_task-{1}_run-{2}_bold_confounds.tsv'.format(
                    participant, task, run))) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    confounds = {}
    for confound_i, confound in enumerate(lines[0]):
        confound_ts = []
        for tr in lines[1:]:
            confound_ts.append(tr[confound_i])
        confounds[confound] = confound_ts

    keep = ['FramewiseDisplacement', 'aCompCor00', 'aCompCor01', 'aCompCor02',
            'aCompCor03', 'aCompCor04', 'aCompCor05', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    ortvec = {c: confounds[c] for c in keep}
    ortvecs.append(ortvec)

    # Create de-meaned and first derivatives of head motion (backward difference with leading zero)
    for motion_reg in ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']:
        motion = [float(m) for m in ortvec[motion_reg]]
        motion_demean = np.array(motion) - np.mean(motion)
        motion_deriv = np.insert(np.diff(motion_demean, n=1), 0, 0)
        assert len(motion_demean) == len(motion_deriv) == len(ortvec[motion_reg])
        ortvec[motion_reg + '_demean'] = ['{:.9f}'.format(m) for m in motion_demean]
        ortvec[motion_reg + '_deriv'] = ['{:.9f}'.format(m) for m in motion_deriv]
        del ortvec[motion_reg]
    assert len(ortvec) == 19

    keep = ortvec.keys()
    assert len(keep) == 19

# Concatenate ortvec time series across runs
rall_ortvec = {}
for confound in keep:
    rall_ts = []
    for run_ortvec in ortvecs:
        rall_ts.extend(run_ortvec[confound])
    assert len(rall_ts) == 980
    rall_ortvec[confound] = rall_ts
assert len(rall_ortvec.keys()) == len(keep)
        
with open(join(reg_dir, 'sub-{0}_task-{1}_rall_bold_ortvec.1D'.format(participant, task)), 'w') as f:
    rows = []
    for tr in range(len(rall_ortvec[keep[0]])):
        row = []
        for confound in keep:
            if rall_ortvec[confound][tr] == 'n/a':
                row.append('0')
            else:
                row.append(rall_ortvec[confound][tr])
        row = '\t'.join(row)
        rows.append(row)
    f.write('\n'.join(rows))

# Reshape BIDS events.tsv to AFNI format
stim_times = {}
beh_reps = []
tax_reps = []
buttons = []
for run in [1, 2, 3, 4, 5]:
    run_stim_times, run_beh_reps, run_tax_reps, run_buttons = {}, [], [], []
    with open(join(func_dir, 'sub-{0}_task-{1}_run-{2}_events.tsv'.format(participant, task, run)), 'r') as f:
        events = [line.strip().split('\t') for line in f.readlines()]
    for event in events[1:]:
        if event[7] != 'none':
            run_buttons.append('{0:.3f}'.format(float(event[0]) + float(event[7])))
        if event[6] == 'behavior':
            run_beh_reps.append(event[0])
        elif event[6] == 'taxonomy':
            run_tax_reps.append(event[0])
        else:
            if run == 1 and event[2] not in stim_times:
                stim_times[event[2]] = []
            if event[2] not in run_stim_times:
                run_stim_times[event[2]] = []
            run_stim_times[event[2]].append(event[0])
    assert len(run_stim_times) == 20
    for c in stim_times:
        # In "beh" task we have an extra non-repetition (bird_eating)  because of collision
        #assert len(run_stim_times[c]) == 4
        stim_times[c].append(run_stim_times[c])
    assert len(stim_times) == 20
    beh_reps.append(run_beh_reps)
    tax_reps.append(run_tax_reps)
    buttons.append(run_buttons)

with open(join(reg_dir, 'sub-{0}_task-{1}_rall_beh_repetitions.txt'.format(participant, task)), 'w') as f:
    f.write('\n'.join([' '.join(run) for run in beh_reps]))
with open(join(reg_dir, 'sub-{0}_task-{1}_rall_tax_repetitions.txt'.format(participant, task)), 'w') as f:
    f.write('\n'.join([' '.join(run) for run in tax_reps]))
with open(join(reg_dir, 'sub-{0}_task-{1}_rall_buttons.txt'.format(participant, task)), 'w') as f:
    f.write('\n'.join([' '.join(run) for run in buttons]))
for category, onsets in stim_times.items():
    with open(join(reg_dir, 'sub-{0}_task-{1}_rall_{2}.txt'.format(participant, task, category)), 'w') as f:
        f.write('\n'.join([' '.join(run) for run in onsets]))
        
# Change directory for AFNI
chdir(glm_dir)

# Run AFNI's 3dDeconvolve
# Set up regressors for 3dDeconvolve
stim_regs = []
for stim_i, stim_label in enumerate(sorted(stim_times.keys())):
    stim_regs.append("-stim_times {0} {1} 'BLOCK(2.0,1)' -stim_label {0} {2}".format(
            stim_i + 1, join(reg_dir, 'sub-{0}_task-{1}_rall_{2}.txt'.format(participant, task, stim_label)), stim_label))

beh_rep_reg = "-stim_times 21 {0} 'BLOCK(2.0,1)' -stim_label 21 behavior_repetition".format(
        join(reg_dir, 'sub-{0}_task-{1}_rall_beh_repetitions.txt'.format(participant, task)))

tax_rep_reg = "-stim_times 22 {0} 'BLOCK(2.0,1)' -stim_label 22 taxonomy_repetition".format(
        join(reg_dir, 'sub-{0}_task-{1}_rall_tax_repetitions.txt'.format(participant, task)))

buttons_reg = "-stim_times 23 {0} 'BLOCK(1.0,1)' -stim_label 23 button_press".format(
        join(reg_dir, 'sub-{0}_task-{1}_rall_buttons.txt'.format(participant, task)))

for side, hemi in [('L', 'lh'), ('R', 'rh')]:
    cmd = ("3dDeconvolve -polort A -jobs 8 -local_times -force_TR 2.0 "
                "-input "
                "{0}/sub-{1}_task-{2}_run-1_bold_space-fsaverage6.{3}.func.gii "
                "{0}/sub-{1}_task-{2}_run-2_bold_space-fsaverage6.{3}.func.gii "
                "{0}/sub-{1}_task-{2}_run-3_bold_space-fsaverage6.{3}.func.gii "
                "{0}/sub-{1}_task-{2}_run-4_bold_space-fsaverage6.{3}.func.gii "
                "{0}/sub-{1}_task-{2}_run-5_bold_space-fsaverage6.{3}.func.gii "
                "-num_stimts 23 ".format(prep_dir, participant, task, side) +
                ' '.join(stim_regs) + ' ' + ' '.join((beh_rep_reg, tax_rep_reg, buttons_reg)) + ' ' +
                "-ortvec {4} "
                "-fout -tout -x1D {0}/sub-{1}_task-{2}_rall_glm.{3}.X.xmat.1D "
                "-xjpeg {0}/sub-{1}_task-{2}_rall_glm.{3}.X.jpg "
                "-fitts {0}/sub-{1}_task-{2}_rall_glm.{3}.fitts "
                "-errts {0}/sub-{1}_task-{2}_rall_glm.{3}.errts "
                "-bucket {0}/sub-{1}_task-{2}_rall_glm.{3}.stats".format(
                    glm_dir, participant, task, hemi, join(reg_dir,
            'sub-{0}_task-{1}_rall_bold_ortvec.1D'.format(participant, task))))
    call(cmd, shell=True)

    # Run AFNI's 3dREMLfit
    chdir(glm_dir)

    with open(join('/dartfs-hpc/scratch/py132_mvpa/MVPA/attention/derivatives/afni/sub-%s/func/' %participant, 'sub-{0}_task-{1}_rall_glm.REML_cmd'.format(participant, task))) as f:
        reml_cmd = '3dREMLfit' + f.read().split('3dREMLfit')[-1]

    # with open(join(glm_dir, 'sub-{0}_task-{1}_rall_glm.REML_cmd'.format(participant, task))) as f: 
    #    reml_cmd = '3dREMLfit' + f.read().split('3dREMLfit')[-1]
    call(reml_cmd, shell=True)

    # Grab coefficients
    chdir(glm_dir)
    call("3dbucket -prefix sub-{0}_task-{1}_rall_glm.{2}.coefs.gii "
         "'sub-{0}_task-{1}_rall_glm.{2}.stats_REML.gii[1..39(2)]'".format(participant, task, hemi), shell=True)

    call("ConvertDset -o_niml_asc -input sub-{0}_task-{1}_rall_glm.{2}.coefs.gii "
         "-prefix sub-{0}_task-{1}_rall_glm.{2}.coefs".format(participant, task, hemi), shell=True)
