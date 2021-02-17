#################################################################################
#
#   ROI Search using Parallel processing to speed things up
#
##################################################################################
from amvpa import *
import sys

sub = sys.argv[1]

# +++++++++++++++++++  SETUP same as in Analysis 1  ++++++++++++++++++++

dpath = "../../preproc/sub-rid0000{}/".format(sub)
opath = "../sub-rid0000{}/".format(sub)

# save to this file:
savefile = "{}svm_sl_20conds.nii".format(opath)

tasks = ["beh","tax"]
data_fn = dpath+"Qtstats_{}_run-{}.nii.gz"
mask_fn = dpath+"glasser_masks.nii.gz"
animals = ['bird','insect','primate','reptile','ungulate']
behaviors = ['eating','fighting','running','swimming']
twenty_conds = ['{}_{}'.format(a,b) for a in animals for b in behaviors]
animals = np.repeat(animals,4) # all animals for one run
behaviors = np.tile(behaviors,5) # all behaviors for each run


## Pre-calculated searchlight map
sl_map_fn = "glassGM_Searchlight.txt"

# END SETUP 

ds = None
for task in tasks:
    for r in range(1,6):
        if ds is None:
            ds = Dataset(data_fn.format(task, r), mask=mask_fn)
        else:
            ds.append(Dataset(data_fn.format(task,r), mask=mask_fn))

ds.set_sa('targets', np.tile(twenty_conds,10))
ds.set_sa('chunks', np.repeat(range(10),20))
ds.set_sa('animals', np.tile(animals,10))
ds.set_sa('behaviors', np.tile(behaviors, 10))

ds.zscore_by_chunk()

############
# Load a pre-computed searchlight space

sl_map = eval(open(sl_map_fn,'r').read())
ds.set_searchlight_map(sl_map)

##### Run searchlight and save to nifti file
measure = cross_validated_classification
measure_args = svm.LinearSVC

# Run for Q1, chunks are runs, targets are 20 conditions
savefile = "{}Q1_svm_sl_20conds.nii".format(opath)

res = searchlight(ds, measure=measure, meas_args=measure_args, nproc=40)
res.save_to_nifti(savefile)
"""
########## Now change the chunks and targets for Q2

ds.targets = ds.sa['animals']
ds.chunks = ds.sa['behaviors']
savefile = "{}Q2_svm_sl_5animals.nii".format(opath)

res = searchlight(ds, measure=measure, meas_args=measure_args, nproc=40)
res.save_to_nifti(savefile)

########## Now change the chunks and targets for Q3

ds.targets = ds.sa['behaviors']
ds.chunks = ds.sa['animals']
savefile = "{}Q3_svm_sl_4behaviors.nii".format(opath)

res = searchlight(ds, measure=measure, meas_args=measure_args, nproc=40)
res.save_to_nifti(savefile)

"""



