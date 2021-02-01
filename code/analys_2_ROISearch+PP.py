#################################################################################
#
#   Anaylsis 2. ROI Search plus Parallelism to speed things up
#
#   This starts with the code from Analysis 1 and adds 'Parallel' and 'delayed' from
#   the joblib Python library.
#
#   Goal: Instead of processing each mask one at a time in
#   a for-loop, attempt to run as parallel threads using multiple cores in
#   parallel.
#
##################################################################################
from amvpa import *
import numpy as np
from sklearn import svm
import sys, os
from joblib import Parallel, delayed
import nibabel as nb

sub = sys.argv[1]

# +++++++++++++++++++  SETUP same as in Analysis 1  ++++++++++++++++++++

dpath = "../../preproc/sub-rid0000{}/".format(sub)
opath = "../sub-rid0000{}/".format(sub)

try:
    os.mkdir(opath)
    os.symlink("{}T1w_USAQ.nii.gz".format(dpath), "{}T1w_USAQ.nii.gz".format(opath)) 
except:
    pass # you can only do that once without causing trouble

tasks = ["beh","tax"]
data_fn = dpath+"Qtstats_{}_run-{}.nii.gz"
mask_fn = dpath+"glasser_masks.nii.gz"
animals = ['bird','insect','primate','reptile','ungulate']
behaviors = ['eating','fighting','running','swimming']
twenty_conds = ['{}_{}'.format(a,b) for a in animals for b in behaviors]

# END SETUP 

# as in analysis 1 collect data in dict
results = {}

# instead of using the ranges for mask ids, instead use the mask data to
# determine mask ids
# OLD: masks = np.hstack((np.arange(1,181), np.arange(1001,1170)))
# new:

# load glasser masks and get the 3-dimensional volume array
glassr = nb.load(mask_fn).get_fdata()

# Flatten the masks array to a vector and call np.unique to get each unique voxel
# value exactly once. save these values as a list of ints (i.e., asarray w
# dtype='int').  
masks = np.asarray(np.unique(glassr.flatten()), dtype='int')

# This gives us what we need except it has an uneeded zero
# which should be in the first position..
masks = masks[1:] # chop off first element.

# to check if everything works so far withour going further uncomment:
#print(masks); sys.exit()


# write the core of the former for-loop as a function
# This function tasks two arguments: a dataset, ds, and an int, m. where m is
# the id of the mask
def compute_roi_searchlight(mask_val):

    for task in tasks:
        for r in range(1,6):
            if ds is None:
                ds = Dataset(data_fn.format(task, r), mask=mask_fn,
                        mask_val=mask_val)
            else:
                ds.append(Dataset(data_fn.format(task,r), mask=mask_fn,
                    mask_val=mask_val))

    ds.set_sa('targets', np.tile(twenty_conds,10))
    ds.set_sa('chunks', np.repeat(range(10),20))
    ds.zscore_by_chunk()

    results[mask_val] = np.mean(cross_validated_classification(ds, svm.LinearSVC))

# Now loop in parallel over the set mask ids
Parallel(n_jobs=8, require='sharedmem')(delayed(compute_roi_search)(m) for m in
        masks)

print("00000____=_+_==--=====+++++...,,,,234723hwefdubc897g4efwev...")


res_ds = Dataset(mask_fn)
res_samp = np.zeros(res_ds.shape)
for m in results:
    res_samp[res_ds.samples==m] = results[m]

res_ds.samples = res_samp
ni = res_ds.map_to_nifti()
ni.to_filename(opath+"roi_search.nii.gz")

print("All done. Go to {} and use Afni, MRICroGl, or whatever to view your "\
        "results".format(opath))

  



