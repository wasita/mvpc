#################################################################################
#
#   ROI Search using Parallel processing to speed things up
#
##################################################################################
from amvpa import *
import numpy as np
from sklearn import svm
import sys, os
from joblib import Parallel, delayed
import nibabel as nb
import time

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

# initialize the file to store data
results = opath+"roiSL.1D"
f = open(results,'w')
f.close() # this ensures a blank new file named $results

# Use this many processing cores
# Increase this number for faster processing using more cores.
# say 40 ?
nproc = 20

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
def compute_roi_searchlight(i,mask_val,chance=.05):
    try:
        ds = None
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
        
        j,k = ds.shape

        #results[mask_val] = np.mean(cross_validated_classification(ds, svm.LinearSVC))
        mu = np.mean(cross_validated_classification(ds, svm.LinearSVC))
    except:
        mu = chance
        k = 999
        print("\n!! WARNING: mask {} did not compute".format(mask_val))
    f = open(results, 'a+')
    f.write("{m:04d} {nvox:03d} {acc:.4f}\n".format(m=mask_val, nvox=k, acc=mu))
    f.close()
    print('{} of {}; id: {} nvox: {} svm: {:.4f}'.format(i, len(masks),
            mask_val, k, mu), end='\r')
    

# Now loop in parallel over the set mask ids
Parallel(n_jobs=nproc)(delayed(compute_roi_searchlight)(i,m) for i,m in
        enumerate(masks))

print("00000____=_+_==--=====+++++...,,,,...DONE")
print("result store in {}".format(results))
print("{}".format(time.time()))


