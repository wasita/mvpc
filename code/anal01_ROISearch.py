from amvpa import *
import numpy as np
from sklearn import svm
import sys, os

sub = sys.argv[1]

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

results = {}

masks = np.hstack((np.arange(1,181), np.arange(1001,1170)))

for i,mask_val in enumerate(masks):
    print("Completed: [ {} / {} ]".format(mask_val, len(masks)), end="\r")
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


    results[mask_val] = np.mean(cross_validated_classification(ds, svm.LinearSVC))


res_ds = Dataset(mask_fn)
res_samp = np.zeros(res_ds.shape)
for m in results:
    res_samp[res_ds.samples==m] = results[m]

res_ds.samples = res_samp
ni = res_ds.map_to_nifti()
ni.to_filename(opath+"roi_search.nii.gz")

print("All done. Go to {} and use Afni, MRICroGl, or whatever to view your "\
        "results".format(opath))

  



