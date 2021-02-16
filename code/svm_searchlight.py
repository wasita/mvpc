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

tasks = ["beh","tax"]
data_fn = dpath+"Qtstats_{}_run-{}.nii.gz"
mask_fn = dpath+"glasser_masks.nii.gz"
animals = ['bird','insect','primate','reptile','ungulate']
behaviors = ['eating','fighting','running','swimming']
twenty_conds = ['{}_{}'.format(a,b) for a in animals for b in behaviors]

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
ds.zscore_by_chunk()

############
ds.set_searchlight_map(eval(open(sl_map_fn,'r').read()))
result = searchlight(ds, nproc=12)

f = open("sl_result.txt",'w')
f.write(str(result))
f.close()

