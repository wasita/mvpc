import numpy as np
from scipy import stats
from amvpa import *
import os

# collect 1D data and and run t-test across subects for ROIs
subs = [1, 12, 17, 24, 27, 31, 32, 33, 34, 36, 37, 41]
sl_fn = "../sub-rid0000{n:02}/roiSL.1D"
dpath = "../../preproc/sub-rid000001/" # any subject just to get mask and T1_USAQ
opath = "../group_results/"

try:
    os.mkdir(opath)
except:
    pass
try:
    os.symlink("{}T1w_USAQ.nii.gz".format(dpath),
        "{}T1w_USAQ.nii.gz".format(opath))
except:
    pass # you can only do that once without causing trouble

mask_fn = dpath+"glasser_masks.nii.gz"
ni_template = dpath+"Qtstats_beh_run-1.nii.gz"

res = None

for s in subs:
    

    dat = np.loadtxt(sl_fn.format(n=s))
    dat = dat[dat[:,0].argsort(),:] # important !
    acc = dat[:,2].reshape((-1,1))
    if res is None:
        res = acc
    else:
        res = np.hstack((res,acc))

T = stats.ttest_1samp(res, popmean=.05, axis=1)

# make a new data matrix to hold results with ROI ids, number of voxels per ROI,
# individual searchlight maps, and Tstats

tstats = T.statistic.reshape((-1,1))
summary_data = np.hstack((dat[:,0:2], res, tstats))

np.savetxt("{}roiSL_summary".format(opath),summary_data, fmt='%.4f')

mask_ds = Dataset(mask_fn)
res_ds = Dataset(ni_template)
j,k = res_ds.shape
res_samp = np.zeros((len(subs)+1,k))

i,j = summary_data.shape


for m in range(i):
    feats = mask_ds.samples[0,:]==int(summary_data[m,0])
    nfeats = sum(feats)
    mres = summary_data[m,2:]
    mres = mres.reshape((-1,1))
    x = np.tile(mres, (1,nfeats))
    res_samp[:, feats] = x
    print("{} / {}".format(m,i),end='\r')


res_ds.samples = res_samp
ni = res_ds.map_to_nifti()
ni.to_filename(opath+"roisearch.nii.gz")

# These next few lines fix the AFNI extension information in the header of the
# results file. Because we used the header from Qtstats_beh_rin-1.nii.gz, some
# of the meta-data is wrong, such as the labels for sub-briks and the degrees of
# freedom for the final T-stats brik. The output file contains 13 volumes: one
# for each subject's ROI-Searchlight map, and one for the T-test output across
# 12 subjects (degrees of freedom == 11)
#
# For simplicity we will use python to call AFNI from within this script.
labels = "'" # start with a blank string with a single quote to start
for s in subs:
    labels = labels + "Subject_{:02} ".format(s)
# now add label for final subbrik
labels = labels + "T-stats'" # end with single quote
afni_cmd="3drefit -relabel_all_str {} {}roisearch.nii.gz".format(labels,opath)
os.system(afni_cmd)

afni_cmd = "3drefit -unSTAT -substatpar 12 fitt 11 {}roisearch.nii.gz".format(opath)
os.system(afni_cmd)

print("All done. Go to {} and use Afni, MRICroGl, or whatever to view your results".format(opath))
