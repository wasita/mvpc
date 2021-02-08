import numpy as np
from scipy import stats
from amvpa import *

# collect 1D data and and run t-test across subects for ROIs
subs = [1, 12, 17, 24, 27, 31, 32, 33, 34, 36, 37, 41]
sl_fn = "../sub-rid0000{n:02}/anal2_roiSL.1D"
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

# make a new data matrix to hold results with ROI ids, number of voxels per ROI, individual searchlight
# maps, and Tstats

tstats = T.statistic.reshape((-1,1))
summary_data = np.hstack((dat[:,0:2], res, tstats))

np.savetxt("../anal2_summary",summary_data, fmt='%.4f')

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
ni.to_filename(opath+"analy2_roisearch.nii.gz")
print("All done. Go to {} and use Afni, MRICroGl, or whatever to view your results".format(opath))
