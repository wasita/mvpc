import numpy as np
from scipy import stats

# collect 1D data and and run t-test across subects for ROIs
subs = [1, 12, 17, 24, 27, 31, 32, 33, 34, 36, 37, 41]
sl_fn = "../sub-rid0000{n:02}/anal2_roiSL.1D"


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
