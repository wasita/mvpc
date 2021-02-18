import nibabel as nb
import numpy as np
import sys
from scipy import stats

class Dataset:
    def __init__(self, dat_fn, mask=None, mask_val=None):
        # Load nifti from file
        ni = nb.load(dat_fn)

        # save affine translation matrix and grid size
        self.a = {'aff':ni.affine}
        self.a['grid'] = ni.shape[:3]

        # Transform data to rectangle
        data = ni.get_fdata()
        data = data.squeeze()
        i,j,k = self.a['grid'] 
        data = data.reshape((i*j*k,-1))

        # get indices for features
        self.fa = {}
        self.fa['f_indx'] = np.arange(i*j*k)

        # if mask provided apply mask to fa and data
        if mask is not None:
            m = nb.load(mask)
            m = m.get_fdata()
            m = m.reshape((-1))
            if m.shape[0] != i*j*k:
                print("ERROR: Mask is not on the same grid as data!!")
                sys.exit()
            if mask_val is not None:
                self.fa['f_indx'] = self.fa['f_indx'][m == mask_val]
                data = data[m == mask_val]
            else:
                self.fa['f_indx'] = self.fa['f_indx'][m >= 1]
                data = data[m >= 1, :]

        # Transpose data so that rows are "samples" and columns are "features"
        self.samples = data.transpose()

        # initialize Sample Attributes "targets" and "chunks"
        self.sa = {'targets':None, 'chunks':None}
        self.targets = None
        self.chunks = None
        self.shape = self.samples.shape

    
    def set_sa(self, sa_name, arr):

        # First check that number of attributes == num samples
        if len(arr) != self.shape[0]:
            print("ERROR: Sample Attributes do not match number of samples")
            sys.exit()
        self.sa[str(sa_name)] = arr
        if sa_name == 'targets':
            self.targets = arr
        if sa_name == 'chunks':
            self.chunks = arr

    def append(self,ds):

        # check if datasets can be appended, i.e., number of features match
        # this is a vertical stacking operation
        if self.samples.shape[1] != ds.samples.shape[1]:
            print("ERROR: Datasets do not match")
            sys.exit()
        self.samples = np.vstack((self.samples, ds.samples))
        self.shape = self.samples.shape

        # deal with appending samples attributes
        # saving this for later. For now use set_sa after appending

    def zscore_by_chunk(self):
        for ch in np.unique(self.chunks):
            X = self.samples[self.chunks == ch, :]
            self.samples[self.chunks == ch, :] = stats.zscore(X)

    def map_to_nifti(self):
        i,j,k = self.a['grid']
        nu_data = np.zeros((i*j*k, self.shape[0]))
        nu_data[self.fa['f_indx'],:] = self.samples.transpose()
        nu_data = nu_data.reshape((i,j,k,-1))
        ni = nb.Nifti1Image(nu_data,self.a['aff'])
        return ni



        
def cross_validated_classification(ds, clf): 
    results = [] 
    for ch in np.unique(ds.chunks): 
        X = ds.samples[ds.chunks != ch,:] 
        y = ds.targets[ds.chunks != ch] 
        test_samp = ds.samples[ds.chunks == ch, :] 
        test_labels = ds.targets[ds.chunks == ch] 
        model = clf(max_iter=2000) 
        model.fit(X,y) 
        pred = model.predict(test_samp) 
        results.append(sum(pred==test_labels)/float(len(pred))) 
    return results        

