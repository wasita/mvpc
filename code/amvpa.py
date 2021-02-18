import nibabel as nb
import numpy as np
import sys
from scipy import stats
from joblib import Parallel, delayed
from sklearn import svm


class Dataset:
    def __init__(self, data_src, mask=None, mask_val=None):

        from_nifti = True
        if type(data_src) == str:
            # Load nifti from file
            ni = nb.load(data_src)
        elif type(data_src) == nb.nifti1.Nifti1Image:
            ni = data_src
        elif (type(data_src) == np.ndarray or
                type(data_src) == np.core.memmap.memmap):
            self.samples = data_src
            self.a = {}
            from_nifti = False
        else:
            print("Error: Unknown datasource")
            sys.exit()

        if from_nifti:
            # save affine translation matrix, grid size, header
            self.a = {'aff': ni.affine}
            self.a['grid'] = ni.shape[:3]
            self.a['header'] = ni.header

            # Transform data to rectangle
            data = ni.get_fdata()
            data = data.squeeze()
            i, j, k = self.a['grid'][:3]
            data = data.reshape((i*j*k, -1))

            # get indices for features
            self.fa = {}
            self.fa['f_indx'] = np.arange(data.shape[0])

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

        else:
            # get indices for features
            self.fa = {}
            self.fa['f_indx'] = np.arange(self.samples.shape[1])

        # initialize Sample Attributes "targets" and "chunks"
        self.sa = {'targets': None, 'chunks': None}
        self.targets = None
        self.chunks = None
        self.shape = self.samples.shape

        self.sl_map = None

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

    def append(self, ds):

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
        i, j, k = self.a['grid']
        nsamp = self.samples.shape[0]
        nu_data = np.zeros((i*j*k, nsamp))
        nu_data[self.fa['f_indx'], :] = self.samples.transpose()
        self.a['header']['dim'][5] = self.samples.shape[0]

        nu_data = nu_data.reshape((i, j, k, 1, nsamp))
        ni = nb.Nifti1Image(nu_data, self.a['aff'], header=self.a['header'])
        return ni

    def save_to_nifti(self, filename):
        ni = self.map_to_nifti()
        ni.to_filename(filename)

    def select_features(self, features, elim_zero_var_feats=True):
        cols = np.arange(self.shape[1])
        sel = np.array([f in features for f in self.fa['f_indx']])
        samples = self.samples[:, sel]
        if elim_zero_var_feats:
            bool_array = np.diag(np.dot(samples.T, samples)) > 0
            samples = samples[:, bool_array]
            features = list(np.array(features)[bool_array])
        nu_ds = Dataset(samples)
        nu_ds.a = self.a
        nu_ds.sa = self.sa
        nu_ds.chunks = self.chunks
        nu_ds.targets = self.targets
        nu_ds.fa = {'f_indx': features}
        nu_ds.shape = nu_ds.samples.shape
        return nu_ds

    def set_searchlight_map(self, sl_map):
        self.sl_map = sl_map


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def cross_validated_classification(ds, clf, return_mean=True):
    try:
        results = []
        for ch in np.unique(ds.chunks):
            X = ds.samples[ds.chunks != ch, :]
            y = ds.targets[ds.chunks != ch]
            test_samp = ds.samples[ds.chunks == ch, :]
            test_labels = ds.targets[ds.chunks == ch]
            model = clf(max_iter=2000)
            model.fit(X, y)
            pred = model.predict(test_samp)
            results.append(sum(pred == test_labels)/float(len(pred)))
        if return_mean:
            results = [np.mean(results)]
        return results
    except:
        return "FAIL"


def _run_searchlight(ds, idx, measure, meas_args=None,
                     i=None, n=None, return_on_fail="chance"):
    print("[{}\t\t]\t{} / {}".format(idx[0], i, n), end="\r")
    m_value = measure(ds, meas_args)
    ret_val = None
    if m_value == "FAIL":
        print("<!> WARNING <!> Searchlight {} FAIL!".format(idx[0]))
        if return_on_fail == "chance":
            chance = 1.0/len(np.unique(ds.targets))
            ret_val = [chance]
            print("\t>>> Imputing chance: {}".format(chance))
        else:
            r = return_on_fail
            print("\t>>> Imputing provided return_on_fail value: {}".format(r))
            ret_val = r
    else:
        ret_val = m_value

    ret_val = np.array(ret_val).reshape((len(ret_val), -1))
    return ret_val


def searchlight(ds, measure=None, meas_args=None, nproc=12):

    sl_result = Parallel(n_jobs=nproc)(
        delayed(_run_searchlight)(
            ds.select_features(idx), idx, measure, meas_args,
            i=i, n=len(ds.sl_map))
        for i, idx in enumerate(ds.sl_map))

    i = sl_result[0].shape[0]
    j = len(sl_result)
    samples = np.zeros((i, j))
    for idx, r in enumerate(sl_result):
        samples[:, idx] = r

    sl_ds = Dataset(samples)
    sl_ds.a = ds.a
    sl_ds.fa = ds.fa

    return sl_ds


def spherical_neighborhood(idx, grid, radius=1):
    """ 
    Compute indices of voxels contained in sphere centered on target voxel. 

    Parameters
    ----------
    idx : int, or list
        Index of center voxel correspondent its parent dataset grid, or a list
        of voxel indices where the first element is assumed to be the center
        voxel.

    grid : tuple
        Three element tuple of integers defining the grid space of the sphere,
        and correspondent to the grid of the parent dataset

    radius : int, default = 1
        Radius of sphere. The distance units of radius can be considered as the
        distance from the center of one voxel to the center of an adjacent along
        one of the three grid dimensions. Thus a "sphere" with radius 1 would
        contain the center voxel plus each of the six face-adjacent neighbors,
        and no diagonal neighbors. The centers of the 20 non-face adjacent,
        i.e., the 12 edge-adjacent and 8 corner-adjacent voxels, all being
        individually greater than 1 unit from the center of the center voxel are
        excluded.

    Returns
    -------
    sphere : list
        An ordered list of integers representing the feature indices of the
        sphere. The first member of the list is the center voxel, and subsequent
        elements are ordered by distance to the center voxel, closest first.

    """
    sphere = None
    x, y, z = grid
    # np.array() gives ordered ints
    vol_idx = np.arange(x*y*z).reshape((x, y, z))
    # reshape into 3D
    # rebuilding original space of ds
    indcs = np.indices((x, y, z))  # xyz are grid coordinates
    # lets you map back..
    nu_indices = []

    if (type(idx) == np.int64 or type(idx) == int):
        sphere = [idx]
    elif (type(idx) == list):
        sphere = idx
    else:
        raise InputError(idx, "Must be an integer or list of integers")

    ctr = indcs[:, vol_idx == sphere[0]]

    for v in sphere:
        i, j, k = indcs[:, vol_idx == v]
        candidates = [(i+1, j, k), (i-1, j, k),
                      (i, j+1, k), (i, j-1, k),
                      (i, j, k+1), (i, j, k-1)]  # new candidate voxels
        for c in candidates:
            dst = np.sqrt((ctr[0]-c[0])**2 +
                          (ctr[1]-c[1])**2 +
                          (ctr[2]-c[2])**2)   # Euclidean distance to
            # center voxel

            #   We need to exclude instances of negative coordinates and in cases
            #   of coordinates beyond the range of the defined grid. In either
            #   case, we simply ignore that candidate and continue on to the
            #   next candidate voxel.
            if c[0] < 0 or c[1] < 0 or c[2] < 0:
                continue  # no negative indices allowed here
            else:
                try:
                    # This fails in cases of coordinates beyond the range of the
                    # defined grid.
                    c_idx = (vol_idx[c[0], c[1], c[2]])[
                        0]  # Candidate voxel index
                except:
                    continue

            #   In order to add a new voxel, it needs to pass 2 tests:
            #   1.  It must not already be in the sphere
            #   2.  Its distance from the center voxel must no be greater than
            #       the prescribed radius
            if (c_idx not in sphere and
                    dst <= radius
                ):
                nu_indices.append(c_idx)

    # do not add voxels more than once
    nu_indices = list(np.unique(nu_indices))
    # If there were any voxels added, then add new voxel indices to the sphere
    # and recursively call function
    if len(nu_indices) > 0:
        sphere = sphere + nu_indices
        return spherical_neighborhood(sphere, grid, radius=radius)

    return sphere


def masked_neighborhood(idx, mask, grid, nvox=20, i=None):
    # takes in single index
    # takes in mask which is list of all indices that are in dataset
    # takes grid which is 3D shape of ori ds (66x77xblahblablah)
    # vbox is optional; how many voxs do you want for searchlight?
    # default val of 20. can play with

    nbhood = [idx]
    nu_nbs = [idx]

    # what's nearby?
    while len(nbhood) <= nvox:

        candidates = []
        # iterate over nu neighbors
        for nb in nu_nbs:
            candidates = candidates + spherical_neighborhood(nb, grid)

        # get rid of duplicates
        candidates = list(np.unique(candidates))

        nu_nbs = []
        #   In order to add a new voxel to nbhood, it must pass the following
        #   tests:
        #   1. It must not already be in nbhood
        #   2. It must be within the mask passed to the function
        for c in candidates:
            if c not in nbhood and c in mask:
                nu_nbs.append(c)

        nbhood = nbhood + nu_nbs

    if i is not None:
        print("{} / {}".format(i, len(mask)), end="\r")
    return nbhood[:nvox]

#


def get_masked_neighborhood(ds, nvox=20, nproc=12):
    grid = ds.a['grid']
    mask = ds.fa['f_indx']  # integer values of the features; large numbers
    # bc correspond to original giant grid
    # so we don't have all of those
    # somewhere around 300-400k grid is where you start
    # now only have what's in the mask
    # original ids are imoprtant to map it back to nifti original space

    # use parallel to iterate over every feature
    # gets put into idx in masked_neighborhood
    sl_nbhoods = Parallel(n_jobs=nproc)(
        delayed(masked_neighborhood)(
            m, mask, grid, nvox=nvox, i=i)
        for i, m in enumerate(mask))

    return sl_nbhoods
