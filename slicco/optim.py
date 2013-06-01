import itertools as it
import multiprocessing as mp

import numpy as np
from scipy import stats

from skimage.segmentation import slic
from skimage.util import regular_grid


def quantile(ar, q, axis=None):
    """Return the given quantile in an array of values.

    Parameters
    ----------
    ar : np.ndarray, arbitrary shape
        The input array of values.
    q : float in [0, 1]
        The requested quantile
    axis : int in {0, ..., ar.ndim-1}, optional
        Compute the quantile on values along this axis. If none is provided,
        compute over the raveled array.

    Returns
    -------
    qval : float or array of float
        The requested quantile.
    """
    percentile = q * 100.0
    if axis is None:
        return stats.scoreatpercentile(ar.ravel(), percentile)
    if axis < 0:
        axis += ar.ndim
    if not 0 <= axis < ar.ndim:
        raise ValueError('axis provided is out of range. Array shape: ' +
                         str(ar.shape) + '. Axis given: ' + str(axis) +
                         ' or ' + str(axis - ar.ndim))
    out_shape = tuple([s for i, s in enumerate(ar.shape) if i != axis])
    qval = np.zeros(out_shape, float)
    return qval


def superpixel_color_variance(image, segments):
    """Return the min, median, and max color intensity of each segment.

    Parameters
    ----------
    """

def slic_cv(image, n_segmentss=[100, 200, 400, 800, 1600, 3200],
            ratios=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], **kwargs):
    """Run SLIC with various parameter settings to minimise variance.

    Parameters
    ----------
    image : np.ndarray, 2D or 3D grayscale or RGB.
        The image to be segmented.
    n_segmentss : list of int
        The values to try for `n_segments` in `skimage.segmentation.slic`.
    ratios : list of float
        The values to try for `ratio` in `skimage.segmentation.slic`.
    **kwargs : dict
        Other keyword arguments for `skimage.segmentation.slic`.

    Returns
    -------
    """
    to_test = list(it.product(n_segmentss, ratios))
    workers = mp.Pool()
    def test_slic(nseg_ratio_tup):
        nseg, ratio = nseg_ratio_tup
        return slic(image, n_segments=nseg, ratio=ratio, **kwargs)
    result = workers.map(test_slic, to_test)
    
