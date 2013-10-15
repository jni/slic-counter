import itertools as it
import functools as ft
import multiprocessing as mp

import numpy as np
from scipy import stats

from skimage.segmentation import slic


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
    pass


def slic_unfold(im, params):
    return slic(im, **params)


def slic_cv(image, n_segmentss=[100, 200, 400, 800, 1600, 3200],
            compactness=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], **kwargs):
    """Run SLIC with various parameter settings for later cross-validation.

    Parameters
    ----------
    image : np.ndarray, 2D or 3D grayscale or RGB.
        The image to be segmented.
    n_segmentss : list of int, optional
        The values to try for `n_segments` in `skimage.segmentation.slic`.
    compactness : list of float, optional
        The values to try for `compactness` in `skimage.segmentation.slic`.
    **kwargs : dict, optional
        Other keyword arguments for `skimage.segmentation.slic`.

    Returns
    -------
    segs : list of array of int
        The segmentations returned by SLIC for each parameter combination.
    params : list of tuple
        The parameters given to SLIC corresponding to each segmentation.
    """
    cv_test = list(it.product(n_segmentss, compactness))
    workers = mp.Pool()
    test_slic = ft.partial(slic_unfold, image)
    test_kwargs = []
    for seg, c in cv_test:
        these_kwargs = kwargs.copy()
        these_kwargs['n_segments'] = seg
        these_kwargs['compactness'] = c
        test_kwargs.append(these_kwargs)
    result = workers.map(test_slic, test_kwargs)
    return result, cv_test

