import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel


__all__ = ['find_illum', 'flat_response']

def find_illum(flat, threshold=0.9):
    '''
    Use threshold to define the illuminated portion of the image.

    Parameters
    ----------
    flat : CCDData object
        An image, typically the median-combined master flat
    threshold : float
        the fraction to clip to determine the illuminated portion (between 0 and 1)

    Returns
    -------
    ilum : numpy array
        the indicies along the spatial dimension that are illuminated
    '''
    Waxis = 1 # wavelength axis
    # Saxis = 0 # spatial axis

    # compress all wavelength for max S/N
    ycompress = np.nansum(flat, axis=Waxis)

    # find rows (along spatial axis) with illumination above threshold
    ilum = np.where( ((ycompress / np.nanmedian(ycompress)) >= threshold) )[0]
    return ilum


def flat_response(medflat, smooth=False, npix=11, display=False):
    '''
    Divide out the spatially-averaged spectrum response from the flat image.
    This is to remove the spectral response of the flatfield (e.g. Quartz) lamp.

    Input flat is first averaged along the spatial dimension to make a 1-D flat.
    This is optionally smoothed, and then the 1-D flat is divided out of each row
    of the image.

    Note: implicitly assumes spatial and spectral axes are orthogonal, i.e. does not
    trace lines of constant wavelength for normalization.

    Parameters
    ----------
    medflat : CCDData object
        An image, typically the median-combined master flat
    smooth : bool (default=False)
        Should the 1-D, mean-combined flat be smoothed before dividing out?
    npix : int (default=11)
        if `smooth=True`, how big of a boxcar smooth kernel should be used (in pixels)?
    display : bool (default=False)

    Returns
    -------
    flat : CCDData object

    '''

    # Waxis = 1 # wavelength axis
    Saxis = 0 # spatial axis

    # average the data together along the spatial axis
    flat_1d = np.nanmean(medflat, axis=Saxis)

    # optionally: add boxcar smoothing
    if smooth:
        flat_1d = convolve(flat_1d, Box1DKernel(npix), boundary='extend')

    # ADD? this averaged curve could be modeled w/ spline, polynomial, etc

    # now simply divide the response from the flat (e.g. quartz) lamp
    flat = np.zeros_like(medflat)
    for i in range(medflat.shape[Saxis]):
        flat[i, :] = medflat[i, :] / flat_1d

    # once again normalize, since (e.g. if haven't trimmed illumination region)
    # averaging could be skewed by including some non-illuminated portion.
    flat = flat / np.nanmedian(flat)

    # the resulting flat should just show the pixel-to-pixel variations we're after
    if display:
        plt.figure()
        plt.imshow(flat, origin='lower', aspect='auto', cmap=plt.cm.inferno)
        cb = plt.colorbar()
        plt.title('flat')
        plt.show()

    return flat