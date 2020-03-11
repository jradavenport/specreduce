import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel


def find_illum(flat, threshold=0.9):
    Waxis = 1 # wavelength axis
    # Saxis = 0 # spatial axis

    # compress all wavelength for max S/N
    ycompress = np.nansum(flat, axis=Waxis)

    # find rows (along spatial axis) with illumination above threshold
    ilum = np.where( ((ycompress / np.nanmedian(ycompress)) >= threshold) )[0]
    return ilum


def flat_response(medflat, smooth=False, npix=11, display=False):
    '''

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