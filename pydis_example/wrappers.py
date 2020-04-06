from astropy.table import Table
from astropy.nddata import CCDData
from ccdproc import trim_image, Combiner
from astropy import units as u
import numpy as np
from .flatfield import find_illum, flat_response
# from .identify import identify


def flatcombine(images, bias, readlist=True, response=True, trim=True, write=False):
    '''
    Combine the flat frames in to a master flat image. Subtracts the
    master bias image first from each flat image.

    Currently only supports median combining the images.

    Parameters
    ----------
    images : string or list of CCDData objects (or list of 2D numpy arrays)
        Can either be the name of a text file containing a simple list of all the
        flat images to read and then combine, or is a list of image data you've
        already read in.
    bias : CCDData object (or 2D numpy array)
        The master bias image, e.g. from `biascombine` to be subtracted from each
        raw flat image, before normalizing, combining, etc. NOTE: If you don't want
        to correct for the bias, just pass a float 0., e.g.:
        >>> flat, ilum = flatcombine('imlist.txt', 0.)
    readlist : bool
        If True, treats `images` as the name of the text file containing a simple
        list of all the flat images. If False, assumes it is a list of CCDData
        objects. (Default: True)
    response : bool
        If True, divides out the flat (e.g. Quartz) lamp spectrum (Default: True)
    trim : bool
        If True, trim every image with the `DATASEC` keyword. NOTE: Only works if
        `readlist=True`. (Default: True)
    write : bool
        Write an output file (BROKEN)

    Returns
    -------
    flat : CCDData object
        The median-combined master flat
    ilum : 1-d numpy array
        The indicies along the spatial axis that register as illuminated

    Improvements Needed
    -------------------
    0) add output writing functionality w/ CCDData
    1) expose header keywords for trim, exptime, etc
    2) enable other combine methods
    3) expose flat_response options (?)

    '''

    # if you pass it a file name containing paths to the flat images
    if readlist:
        thefiles = Table.read(images, format='ascii.no_header', names=['impath'])

        imlist = []
        for k in range(len(thefiles)):
            img = CCDData.read(thefiles['impath'][k], unit=u.adu)

            # bias subtract first
            img.data = img.data - bias

            # put in units of ADU/s
            img.data = img.data / (img.header['EXPTIME'])
            img.unit = u.adu / u.s

            # trim off bias/overscan region
            if trim:
                img = trim_image(img, fits_section=img.header['DATASEC'])

            # normalize each flat by its median
            img.data = img.data / np.nanmedian(img.data)

            imlist.append(img)
        else:
            # if you read the images yourself and combine into a list
            imlist = images

    # now stack the flat images using the ccdproc median combine method
    medflat = Combiner(imlist).median_combine()

    # find the illuminated region of the CCD along the spatial axis
    ilum = find_illum(medflat)

    if response:
        # trim to illuminated region only, and divide out Quartz lamp shape
        flat = flat_response(medflat[ilum, :], smooth=False)
    else:
        # or just trim to illuminated region only
        flat = medflat[ilum, :]

    # if write:
    #     # write output to disk for later use
    #     hduOut = fits.PrimaryHDU(flat)
    #     hduOut.writeto(output, overwrite=True)

    # have a nice day
    return flat, ilum


def identify_image(image, bias):
    '''
    a wrapper to read a HeNeAr image, trim, bias correct, extract a horizontal line, identify

    '''


    return
