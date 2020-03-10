import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel


# def flatcombine(flatlist, bias, output='FLAT.fits', response=True, trim=True,
#                 display=False):
#     """
#
#     WRAPPER FUNCTION (move?)
#
#     Combine the flat frames in to a master flat image. Subtracts the
#     master bias image first from each flat image. Currently only
#     supports median combining the images.
#
#     Parameters
#     ----------
#     flatlist : str
#         Path to file containing list of flat images.
#     bias : str or 2-d array
#         Either the path to the master bias image (str) or
#         the output from 2-d array output from biascombine
#     output : str, optional
#         Name of the master flat image to write. (Default is "FLAT.fits")
#     response : bool, optional
#         If set to True, first combines the median image stack along the
#         spatial (Y) direction, then fits polynomial to 1D curve, then
#         divides each row in flat by this structure. This nominally divides
#         out the spectrum of the flat field lamp. (Default is True)
#     trim : bool, optional
#         Trim the image using the DATASEC keyword in the header, assuming
#         has format of [0:1024,0:512] (Default is True)
#     display : bool, optional
#         Set to True to show 1d flat, and final flat (Default is False)
#     flat_poly : int, optional
#         Polynomial order to fit 1d flat curve with. Only used if
#         response is set to True. (Default is 5)
#
#
#     Returns
#     -------
#     flat : 2-d array
#         The median combined master flat
#     """
#
#     # read the bias in, BUT we don't know if it's the numpy array or file name
#     if isinstance(bias, str):
#         # read in file if a string
#         bias_im = fits.open(bias)[0].data
#     else:
#         # assume is proper array from biascombine function
#         bias_im = bias
#
#     # assume flatlist is a simple text file with image names
#     # e.g. ls flat.00*b.fits > bflat.lis
#     # files = np.genfromtxt(flatlist,dtype=np.str)
#     files = np.loadtxt(flatlist, dtype=np.str)
#
#     for i in range(0,len(files)):
#         hdu_i = fits.open(files[i])
#         if trim is False:
#             im_i = hdu_i[0].data - bias_im
#         if trim is True:
#             datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':',',').split(',')
#             d = list(map(int, datasec))
#             im_i = hdu_i[0].data[d[2]-1:d[3],d[0]-1:d[1]] - bias_im
#
#         # check for bad regions (not illuminated) in the spatial direction
#         ycomp = im_i.sum(axis=Saxis) # compress to spatial axis only
#         illum_thresh = 0.8 # value compressed data must reach to be used for flat normalization
#         ok = np.where( (ycomp>= np.nanmedian(ycomp)*illum_thresh) )[0]
#
#         # assume a median scaling for each flat to account for possible different exposure times
#         if (i==0):
#             all_data = im_i / np.nanmedian(im_i[ok,:])
#         elif (i>0):
#             all_data = np.dstack( (all_data, im_i / np.nanmedian(im_i[ok,:])) )
#         hdu_i.close(closed=True)
#
#     # do median across whole stack of flat images
#     flat_stack = np.nanmedian(all_data, axis=2)
#
#     # define the wavelength axis
#     Waxis = 0
#     # add a switch in case the spatial/wavelength axis is swapped
#     if Saxis is 0:
#         Waxis = 1
#
#     if response is True:
#         xdata = np.arange(all_data.shape[1]) # x pixels
#
#         # AVERAGE along spatial axis, smooth w/ 5pixel boxcar, take log of summed flux
#         # use the illuminated portion only - NOTE: has axis hard-coded...
#         flat_1d = np.log10(convolve(flat_stack[ok,:].mean(axis=Waxis), Box1DKernel(5)))
#
#         if mode=='spline':
#             spl = UnivariateSpline(xdata, flat_1d, ext=0, k=2 ,s=0.001)
#             flat_curve = 10.0**spl(xdata)
#         elif mode=='poly':
#             # fit log flux with polynomial
#             flat_fit = np.polyfit(xdata, flat_1d, flat_poly)
#             # get rid of log
#             flat_curve = 10.0**np.polyval(flat_fit, xdata)
#
#         if display is True:
#             plt.figure()
#             plt.plot(10.0**flat_1d)
#             plt.plot(xdata, flat_curve,'r', alpha=0.5, lw=0.5)
#             plt.show()
#
#         # divide median stacked flat by this RESPONSE curve
#         flat = np.zeros_like(flat_stack)
#
#         if Saxis is 1:
#             for i in range(flat_stack.shape[Waxis]):
#                 flat[i,:] = flat_stack[i,:] / flat_curve
#         else:
#             for i in range(flat_stack.shape[Waxis]):
#                 flat[:,i] = flat_stack[:,i] / flat_curve
#     else:
#         flat = flat_stack
#
#     if display is True:
#         plt.figure()
#         plt.imshow(flat, origin='lower',aspect='auto')
#         plt.show()
#
#     # write output to disk for later use
#     hduOut = fits.PrimaryHDU(flat)
#     hduOut.writeto(output, overwrite=True)
#
#     return flat ,ok


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