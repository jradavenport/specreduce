import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


__all__ = ['identify']


def _gaus(x, a, b, x0, sigma):
    """
    Define a simple Gaussian curve

    Could maybe be swapped out for astropy.modeling.models.Gaussian1D

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def find_peaks(wave, flux, pwidth=10, pthreshold=97, minsep=1):
    '''
    Given a slice thru an arclamp image, find the significant peaks.
    Originally from PyDIS

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelength
    flux : `~numpy.ndarray`
        Flux
    pwidth : float
        the number of pixels around the "peak" to fit over
    pthreshold : float
        Peak threshold
    minsep : float
        Minimum separation

    Returns
    -------
    Peak Pixels, Peak Wavelengths
    '''
    # sort data, cut top x% of flux data as peak threshold
    flux_thresh = np.percentile(flux, pthreshold)

    # find flux above threshold
    high = np.where((flux >= flux_thresh))

    # find  individual peaks (separated by > 1 pixel)
    pk = high[0][1:][ ( (high[0][1:]-high[0][:-1]) > minsep ) ]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    pcent_pix = np.zeros_like(pk,dtype='float')
    wcent_pix = np.zeros_like(pk,dtype='float') # wtemp[pk]
    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt,pcov = curve_fit(_gaus, np.arange(len(xi),dtype='float'), yi,
                                  p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

        except RuntimeError:
            pcent_pix[i] = float('nan')
            wcent_pix[i] = float('nan')

    wcent_pix, ss = np.unique(wcent_pix, return_index=True)
    pcent_pix = pcent_pix[ss]
    okcent = np.where((np.isfinite(pcent_pix)))[0]
    return pcent_pix[okcent], wcent_pix[okcent]


def identify(xpixels, flux, identify_mode='interact', fit_mode='spline',
             linewave=[], autotol=10, polydeg=7):
    '''
    identify's job is to find peaks/features and identify which wavelength they are

    identify methods to develop:
    - interactive widget (basically done)
    - nearest guess from a line list (using approximate wavelength sol'n, e.g. from header)
    - cross-corl from previous solution (best auto method)
    - automatic line hash (maybe get to)

    then it can (by default) generate a solution for the lines/features -> all xpixels
    - polynomial (easy - can use BIC to guess the order)
    - interpolation (easy, but not smooth)
    - spline (fairly simple, but fickle esp. at edges)
    - GaussianProcess

    '''

    #######
    if identify_mode.lower() == 'interact':
        # Keaton Bell helped greatly in getting this simple widget framework put together!
        # Instructions
        # ------------
        # 0) For proper interactive widgets, ensure you're using the Notebook backend
        # in the Jupyter notebook:
        # >>> %matplotlib notebook
        # 1) Click on arc-line features (peaks) in the plot. The Pixel Value box should update.
        # 2) Enter the known wavelength of the feature in the Wavelength box.
        # 3) Click the Assign button, a red line will be drawn marking the feature.
        # 4) Stop the interaction of (or close) the figure.
        xpxl = []
        waves = []

        # Create widgets, two text boxes and a button
        xval = widgets.BoundedFloatText(
            value=5555.0,
            min=np.nanmin(xpixels),
            max=np.nanmax(xpixels),
            step=0.1,
            description='Pixel Value (from click):',
            style={'description_width': 'initial'})

        linename = widgets.Text(  # value='Enter Wavelength',
            placeholder='Enter Wavelength',
            description='Wavelength:',
            style={'description_width': 'initial'})

        button = widgets.Button(description='Assign')

        fig, ax = plt.subplots(figsize=(9, 3))

        # Handle plot clicks
        def onplotclick(event):
            # try to fit a Gaussian in the REGION (rgn) near the click
            rgn = np.where((np.abs(xpixels - event.xdata ) <= 5.))[0]
            try:
                sig_guess = 3.
                p0 = [np.nanmax(flux[rgn]), np.nanmedian(flux), event.xdata, sig_guess]
                popt, _ = curve_fit(_gaus, xpixels[rgn], xpixels[rgn], p0=p0)
                # Record x value of click in text box
                xval.value = popt[2]
            except RuntimeError:
                # fall back to click itself if that doesnt work
                xval.value = event.xdata
            return

        fig.canvas.mpl_connect('button_press_event', onplotclick)

        # Handle button clicks
        def onbuttonclick(_):
            xpxl.append(xval.value)
            waves.append(linename.value)

            ax.axvline(xval.value, lw=1, c='r', alpha=0.7)
            return

        button.on_click(onbuttonclick)

        # Do the plot
        ax.plot(xpixels, flux)
        plt.draw()

        # Display widgets
        display(widgets.HBox([xval, linename, button]))

        xpoints = np.array(xpxl)
        lpoints = np.array(waves)

        # NEED TO DO:
        # Write the lines to a file for later use (Optional?)

    #######
    if identify_mode.lower() == 'nearest':
        # linewave = np.genfromtxt(os.path.join(linelists_dir, linelist), dtype='float',
        #                          skip_header=1,usecols=(0,),unpack=True)

        # in this mode, the xpixel input array is actually the approximate
        # wavelength solution (e.g. from the header info)
        pcent_pix, wcent_pix = find_peaks(xpixels, flux, pwidth=10, pthreshold=97)

        # A simple, greedy, line-finding solution.
        # Loop thru each detected peak, from center outwards. Find nearest
        # known list line. If no known line within tolerance, skip
        xpoints = np.array([], dtype=np.float) # pixel line centers
        lpoints = np.array([], dtype=np.float) # wavelength line centers

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix - np.nanmedian(xpixels)))

        # PLAN: predict solution w/ spline, start in middle, identify nearest match,
        # every time there's a new match, recalc the spline sol'n, work all the way out
        # this both identifies lines, and has byproduct of ending w/ a spline model

        # 1st guess is the peak locations in the wavelength units as given by user
        wcent_guess = wcent_pix

        for i in range(len(pcent_pix)):
            # if there is a match within the tolerance
            if (np.nanmin(np.abs(wcent_guess[ss][i] - linewave)) < autotol):
                # add corresponding pixel and known wavelength to output vectors
                xpoints = np.append(xpoints, pcent_pix[ss[i]])
                lpoints = np.append(lpoints, linewave[np.nanargmin(np.abs(wcent_guess[ss[i]] - linewave))])

                # start guessing new wavelength model after first few lines identified
                if (len(lpoints) > 4):
                    xps = np.argsort(xpoints)
                    spl = UnivariateSpline(xpoints[xps], lpoints[xps], ext=0, k=3, s=1e3)
                    wcent_guess = spl(pcent_pix)


    #######
    # if identify_mode.lower() == 'crosscor':



    # now turn the (xpixel, wavelength) points -> wavelength(x)

    #######
    # Require at least... 4 lines to generate a solution?
    if (len(xpoints) > 4):
        if (fit_mode.lower() == 'spline'):
            # assuming there is a flux value for every xpixel of interest
            # and that it starts at pixel = 0
            # apply our final wavelength spline solution to the entire array
            spl = UnivariateSpline(xpoints, lpoints, ext=0, k=3, s=1e3)
            wavesolved = spl(np.arange(np.size(flux)))

        if (fit_mode.lower() == 'poly'):
            fit = np.polyfit(xpoints, lpoints, polydeg)
            wavesolved = np.polyval(fit, np.arange(np.size(flux)))

    return wavesolved
