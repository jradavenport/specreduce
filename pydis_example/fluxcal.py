

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline
from astropy.constants import c as cc

def _mag2flux(wave, mag, zeropt=48.60):
    '''
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    wave : 1d numpy array
        The wavelength of the data points in Angstroms
    mag : 1d numpy array
        The magnitudes of the data
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60)

    Returns
    -------
    Flux values

    Improvements Needed
    -------------------
    1) make input units awareness work (angstroms)
    2) use Spectrum1D object?
    3) is this a function that should be moved to a different package? (specutils)
    '''

    # c = 2.99792458e18 # speed of light, in A/s
    flux = (10.0**( (mag + zeropt) / (-2.5))) * (cc.to('AA/s').value / wave ** 2.0)
    return flux


def AirmassCor(obj_wave, obj_flux, airmass, airmass_file=''):
    """
    Correct the spectrum based on the airmass. Requires observatory extinction file

    Parameters
    ----------
    obj_wave : 1-d array
        The 1-d wavelength array of the spectrum
    obj_flux : 1-d or 2-d array
        The 1-d or 2-d flux array of the spectrum
    airmass : float
        The value of the airmass. Note: not the header keyword.
    airmass_file : str, {'apoextinct.dat', 'ctioextinct.dat', 'kpnoextinct.dat', 'ormextinct.dat'}
        The path to the airmass extinction ascii file, with format:
        [wavelength (AA), Extinction (Mag per Airmass)]

    Returns
    -------
    The extinction corrected flux array

    Improvements Needed
    -------------------
    1) switch inputs to utilize Spectrum1D object
    2) airmass extinction file Table should be astropy units aware
    3) make other wavelength interpolation methods available (?)
    4) figure out if/how to point to observatory extinction library (?)
    """

    if len(airmass_file) == 0:
        raise ValueError('Must select an observatory extinction file.')

    # read in the airmass extinction curve
    Xfile = Table.read(airmass_file, format='ascii', names=('wave', 'X'))
    Xfile['wave'].unit = 'AA'

    # linear interpol airmass extinction onto observed wavelengths
    new_X = np.interp(obj_wave, Xfile['wave'], Xfile['X'])

    # air_cor in units of mag/airmass, convert to flux/airmass
    airmass_ext = 10.0**(0.4 * airmass * new_X)

    # arimas_ext is broadcast to obj_flux if it is a 2-d array
    return obj_flux * airmass_ext


def standard_sensfunc(obj_wave, obj_flux, stdstar='', mode='spline', polydeg=9,
                      display=False):
    """
    Compute the standard star sensitivity function.

    Parameters
    ----------
    obj_wave : 1-d array
        The wavelength array of the observed standard star spectrum in Angstroms
    obj_flux : 1-d array
        The flux array of the observed standard star spectrum
    stdstar : str
        Path to the standard star file to use for flux calibration
    mode : str, optional
        either "linear", "spline", or "poly" (Default is spline)
    polydeg : float, optional
        set the order of the polynomial to fit through (Default is 9)
    display : bool, optional
        If True, plot the sensfunc (Default is False)

    Returns
    -------
    sensfunc : 1-d array
        The sensitivity function for the standard star

    Improvements Needed
    -------------------
    1) use Spectrum1D object for observed std spectrum
    2) make units aware w/ catalog standard
    3) figure out if/how to point to standard star library (?)
    """

    std = Table.read(stdstar, format='ascii', names=('wave', 'mag', 'width'))

    # standard star spectrum is stored in magnitude units (IRAF conventions)
    std_flux = _mag2flux(std['wave'], std['mag'])

    # Automatically exclude some lines b/c resolution dependent response
    badlines = np.array([6563, 4861, 4341], dtype='float') # Balmer lines

    # down-sample (ds) the observed flux to the standard's bins
    obj_flux_ds = np.array([], dtype=np.float)
    obj_wave_ds = np.array([], dtype=np.float)
    std_flux_ds = np.array([], dtype=np.float)
    for i in range(len(std_flux)):
        rng = np.where((obj_wave >= std['wave'][i] - std['width'][i] / 2.0) &
                       (obj_wave < std['wave'][i] + std['width'][i] / 2.0))[0]

        IsH = np.where((badlines >= std['wave'][i] - std['width'][i] / 2.0) &
                       (badlines < std['wave'][i] + std['width'][i] / 2.0))[0]

        # does this bin contain observed spectra, and no Balmer lines?
        if (len(rng) > 1) and (len(IsH) == 0):
            obj_flux_ds = np.append(obj_flux_ds, np.nanmean(obj_flux[rng]))
            obj_wave_ds = np.append(obj_wave_ds, std['wave'][i])
            std_flux_ds = np.append(std_flux_ds, std_flux[i])

    # the ratio between the standard star catalog flux and observed flux
    ratio = np.abs(std_flux_ds / obj_flux_ds)

    # actually fit the log of this sensfunc ratio
    # since IRAF does the 2.5*log(ratio), everything in mag units!
    LogSensfunc = np.log10(ratio)

    # if invalid interpolation mode selected, make it spline
    if mode.lower() not in ('linear', 'spline', 'poly'):
        mode = 'spline'
        print("WARNING: invalid mode set. Changing to spline")

    # interpolate the calibration (sensfunc) on to observed wavelength grid
    if mode.lower()=='linear':
        sensfunc2 = np.interp(obj_wave, obj_wave_ds, LogSensfunc)
    elif mode.lower()=='spline':
        spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2 ,s=0.0025)
        sensfunc2 = spl(obj_wave)
    elif mode.lower()=='poly':
        fit = np.polyfit(obj_wave_ds, LogSensfunc, polydeg)
        sensfunc2 = np.polyval(fit, obj_wave)

    sensfunc_out = 10 ** sensfunc2

    if display is True:
        plt.figure()
        plt.plot(obj_wave, obj_flux * sensfunc_out, c='C0',
                    label='Observed x sensfunc', alpha=0.5)
        # plt.scatter(std['wave'], std_flux, color='C1', alpha=0.75, label=stdstar)
        plt.scatter(obj_wave_ds, std_flux_ds, color='C1', alpha=0.75, label=stdstar)

        plt.xlabel('Wavelength')
        plt.ylabel('Flux')

        plt.xlim(np.nanmin(obj_wave), np.nanmax(obj_wave))
        plt.ylim(np.nanmin(obj_flux * sensfunc_out)*0.98, np.nanmax(obj_flux * sensfunc_out) * 1.02)
        plt.legend()
        plt.show()


    return sensfunc_out


def apply_sensfunc(obj_wave, obj_flux, obj_err, cal_wave, sensfunc):
    # the sensfunc should already be BASICALLY at the same wavelength grid as the target
    # BUT, just in case, we linearly resample it:

    # ensure input array is sorted!
    ss = np.argsort(cal_wave)

    sensfunc2 = np.interp(obj_wave, cal_wave[ss], sensfunc[ss])

    # then simply apply re-sampled sensfunc to target flux
    return obj_flux * sensfunc2, obj_err * sensfunc2
