import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.optimize import curve_fit


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


def identify_widget(x, y):
    '''
    Keaton Bell helped greatly in getting this simple widget framework put together!

    >>> xpxl, waves = identify_widget(xpixels, flux)

    Instructions
    ------------
    0) For proper interactive widgets, ensure you're using the Notebook backend
    in the Jupyter notebook:
    >>> %matplotlib notebook
    1) Click on arc-line features (peaks) in the plot. The Pixel Value box should update.
    2) Enter the known wavelength of the feature in the Wavelength box.
    3) Click the Assign button, a red line will be drawn marking the feature.
    4) Stop the interaction of (or close) the figure.

    '''

    xpxl = []
    waves = []

    # Create widgets, two text boxes and a button
    xval = widgets.BoundedFloatText(
        value=5555.0,
        min=np.nanmin(x),
        max=np.nanmax(x),
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
        rgn = np.where((np.abs(x - event.xdata ) <= 5.))[0]
        try:
            sig_guess = 3.
            p0 = [np.nanmax(y[rgn]), np.nanmedian(y), event.xdata, sig_guess]
            popt, _ = curve_fit(_gaus, x[rgn], y[rgn], p0=p0)
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

        # # clear plot, draw all lines
        # clear_output()
        # fig, ax = plt.subplots(figsize=(9, 3))
        # ax.plot(x, y)
        # fig.canvas.mpl_connect('button_press_event', onplotclick)

        # for i in xpxl:
        #     ax.axvline(i, lw=1, c='r', alpha=0.7)
        # plt.draw()
        # display(widgets.HBox([xval, linename, button]))

        # ax.text(xval.value, pk, linename.value, rotation=90, ha='right')
        ax.axvline(xval.value, lw=1, c='r', alpha=0.7)
        return

    button.on_click(onbuttonclick)

    # Do the plot
    ax.plot(x, y)
    plt.draw()

    # Display widgets
    display(widgets.HBox([xval, linename, button]))

    return np.array(xpxl), np.array(waves)
