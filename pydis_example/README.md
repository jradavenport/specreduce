# PyDIS Example

This is a working directory for taking the functionality of PyDIS,
converting it to using more `astropy` methods/objects, 
and generalize to remove IRAF or DIS specifics where possible.

The primary example data comes from the red channel of DIS, a 2005 
observation of the M dwarf "Gl 669A". 

## Functions
- [x] aperture trace ([apextract.py]())
- [x] aperture extract  ([apextract.py]())
- [x] compute sensitivity function (sensfunc) from standard star ([fluxcal.py]())
- [x] apply sensfunc ([fluxcal.py]())
- [x] apply airmass extinction correction ([fluxcal.py]()) 
- [ ] spectral flat fielding (+ilumination corr)
- [ ] identify (+wavelength cal, reidentify, etc)

## Demo Notebooks
- [extract and trace](apextract_demo.ipynb)
- [standard star sensfunc](fluxcal_demo.ipynb) 
- spectral flat fielding
- identify (basic auto?)
- compute & apply wavelength solution
- [full basic reduction with PyDIS functions](apo05.ipynb) (a reference, not suggested for use)