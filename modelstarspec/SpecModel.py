from __future__ import print_function

import os
import glob

import numpy as np

from measure_extinction.stardata import StarData


__all__ = ["FullSpecModel"]


class FullSpecModel():
    """
    Model of the spectrum of a star including dust extinction
    based on stellar atmosphere and extinction models.

    Written to model observed spectra of stars to constrain their
    stellar and dust parameters.  Inspired by the work of
    Fitzpatrick & Massa (1999, ApJ, 525, 1011; 2005, AJ, 130, 1127).

    Uses the data formats from the measure_extinction package to
    give the model spectra in the same format as the observed data.
    """
    def __init__(self,
                 filebase='/home/kgordon/Dust/Ext/Model_Standards_Data/'):
        """
        Read in the stellar atmosphere model and set the dust
        exitinction model.
        """
        self.filebase = filebase

        # save the current directory
        curdir = os.getcwd()

        # read in the model data
        os.chdir("/home/kgordon/Dust/Ext/Model_Standards_Data/")
        model_files = glob.glob(filebase)

        # basic stellar atmosphere data
        self.n_models = len(model_files)
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)

        # read in the stellar atmosphere models
        for i, file in enumerate(model_files):

        # SED
        self.n_bands = reddened_star.data['BANDS'].n_flat_bands
        self.bands = np.zeros((self.n_models,self.n_bands))

        # STIS spectra
        stis_gindxs, = np.where(reddened_star.data['STIS'].npts > 0)
        self.n_stis = len(stis_gindxs)
        self.stis_spectra = np.zeros((self.n_models,self.n_stis))

        for i, file in enumerate(model_files):
            # decode the filename to get the stellar parameters
            Tpos = string.find(file,'T')
            gpos = string.find(file,'g')
            vpos = string.find(file,'v')
            zpos = string.find(file,'z')
            dpos = string.find(file,'.dat')
            self.temps[i] = math.log10(float(file[Tpos+1:gpos]))
            self.gravs[i] = float(file[gpos+1:vpos])*1e-2
            self.mets[i] = math.log10(float(file[zpos+1:dpos]))

            # get the data
            _standard_star = StarData('Model_Standards_Data/'+file, path='../')

            # store the SED data in a useful form for the fitting
            for k, cbandname in enumerate(reddened_star.data['BANDS'].band_fluxes.keys()):
                if _standard_star.data['BANDS'].band_fluxes.get(cbandname) != None:
                    self.bands[i,k] = _standard_star.data['BANDS'].band_fluxes[cbandname][0]
                else:
                    print('Error in standard star import - missing photometric band')
                    print(cbandname)
                    exit()

            # store this STIS data in a useful form for the fitting
            if _standard_star.data['STIS'].n_waves == reddened_star.data['STIS'].n_waves:
                self.stis_spectra[i,:] = _standard_star.data['STIS'].flux[stis_gindxs]
            else:
                print('Error in standard star import - # STIS points different between reddened and standard stars')
                print(_standard_star.data['STIS'].n_waves, reddened_star.data['STIS'].n_waves)
                exit()

        # cd back to the starting directory
        os.chdir(curdir)

        # provide the width in model space for each parameter
        # used in calculating the nearest neighbors
        self.temp_min = min(self.temps)
        self.temp_max = max(self.temps)
        self.temp_width2 = (self.temp_max - self.temp_min)**2
        self.temp_width2 = 1.0

        self.grav_min = min(self.gravs)
        self.grav_max = max(self.gravs)
        self.grav_width2 = (self.grav_max - self.grav_min)**2

        self.met_min = min(self.mets)
        self.met_max = max(self.mets)
        self.met_width2 = (self.met_max - self.met_min)**2
        self.met_width2 *= 4.0

        #print(math.sqrt(self.temp_width2), math.sqrt(self.grav_width2), math.sqrt(self.met_width2))
