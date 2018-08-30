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
                 modeltype='TLusty'):
        """
        Read in the stellar atmosphere model and set the dust
        exitinction model.
        """
        if modeltype == 'TLusty':
            self.modtype = 'TLusty_v10'
            self.filebase = 'T*v10_z*.dat'
            self.path = '/home/kgordon/Dust/Ext/Model_Standards_Data/'
            self.read_tlusty_models(self.filebase, self.path)
        else:
            print('model type not supported')
            exit()

    def read_tlusty_models(self,
                           filebase,
                           path):
        """
        Read in the TLusty stellar model atmosphere predictions.
        All are assumed to be in the measure_extinction data format.
        """
        self.filebase = filebase

        # read in the model data
        model_files = glob.glob("%s/%s" % (path, filebase))

        # basic stellar atmosphere data
        self.n_models = len(model_files)
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)

        # read in the stellar atmosphere models
        self.model_spectra = []
        for i, file in enumerate(model_files):
            # decode the filename to get the stellar parameters
            spos = file.rfind('/')
            Tpos = file.find('T', spos)
            gpos = file.find('g', spos)
            vpos = file.find('v', spos)
            zpos = file.find('z', spos)
            dpos = file.find('.dat', spos)
            self.temps[i] = np.log10(float(file[Tpos+1:gpos]))
            self.gravs[i] = float(file[gpos+1:vpos])*1e-2
            self.mets[i] = np.log10(float(file[zpos+1:dpos]))

            # get the data
            self.model_spectra.append(StarData(file),
                                      path='../')

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
