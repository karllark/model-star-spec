#!/usr/bin/env python
#
# Program to simultaneously fit an extinguished star observation
# consisting of both photometric and spectroscopic data.  The fit
# parameters are
# stellar [log(T), log(g), Z],
# dust extinction [A(V), R(V), C1, C2, C3, C4, x_o, gamma], and
# gas absorption [N(HI), vel(HI)] (Ly-alpha only).
#
# written Dec 2014/Jan 2015 by Karl Gordon (kgordon@stsci.edu)
# based strongly on IDL programs created over the previous 10 years
#

from __future__ import print_function
import glob
import os
import string
import math
import time

import argparse

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as pyplot
import emcee
import triangle

from astropy.table import Table, Column
from astropy.io import ascii
from astropy.io import fits

from getstardata import *
from extdata import *
import ccm89
import fm90
import f99

# the object to store the stellar models
class StellarModels():
    def __init__(self, filebase, reddened_star):
        self.filebase = filebase

        # save the current directory
        curdir = os.getcwd()

        # read in the model data
        os.chdir("/home/kgordon/Dust/Ext/Model_Standards_Data/")
        model_files = glob.glob(filebase)
        
        # basic data
        self.n_models = len(model_files)
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)

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

# get stellar model photometry at a specified log(T), log(g), and Z
# params = [logT, logg, Z]
def get_stellar_photometry(params, stellar_models):
    # determine the distance between the requested point and the full list of stellar models
    dist2 = (params[0] - stellar_models.temps)**2/stellar_models.temp_width2 + \
            (params[1] - stellar_models.gravs)**2/stellar_models.grav_width2 + \
            (params[2] - stellar_models.mets)**2/stellar_models.met_width2

    sindxs = np.argsort(dist2)

    # number of nearest neighbors to use
    n_near = 11

    # generate model SED form nearest neighbors
    weights = (1.0/np.sqrt(dist2[sindxs[0:n_near]]))
    weights /= np.sum(weights)
    model_fluxes = np.zeros(stellar_models.n_bands)
    for i in range(n_near):
        k = sindxs[i]
        model_fluxes += weights[i]*stellar_models.bands[k,:]

    return model_fluxes

def get_nir_rv(params, reddened_star, standards):

    bindxs = np.argsort(abs(reddened_star.data['BANDS'].flat_bands_waves - 0.438))
    bk = bindxs[0]
    vindxs = np.argsort(abs(reddened_star.data['BANDS'].flat_bands_waves - 0.545))
    vk = vindxs[0]
    kindxs = np.argsort(abs(reddened_star.data['BANDS'].flat_bands_waves - 2.1))
    kk = kindxs[0]

    model_fluxes_bands = get_stellar_photometry(params, standards)
    evk = -2.5*math.log10(reddened_star.data['BANDS'].flat_bands_fluxes[kk]/model_fluxes_bands[kk]) + 2.5*math.log10(reddened_star.data['BANDS'].flat_bands_fluxes[vk]/model_fluxes_bands[vk])
    ebv = -2.5*math.log10(reddened_star.data['BANDS'].flat_bands_fluxes[bk]/model_fluxes_bands[bk]) + 2.5*math.log10(reddened_star.data['BANDS'].flat_bands_fluxes[vk]/model_fluxes_bands[vk])

    return -1.1*evk/ebv
    
# get stellar SED at a specified log(T), log(g), and Z
# params = [logT, logg, Z]
def get_stellar_sed(params, stellar_velocity, waves, stellar_models, n_bands):
    # determine the distance between the requested point and the full list of stellar models
    dist2 = (params[0] - stellar_models.temps)**2/stellar_models.temp_width2 + \
            (params[1] - stellar_models.gravs)**2/stellar_models.grav_width2 + \
            (params[2] - stellar_models.mets)**2/stellar_models.met_width2

    sindxs = np.argsort(dist2)

    # number of nearest neighbors to use
    n_near = 11

    # generate model SED form nearest neighbors
    weights = (1.0/np.sqrt(dist2[sindxs[0:n_near]]))
    weights /= np.sum(weights)
    model_fluxes = np.zeros(stellar_models.n_bands + stellar_models.n_stis)
    for i in range(n_near):
        k = sindxs[i]
        model_fluxes += weights[i]*np.concatenate([stellar_models.bands[k,:], stellar_models.stis_spectra[k,:]])
        #print(dist2[k], weights[i] , stellar_models.temps[k], stellar_models.gravs[k], stellar_models.mets[k])

    # shift the model a systemic velocity
    model_fluxes_shifted = np.copy(model_fluxes)
    model_fluxes_shifted[n_bands:] = np.interp(waves[n_bands:], (1.0 + stellar_velocity/3e5)*waves[n_bands:], model_fluxes[n_bands:])

    #sindxs = np.argsort(waves)
    #fig = pyplot.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(waves[sindxs], model_fluxes[sindxs], 'g-')
    #ax.plot(waves[sindxs], model_fluxes_shifted[sindxs], 'r-')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #pyplot.show()
    #exit()

    return model_fluxes_shifted

# extingish an SED/spectrum by dust with a specific A(V), R(V), and FM parameters
# params = [Av, Rv, C2, C3, C4, x0, gamma]
def extinguish_by_dust(params, sed_x, sed_flux):
    #alav = np.zeros(len(sed_x))
    # use the CCM89 relationship for IR/Opt
    #niropt_indxs = np.where(sed_x < 1.0/0.3)
    #alav[niropt_indxs] = ccm89.ccm89(params[1],sed_x[niropt_indxs])
        
    # use the FM90 parametrers for the UV
    #uv_indxs = np.where(sed_x >= 1.0/0.3)
    #_c1 = 2.09 - 2.84*params[2]  # upated for FM07
    #alav[uv_indxs] = (fm90.fm90(np.insert(params[2:],0,_c1),sed_x[uv_indxs])/params[1]) + 1.0

    alav = f99.f99(params[1], sed_x, c2=params[2], c3=params[3], c4=params[4], x0=params[5], gamma=params[6])

    #alav = ccm89.ccm89(params[1],sed_x)

    #fig = pyplot.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(sed_x, alav, 'r-')
    #ax.plot(sed_x, alav_f99, 'g-')
    #pyplot.show()

    return sed_flux*(10**(-0.4*alav*params[0]))

# put in the hi absorption at Ly-alpha to measure of the HI column
# params = [hi_col, hi_vel]
def hi_absorption(params, velocities, waves, sed_flux):
    # wavelengths of HI lines
    # only use Ly-alpha right now - others useful later
    h_lines = np.array([1215.0,1025.0,972.0,949.0,937.0,930.0,926.0,923.0,920,919.0,918.])*1e-4
    h_width = 100.*1e-4

    # get the wavelengths that will be affected by Ly-alpha absorption
    hindxs, = np.where(abs(waves - h_lines[0]) < h_width)

    hi_sed_flux = np.copy(sed_flux)

    if len(hindxs) > 0:

        # compute the Ly-alpha abs
        # from Bohlin et al. (197?)

        # do all velocity components (# params = # velocities)
        for i, vel in enumerate(velocities):
            lya_abs_wave = (1.0 + (vel/3e5))*h_lines[0]
            phi = 4.26e-20/((1e4*(waves[hindxs] - lya_abs_wave))**2 + 6.04e-10)

            nhi = 10**params[i]
            for k, cphi in enumerate(phi):
                hi_sed_flux[hindxs[k]] *= math.exp(-nhi*cphi)
                
        #fig = pyplot.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.plot(waves[hindxs], sed_flux[hindxs], 'bo')
        #ax.plot(waves[hindxs], hi_sed_flux[hindxs], 'go')
        #pyplot.show()

    # return the SED flux even if the wavelength range does not include HI abs lines
    return hi_sed_flux

# computer the ln(prob) for an input set of model parameters
def lnprobsed(params, waves, flux, weights, stellar_models, param_limits, n_bands, velocities, foreground_params=None, sptype_params=None):

    # make sure the parameters are within the specific limits
    for k, plimit in enumerate(param_limits):
        if (params[k] < plimit[0]) | (params[k] > plimit[1]):
            return -np.inf

    # get the interpolated stellar model
    model_fluxes = get_stellar_sed(params[0:3], velocities[0], waves, stellar_models, n_bands)

    # extinguish by a fixed foreground component
    if foreground_params is not None:
        model_fluxes *= 10**(-0.4*f99.f99(foreground_params[1], 1.0/waves)*foreground_params[0])

    # get the dust extinguished SED (account for the systemic velocity of the galaxy [opposite regular sense])
    shifted_waves = (1.0 - velocities[0]/3e5)*waves
    ext_model_fluxes = extinguish_by_dust(params[3:10], 1.0/shifted_waves, model_fluxes)

    # absorption by HI at Ly-alpha
    hi_ext_model_fluxes = hi_absorption(params[10:], velocities, waves, ext_model_fluxes)

    # get the normalization value for the model
    # basically normalize to have the same average flux in the NIR/OPT
    #indxs = np.where(waves > 0.35)
    #_data_ave = np.average(flux[indxs])
    #_model_ave = np.average(ext_model_fluxes[indxs])
    _data_ave = np.average(flux[0:n_bands])
    _model_ave = np.average(ext_model_fluxes[0:n_bands])

    # compute the ln(prob)
    lnp = -0.5*np.sum(((flux/_data_ave - hi_ext_model_fluxes/_model_ave)**2)*weights)

    # multiply by the prior from the spectral type
    if sptype_params is not None:
        lnp_prior_sptype = -0.5*((params[0] - sptype_params[0])/sptype_params[1])**2
        lnp += lnp_prior_sptype

    if math.isinf(lnp) | math.isnan(lnp):
        print(lnp)
        print(params)
        exit()
    else:
        return lnp

def get_best_fit_params(sampler, reddended_star, standards):

    max_lnp = -1e6
    nwalkers = len(sampler.lnprobability)
    for k in range(nwalkers):
        tmax_lnp = np.max(sampler.lnprobability[k])
        if tmax_lnp > max_lnp:
            max_lnp = tmax_lnp
            indxs, = np.where(sampler.lnprobability[k] == tmax_lnp)
            fit_params_best = sampler.chain[k,indxs[0],:]

    ndim = len(fit_params_best)
    params_best = np.zeros((ndim+4))
    params_best[0:ndim] = fit_params_best
    params_best[ndim] = params_best[3]/params_best[4]
    params_best[ndim+1] = (10**params_best[10])/params_best[3]
    params_best[ndim+2] = (10**params_best[10])/params_best[ndim]

    # get the NIR derived R(V)
    params_best[ndim+3] = get_nir_rv(params_best[0:3], reddened_star, standards)

    return params_best

def get_percentile_params(samples, reddended_star, standards):

    # add in E(B-V) and N(HI)/A(V) and N(HI)/E(B-V)
    samples_shape = samples.shape
    new_samples_shape = (samples_shape[0], samples_shape[1]+4)
    new_samples = np.zeros(new_samples_shape)
    new_samples[:,0:ndim] = samples
    new_samples[:,ndim] = samples[:,3]/samples[:,4]
    new_samples[:,ndim+1] = (10**samples[:,10])/samples[:,3]
    new_samples[:,ndim+2] = (10**samples[:,10])/new_samples[:,ndim]

    # compute the NIR R(V)
    for k in range(samples_shape[0]):
        new_samples[k,ndim+3] = get_nir_rv(new_samples[k,0:3], reddened_star, standards)

    per_params = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(new_samples, [16, 50, 84],
                                        axis=0)))

    return per_params

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of star (assumes starname.dat and starname_stis.dat files exist)")
    parser.add_argument("-f", "--fast", help="Use minimal walkers, n_steps, n_burn to debug code",
                        action="store_true")
    parser.add_argument("-t", "--triangle", help="Plot the 'triangle' plot showing the 1D and 2D likelihood functions",
                        action="store_true")
    parser.add_argument("-w", "--walkers", help="Plot the walker values",
                        action="store_true")
    parser.add_argument("--m31", help="Use M31 parameters",
                        action="store_true")
    parser.add_argument("--smc", help="Use SMC parameters",
                        action="store_true")
    parser.add_argument("--spteff", metavar=('logteff', 'logteff_unc'), type=float, nargs=2,
                        help="Spectral type prior: log(Teff) sigma_log(Teff)")

    parser.add_argument("--velocity", metavar=('velocity'), type=float, nargs=1,
                        help="Stellar velocity")

    args = parser.parse_args()

    # put together the sptype prior info
    if args.spteff:
        spteff_info = args.spteff
    else:
        spteff_info = None

    # save the start time 
    start_time = time.clock()

    # get the reddened star data
    #starname = 'j003733.35+400036.6'
    starname = args.starname
    reddened_star = StarData('DAT_files/'+starname+'.dat', path='/home/kgordon/Dust/Ext/')

    # get the stellar model grid
    #standards = StellarModels("T*v2_z*.dat",reddened_star)
    standards = StellarModels("T*v10_z*.dat",reddened_star)

    # combine the reddened data into single vectors
    stis_gindxs, = np.where(reddened_star.data['STIS'].npts > 0)
    n_bands = len(reddened_star.data['BANDS'].band_fluxes)
    red_waves = np.concatenate([reddened_star.data['BANDS'].flat_bands_waves,reddened_star.data['STIS'].waves[stis_gindxs]])
    red_fluxes = np.concatenate([reddened_star.data['BANDS'].flat_bands_fluxes,reddened_star.data['STIS'].flux[stis_gindxs]])
    red_uncs = np.concatenate([reddened_star.data['BANDS'].flat_bands_uncs,reddened_star.data['STIS'].uncs[stis_gindxs]])

    _data_ave = np.average(red_fluxes[0:n_bands])
    #red_weights = (red_uncs/_data_ave)**(-2.)
    #red_weights /= np.average(red_weights)  # to avoid numerical issues
    red_weights = np.full(len(red_uncs),1.0)

    # make the band data have a higher weight
    red_weights[0:n_bands] *= 100.

    # do the cropping of "bad" spectral regions
    #  bad regions are defined as those were we know the models do not work
    ex_regions = [[8.23-0.1,8.23+0.1],  # geocoronal line
                  [8.7,10.0],  # bad data from STIS
                  [3.55,3.6],
                  [3.80,3.90],
                  [4.15,4.3],
                  [6.4,6.6], 
                  [7.1,7.3], 
                  [7.45,7.55],
                  [7.65,7.75],
                  [7.9,7.95],
                  [8.05,8.1]]

    x = 1.0/red_waves[n_bands:]
    weight_fac = np.full(len(x), 1.0)
    for exreg in ex_regions:
        indxs, = np.where((x >= exreg[0]) & (x <= exreg[1]))
        if len(indxs) > 0:
            weight_fac[indxs] = 0.0
            reddened_star.data['STIS'].npts[stis_gindxs[indxs]] = 0
    red_weights[n_bands:] *= weight_fac

    # full list of parameter names (including derived parameters)
    var_names = ['log($T_{eff}$)','log(g)','log(Z)',
                 'A(V)','R(V)','C$_2$','C$_3$','C$_4$','x$_0$','$\gamma$',
                 'log(NHI)','logMW(NHI)','E(B-V)','N(HI)/A(V)','N(HI)/E(B-V)','R(V)IR']

    # setup the EMCEE parameters
    # setup the (optional) foreground dust extinction paramters
    ext_max_yval = 15.0
    if args.m31:
        foreground_params = [0.06*3.1, 3.1]
        print('using M31 foreground extinction parameters and param limits')
        param_limits = [[standards.temp_min, standards.temp_max],
                        [standards.grav_min, standards.grav_max],
                        [-0.3,standards.met_max],
                        [0.0,4.0],
                        [1.0,7.0],
                        [-0.5, 1.5],
                        [0.0, 6.0],
                        [-0.2, 2.0],
                        [4.55, 4.65],
                        [0., 2.5],
                        [17., 24.],
                        [17., 22.]]

        # inital guesses at parameters
        p0 = np.array([4.5,2.5,0.1,
                       1.0,3.1,0.11,0.25,0.09,4.60,1.40,
                       21., 19.])

        if args.velocity:
            velocities = [args.velocity[0], 0.0]
        else:
            velocities = [-300., 0.0]

        print(velocities)

    elif args.smc:
        foreground_params = None

        param_limits = [[standards.temp_min, standards.temp_max],
                        [standards.grav_min, standards.grav_max],
                        [standards.met_min,standards.met_max],
                        [0.0,4.0],
                        [1.0,7.0],
                        [-0.5, 3.0],
                        [-0.5, 6.0],
                        [-1.0, 2.0],
                        [4.55, 4.65],
                        [0.25, 1.5],
                        [17., 24.],
                        [17., 22.]]
        # inital guesses at parameters
        p0 = np.array([4.5,2.5,0.1,
                       1.0,3.1,0.11,0.25,0.09,4.60,1.40,
                       21., 19.])

        velocities = [158., 0.0]

        ext_max_yval = 20.0
    else:
        foreground_params = None

        param_limits = [[standards.temp_min, standards.temp_max],
                        [standards.grav_min, standards.grav_max],
                        [standards.met_min,standards.met_max],
                        [0.0,4.0],
                        [1.0,7.0],
                        [-0.5, 1.5],
                        [0.0, 6.0],
                        [-0.2, 2.0],
                        [4.4, 4.8],
                        [0., 2.5],
                        [17., 24.],
                        [17., 22.]]
        # inital guesses at parameters
        p0 = np.array([4.5,2.5,0.1,
                       1.0,3.1,0.11,0.25,0.09,4.60,1.40,
                       21., 19.])

        velocities = [0.0, 0.0]

    setup_time = time.clock()
    print('setup time taken: ',(setup_time - start_time)/60., ' min')

    ndim = len(p0)
    if args.fast:
        nwalkers = 2*ndim
        nsteps = 50
        burn   = 5
    else:
        nwalkers = 100
        nsteps = 5000
        burn   = 2000
        #nwalkers = 100
        #nsteps = 5000
        #burn   = 2000

    # setting up the walkers to start "near" the inital guess
    p  = [ p0*(1+0.01*np.random.normal(0,1.,ndim))  for k in xrange(nwalkers)]

    # ensure that all the walkers start within the allowed volume
    for pc in p:
        for k, plimit in enumerate(param_limits):
            if pc[k] < plimit[0]:
                pc[k] = plimit[0]
            elif pc[k] > plimit[1]:
                pc[k] = plimit[1]

    # setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobsed,
                                    args=(red_waves,red_fluxes,red_weights,standards, param_limits,
                                          n_bands, velocities, foreground_params, spteff_info))

    # burn in the walkers
    pos, prob, state = sampler.run_mcmc(p, burn)

    # rest the sampler
    sampler.reset()

    # do the full sampling
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state )

    emcee_time = time.clock()
    print('emcee time taken: ',(emcee_time - setup_time)/60., ' min')

    # create the samples variable for later use
    samples = sampler.chain.reshape((-1, ndim))

    # get the best fit values
    params_best = get_best_fit_params(sampler, reddened_star, standards)

    # get the 16, 50, and 84 percentiles
    params_per = get_percentile_params(samples, reddened_star, standards)

    # save the best fit and p50 +/- uncs values to a file
    # save as a single row table to provide a uniform format
    f = open(starname+'_fit_params.dat', 'w')
    f.write("# best fit, p50, +unc, -unc\n")
    for k, val in enumerate(params_per):
        if (params_best[k] < 1e4):
            formatstr = "%.2f"
        else:
            formatstr = "%.2e"

        f.write(formatstr%params_best[k] + '  ' + formatstr%val[0] + '  ' + formatstr%val[1] + ' ' +
                formatstr%val[2] + ' # ' + var_names[k] + '\n')

    # create the p50 parameters with symmetric error bars
    params_50p = np.zeros(len(params_per))
    params_50p_uncs = np.zeros(len(params_per))
    for k, cval in enumerate(params_per):
        params_50p[k] = cval[0]
        params_50p_uncs[k] = 0.5*(cval[1] + cval[2])

    # get the model spectrum that corresponds to the 50p parameters
    #model_fluxes = get_stellar_sed(params_50p[0:4], red_waves, standards, n_bands)
    #if len(foreground_params) > 1:
    #    model_fluxes *= 10**(-0.4*f99.f99(foreground_params[1], 1.0/red_waves)*foreground_params[0])
    #ext_model_fluxes = extinguish_by_dust(params_50p[4:11], 1.0/red_waves, model_fluxes)
    #hi_ext_model_fluxes = hi_absorption(params_50p[11:], red_waves, ext_model_fluxes)
    # useful for making the "observed" extinction curve
    #model_fluxes_w_hi_abs = hi_absorption(params_50p[11:], red_waves, model_fluxes)

    # get the model spectrum that corresponds to the best parameters
    model_fluxes = get_stellar_sed(params_best[0:3], velocities[0], red_waves, standards, n_bands)
    if foreground_params is not None:
        model_fluxes *= 10**(-0.4*f99.f99(foreground_params[1], 1.0/red_waves)*foreground_params[0])
    ext_model_fluxes = extinguish_by_dust(params_best[3:10], 1.0/red_waves, model_fluxes)
    hi_ext_model_fluxes = hi_absorption(params_best[10:], velocities, red_waves, ext_model_fluxes)
    # useful for making the "observed" extinction curve
    model_fluxes_w_hi_abs = hi_absorption(params_best[10:], velocities, red_waves, model_fluxes)

    # need the model ave of the optical/nir photometry for normalization
    _model_ave = np.average(hi_ext_model_fluxes[0:n_bands])

    # split off the spectra data to plot as lines instead of points
    red_waves_spectra = red_waves[n_bands:]
    red_fluxes_spectra = red_fluxes[n_bands:]
    red_uncs_spectra = red_uncs[n_bands:]
    red_weights_spectra = red_weights[n_bands:]
    model_fluxes_spectra = model_fluxes[n_bands:]
    model_fluxes_spectra_w_hi_abs = model_fluxes_w_hi_abs[n_bands:]
    hi_ext_model_fluxes_spectra = hi_ext_model_fluxes[n_bands:]

    # sort to provide a clean plot
    sindxs = np.argsort(red_waves_spectra)
    gsindxs, = np.where(red_weights[sindxs] > 0.0)
    sindxs_good = sindxs[gsindxs]
    bsindxs, = np.where(red_weights[sindxs] == 0.0)
    sindxs_bad = sindxs[bsindxs]

    #fig = pyplot.figure()
    #ax = fig.add_subplot(2,1,1)
    fig, ax = pyplot.subplots(nrows=3,ncols=2, sharex=False, figsize=(20,13))

    # spectra
    ax[1,0].plot(1.0/red_waves[0:n_bands], red_fluxes[0:n_bands]/_data_ave, 'bo')
    ax[1,0].plot(1.0/red_waves_spectra[sindxs_good], red_fluxes_spectra[sindxs_good]/_data_ave, 'go')
    ax[1,0].plot(1.0/red_waves_spectra[sindxs_bad], red_fluxes_spectra[sindxs_bad]/_data_ave, 'ro')

    ax[1,0].plot(1.0/red_waves[0:n_bands], hi_ext_model_fluxes[0:n_bands]/_model_ave, 'ko')
    ax[1,0].plot(1.0/red_waves_spectra[sindxs], hi_ext_model_fluxes_spectra[sindxs]/_model_ave, 'k-')

    ax[1,0].plot(1.0/red_waves[0:n_bands], model_fluxes_w_hi_abs[0:n_bands]/_model_ave, 'co')
    ax[1,0].plot(1.0/red_waves_spectra[sindxs], model_fluxes_spectra_w_hi_abs[sindxs]/_model_ave, 'c-')

    ax[1,0].set_yscale('log')
    ax[1,0].set_ylim(1e-1,1.2*max(model_fluxes_spectra_w_hi_abs[sindxs_good]/_model_ave))
    ax[1,0].set_ylabel('F/ave(optir)')

    # residuals
    residuals_bands = ((red_fluxes[0:n_bands]/_data_ave) - (hi_ext_model_fluxes[0:n_bands]/_model_ave))/(hi_ext_model_fluxes[0:n_bands]/_model_ave)
    residuals_spectra = ((red_fluxes_spectra/_data_ave) - (hi_ext_model_fluxes_spectra/_model_ave))/(hi_ext_model_fluxes_spectra/_model_ave)
    ax[2,0].plot(1.0/red_waves_spectra[sindxs_good], residuals_spectra[sindxs_good], 'go')
    ax[2,0].plot(1.0/red_waves_spectra[sindxs_bad], residuals_spectra[sindxs_bad], 'ro')
    ax[2,0].errorbar(1.0/red_waves[0:n_bands], residuals_bands, yerr=red_uncs[0:n_bands]/(hi_ext_model_fluxes[0:n_bands]/_model_ave), fmt='bo', markersize=4)

    #ax[2].plot(1.0/red_waves, red_weights*0.2, 'ro')
    ax[2,0].set_ylabel('(Fd - Fm)/Fm')
    ax[2,0].set_ylim(-0.3,0.3)
    ax[2,0].set_xlabel('1/$\lambda$ [$\mu$m$^{-1}$]')

    # show blowups of the fit with residuals
    hrange = 1.0/np.array([5.0,3.0])
    hindxs, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]))
    hindxs_good, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]) & (red_weights_spectra > 0))
    ax[0,1].errorbar(1.0/red_waves_spectra[hindxs], red_fluxes_spectra[hindxs]/_data_ave, yerr=red_uncs_spectra[hindxs]/_data_ave, fmt='ro')
    ax[0,1].errorbar(1.0/red_waves_spectra[hindxs_good], red_fluxes_spectra[hindxs_good]/_data_ave, yerr=red_uncs_spectra[hindxs_good]/_data_ave, fmt='go')
    ax[0,1].plot(1.0/red_waves_spectra[hindxs], hi_ext_model_fluxes_spectra[hindxs]/_model_ave, 'k-')
    ax[0,1].plot(1.0/red_waves_spectra[hindxs], abs((red_fluxes_spectra[hindxs]/_data_ave) - (hi_ext_model_fluxes_spectra[hindxs]/_model_ave)), 'c-')
    ax[0,1].set_ylabel('F/optir_ave')
    ax[0,1].set_ylim(-0.1,1.2*max(red_fluxes_spectra[hindxs_good]/_data_ave))

    # show blowups of the fit with residuals
    hrange = 1.0/np.array([7.0,5.0])
    hindxs, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]))
    hindxs_good, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]) & (red_weights_spectra > 0))
    ax[1,1].errorbar(1.0/red_waves_spectra[hindxs], red_fluxes_spectra[hindxs]/_data_ave, yerr=red_uncs_spectra[hindxs]/_data_ave, fmt='ro')
    ax[1,1].errorbar(1.0/red_waves_spectra[hindxs_good], red_fluxes_spectra[hindxs_good]/_data_ave, yerr=red_uncs_spectra[hindxs_good]/_data_ave, fmt='go')
    ax[1,1].plot(1.0/red_waves_spectra[hindxs], hi_ext_model_fluxes_spectra[hindxs]/_model_ave, 'k-')
    ax[1,1].plot(1.0/red_waves_spectra[hindxs], abs((red_fluxes_spectra[hindxs]/_data_ave) - (hi_ext_model_fluxes_spectra[hindxs]/_model_ave)), 'c-')
    ax[1,1].set_ylabel('F/optir_ave')
    ax[1,1].set_ylim(-0.1,1.2*max(red_fluxes_spectra[hindxs_good]/_data_ave))

    # show blowups of the fit with residuals
    hrange = 1.0/np.array([9.0,7.0])
    hindxs, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]))
    hindxs_good, = np.where((red_waves_spectra > hrange[0]) & (red_waves_spectra < hrange[1]) & (red_weights_spectra > 0))
    ax[2,1].errorbar(1.0/red_waves_spectra[hindxs], red_fluxes_spectra[hindxs]/_data_ave, yerr=red_uncs_spectra[hindxs]/_data_ave, fmt='ro')
    ax[2,1].errorbar(1.0/red_waves_spectra[hindxs_good], red_fluxes_spectra[hindxs_good]/_data_ave, yerr=red_uncs_spectra[hindxs_good]/_data_ave, fmt='go')
    ax[2,1].plot(1.0/red_waves_spectra[hindxs], hi_ext_model_fluxes_spectra[hindxs]/_model_ave, 'k-')
    ax[2,1].plot(1.0/red_waves_spectra[hindxs], abs((red_fluxes_spectra[hindxs]/_data_ave) - (hi_ext_model_fluxes_spectra[hindxs]/_model_ave)), 'c-')
    ax[2,1].set_ylabel('F/optir_ave')
    ax[2,1].set_ylim(-0.1,1.2*max(red_fluxes_spectra[hindxs_good]/_data_ave))

    # extinction curve best fit extinction curve
    ext_data = ExtData()
    ext_data.calc_ext_elvebv(reddened_star, model_fluxes_w_hi_abs[0:n_bands], model_fluxes_w_hi_abs[n_bands:][sindxs_good], params_best[ndim])

    # get the uncertainties on the extinction curve
    ext_data.ext_uncs(samples)

    # save the extinction curve to a FITS file
    ext_data.save_ext_data(starname+'_best_ext.fits', params_best, params_50p_uncs)

    # extinction curve 50p fit extinction curve
    ext_data_50p = ExtData()
    ext_data_50p.calc_ext_elvebv(reddened_star, model_fluxes_w_hi_abs[0:n_bands], model_fluxes_w_hi_abs[n_bands:][sindxs_good], params_50p[ndim])

    # get the uncertainties on the extinction curve
    ext_data_50p.ext_uncs(samples)

    # save the extinction curve to a FITS file
    ext_data_50p.save_ext_data(starname+'_50p_ext.fits', params_50p, params_50p_uncs)

    ax[0,0].plot(ext_data.ext_x['STIS'], ext_data.ext_curve['STIS'], 'b-', label="STIS")
    ax[0,0].plot(ext_data.ext_x['BANDS'], ext_data.ext_curve['BANDS'], 'ro', label="Photometry")
    
    # define an x value array that spans the full wavelength range with good wavelength sampling
    x_npts = 101
    x_range = [0.3,10.0]
    x_vals = x_range[0] + (x_range[1] - x_range[0])*np.array(range(x_npts))/(x_npts-1)

    # plot of best fit model
    alav_50p = f99.f99(params_50p[4], x_vals, c2=params_50p[5], c3=params_50p[6], c4=params_50p[7], x0=params_50p[8], gamma=params_50p[9])
    elvebv_50p = (alav_50p - 1.0)*params_50p[4]
    ax[0,0].plot(x_vals, elvebv_50p, 'g--', label="p50 model")

    # plot of best fit model
    alav_best = f99.f99(params_best[4], x_vals, c2=params_best[5], c3=params_best[6], c4=params_best[7], x0=params_best[8], gamma=params_best[9])
    elvebv_best = (alav_best - 1.0)*params_best[4]
    ax[0,0].plot(x_vals, elvebv_best, 'c--', label='best fit model')

    #ax[0,0].set_ylim(1.2*min(elvebv_50p),1.2*max(elvebv_50p))
    ax[0,0].set_ylim(-5.0,ext_max_yval)
    spec_xlim = ax[1,0].get_xlim()
    ax[0,0].set_xlim(spec_xlim[0],spec_xlim[1])

    ax[0,0].set_ylabel('$E(\lambda - V)/E(B-V)$')
    ax[0,0].legend(loc=2)

    fig.savefig(starname+'_sed_data_p50_model.png')

    if args.walkers:
        # plot the walker chains for all parameters
        fig, ax = pyplot.subplots(ndim, sharex=True, figsize=(13,13))
        walk_val = np.arange(nsteps)
        for i in range(ndim):
            for k in range(nwalkers):
                ax[i].plot(walk_val,sampler.chain[k,:,i],'-')
                ax[i].set_ylabel(var_names[i])
        fig.savefig(starname+'_walker_param_values.png')

    if args.triangle:

        # plot the 1D and 2D likelihood functions in a traditional triangle plot
        fig = triangle.corner(samples, labels=var_names, show_titles=True, extents=param_limits)
        fig.savefig(starname+'_param_triangle.png')

    plot_time = time.clock()
    print('plot time taken: ',(plot_time - emcee_time)/60., ' min')
