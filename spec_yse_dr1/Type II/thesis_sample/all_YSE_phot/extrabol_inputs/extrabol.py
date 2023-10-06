#!/usr/bin/env python

import numpy as np
from astroquery.svo_fps import SvoFps
import matplotlib.pyplot as plt
import george
from scipy.optimize import minimize, curve_fit
import argparse
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import os
from astropy.table import Table
from astropy.io import ascii
import matplotlib.cm as cm
import sys
from scipy import interpolate as interp
from george.modeling import Model
import extinction
import emcee
import importlib_resources

# Define a few important constants that will be used later
epsilon = 0.001
c = 2.99792458E10  # cm / s
sigsb = 5.6704e-5  # erg / cm^2 / s / K^4
h = 6.62607E-27
ang_to_cm = 1e-8
k_B = 1.38064852E-16  # cm^2 * g / s^2 / K


def bbody(lam, T, R):
    '''
    Calculate BB L_lam (adapted from superbol, Nicholl, M. 2018, RNAAS)

    Parameters
    ----------
    lam : float
        Reference wavelengths in Angstroms
    T : float
        Temperature in Kelvin
    R : float
        Radius in cm

    Output
    ------
    L_lam in erg/s/cm
    '''

    lam_cm = np.array(lam) * ang_to_cm
    exponential = (h*c) / (lam_cm*k_B*T)
    blam = ((2.*np.pi*h*c**2) / (lam_cm**5)) / (np.exp(exponential)-1.)
    area = 4. * np.pi * R**2
    lum = blam * area

    return lum



def read_in_photometry(filename, dm, redshift, start, end, snr, mwebv,
                       use_wc, verbose, settings):
    '''
    Read in SN file

    Parameters
    ----------
    filename : string
        Input file name
    dm : float
        Distance modulus
    redshift : float
        redshift
    start : float
        The first time point to be accepted
    end : float
        The last time point to be accepted
    snr : float
        The lowest signal to noise ratio to be accepted
    mwebv : float
        Extinction to be removed
    use_wc : bool
        If True, use redshift corrected wv for extinction correction

    Output
    ------
    lc : numpy.array
        Light curve array
    wv_corr : float
        Mean of wavelengths, used in GP pre-processing
    flux_corr : float
        Flux correction, used in GP pre-processing
    my_filters : list
        List of filter names
    '''

    try:
        with open(settings, 'r') as f:
            lines = f.readlines()
        filter_mean_function = {i.split()[0]:(i.split()[1]) for i in lines}
        filters_in_settings = [i.split()[0] for i in lines]
    
    except FileNotFoundError:
        print("settings file not found, using 0 as mean function...")
        filter_mean_function = {} 
        filters_in_settings = [] 
        
    photometry_data = np.loadtxt(filename, dtype=str, skiprows=2)
    # Extract key information into seperate arrays
    phases = np.asarray(photometry_data[:, 0], dtype=float)
    errs = np.asarray(photometry_data[:, 2], dtype=float)
    if verbose:
        print('Getting Filter Data...')
    index = SvoFps.get_filter_index(wavelength_eff_min=100*u.angstrom,
                                    wavelength_eff_max=30000*u.angstrom,
                                    timeout=3600)
    filterIDs = np.asarray(index['filterID'].data, dtype=str)
    wavelengthEffs = np.asarray(index['WavelengthEff'].data, dtype=float)
    widthEffs = np.asarray(index['WidthEff'].data, dtype=float)
    zpts_all = np.asarray(index['ZeroPoint'].data, dtype=str)

    # Extract filter names and effective wavelengths
    wv_effs = []
    width_effs = []
    my_filters = []
    for ufilt in photometry_data[:, 3]:
        gind = np.where(filterIDs == ufilt)[0]
        if len(gind) == 0:
            sys.exit(f'Cannot find {str(ufilt)} in SVO.')
        wv_effs.append(wavelengthEffs[gind][0])
        width_effs.append(widthEffs[gind][0])
        my_filters.append(ufilt)
    wv_effs = np.asarray(wv_effs)

    # Convert brightness data to flux
    zpts = []
    fluxes = []
    for datapoint in photometry_data:
        mag = float(datapoint[1]) - dm
        if datapoint[-1] == 'AB':
            zpts.append(3631.00)
        else:
            gind = np.where(filterIDs == datapoint[3])
            zpts.append(float(zpts_all[gind[0]][0]))

        flux = 10.**(mag/-2.5) * zpts[-1] * (1.+redshift)

        # Convert Flux to log-flux space
        # This is easier on the Gaussian Process
        # 'fluxes' is also equivilant to the negative absolute magnitude
        flux = 2.5 * (np.log10(flux)-np.log10(3631.00))
        fluxes.append(flux) 
    

    # Remove extinction
    if use_wc:
        ext = extinction.fm07(wv_effs / (1.+redshift), mwebv)
    else:
        ext = extinction.fm07(wv_effs, mwebv)
    for i in np.arange(len(fluxes)):
        fluxes[i] = fluxes[i] + ext[i]

    # The GP prefers when values are relatively close to zero
    # so we adjust wavelength and flux accordingly
    # This will be undone after interpolation
    wv_corr = np.mean(wv_effs / (1.+redshift))
    flux_corr = np.min(fluxes) - 1.0
    wv_effs = wv_effs - wv_corr
    fluxes = np.asarray(fluxes) #- flux_corr

    # Eliminate any data points bellow threshold snr
    gis = []
    for i in np.arange(len(phases)):
        if (1/errs[i]) >= snr:
            gis.append(i)
    gis = np.asarray(gis, dtype=int)


    phases = phases[gis]
    fluxes = fluxes[gis]
    wv_effs = wv_effs[gis]
    errs = errs[gis]
    width_effs = np.asarray(width_effs)
    width_effs = width_effs[gis]
    my_filters = np.asarray(my_filters)
    my_filters = my_filters[gis] 
    print('unique filters:', np.unique(my_filters))
    
    if settings: 
        settings_filters = set(filters_in_settings) 
        my_filters_unique = set(np.unique(my_filters))
        if settings_filters != my_filters_unique:
            print('settings filters:', settings_filters)
            print('dataset filters:', my_filters_unique)
            sys.exit("Settings file filters do not match filters in dataset")

    # Set the peak flux to t=0
    peak_i = np.argmax(fluxes)
    if verbose:
        print('Peak Luminosity occurrs at MJD:', phases[peak_i])
    phases = np.asarray(phases) - phases[peak_i]

    # Eliminate any data points outside of specified range
    # With respect to first data point
    gis = []
    for i in np.arange(len(phases)):
        if phases[i] <= end and phases[i] >= start:
            gis.append(i)
    gis = np.asarray(gis, dtype=int)

    phases = phases[gis]
    fluxes = fluxes[gis]
    wv_effs = wv_effs[gis]
    errs = errs[gis]
    width_effs = width_effs[gis]
    my_filters = my_filters[gis] 
    
    #map filter names to effective wavelengths 
    wv_eff_in_angstroms = wv_effs + wv_corr 
    wv, idx = np.unique(wv_eff_in_angstroms, return_index = True)
    wv_idx = wv[np.argsort(idx)]
    name, ind = np.unique(my_filters, return_index = True)
    name_ind = name[np.argsort(ind)]
    filter_name_to_effwv = dict(zip(name_ind, wv_idx))
    print('filter to effwv:', filter_name_to_effwv)
    
    #keep track of which filters use template and 0 as mean function
    linear_filters = [] 
    cubic_filters = []
    template_filters = [] 
    zero_filters = [] 
    for key, value in filter_mean_function.items():        
        if value == 'linear':
            linear_filters.append(key)
        elif value == 'cubic':
            cubic_filters.append(key)
        elif value == 'template':
            template_filters.append(key)
        else:
            zero_filters.append(key)

    lc = np.vstack((phases, fluxes, wv_effs / 1000., errs, width_effs, my_filters))

    return (lc, wv_corr, flux_corr, my_filters, filter_mean_function, filter_name_to_effwv, 
linear_filters, cubic_filters, template_filters)
    


def chi_square(dat, model, uncertainty):
    '''
    Calculate the chi squared of a model given a set of data

    Parameters
    ----------
    dat : numpy.array
        Experimental data for the model to be tested against
    model : numpy.array
        Model data being tested
    uncertainty : numpy.array
        Error on experimental data

    Output
    ------
    chi2 : float
        the chi sqaured value of the model
    '''

    chi2 = 0.
    for i in np.arange(len(dat)):
        chi2 += ((model[i]-dat[i]) / uncertainty[i])**2.

    return chi2

def generate_template(filter_wv, sn_type):
    '''
    Prepare and interpolate SN1a Template

    Parameters
    ----------
    fiter_wv : numpy.array
        effective wavelength of filters in Angstroms
    sn_type : string
        The type of supernova template to be used for GP mean function

    Output
    ------
    temp_interped : RectBivariateSpline object
        interpolated template
    '''



    my_template_file = importlib_resources.files('extrabol.template_bank') / ('smoothed_sn' + sn_type + '.npz')
    template = np.load(my_template_file)
    temp_times = template['time']
    temp_wavelength = template['wavelength']
    temp_f_lambda = template['f_lambda']

    # The template is too large, so we thin it out
    # First chop off unnecessary ends
    gis = []
    for i in np.arange(len(temp_wavelength)):
        if temp_wavelength[i] < np.amax(filter_wv) and\
                temp_wavelength[i] > np.amin(filter_wv):
            gis.append(i)
    temp_times = temp_times[gis]
    temp_wavelength = temp_wavelength[gis]
    temp_f_lambda = temp_f_lambda[gis]

    # Remove every other time point
    gis = []
    for i in np.arange(len(temp_times)):
        if temp_times[i] % 2. == 0:
            gis.append(i)
    temp_times = temp_times[gis]
    temp_wavelength = temp_wavelength[gis]
    temp_f_lambda = temp_f_lambda[gis]

    # Remove every other wavelength point
    gis = []
    for i in np.arange(len(temp_wavelength)):
        if temp_wavelength[i] % 20. == 0:
            gis.append(i)
    temp_times = temp_times[gis]
    temp_wavelength = temp_wavelength[gis]
    temp_f_lambda = temp_f_lambda[gis]

    # Remove initial rise
    # If the data point is very dim, it likely has a low snr
    gis = []
    for i in np.arange(len(temp_times)):
        if temp_times[i] >= 1.:
            gis.append(i)
    temp_times = temp_times[gis]
    temp_wavelength = temp_wavelength[gis]
    temp_f_lambda = temp_f_lambda[gis]

    # Set peak flux to t=0
    peak_i = np.argmax(temp_f_lambda)
    temp_times = np.asarray(temp_times) - temp_times[peak_i]

    # RectBivariateSpline requires
    # that x and y are 1-d arrays, strictly ascending
    temp_times_u = np.unique(temp_times)
    temp_wavelength_u = np.unique(temp_wavelength)
    temp_f_lambda_u = np.zeros((len(temp_times_u), len(temp_wavelength_u)))
    for i in np.arange(len(temp_times_u)):
        gis = np.where(temp_times == temp_times_u[i])
        temp_f_lambda_u[i, :] = temp_f_lambda[gis]
    # Template needs to be converted to log(flux) to match data
    for i in np.arange(len(temp_wavelength_u)):
        wv = temp_wavelength_u[i]
        temp_f_lambda_u[:, i] = 2.5 * np.log10((wv**2) * temp_f_lambda_u[:, i])

    temp_interped = interp.RectBivariateSpline(temp_times_u, temp_wavelength_u,
                                               temp_f_lambda_u)

    return temp_interped


def fit_template(wv, template_to_fit, filts, wv_corr, flux, time,
                 errs, z, output_chi=False, output_params=True):
    '''
    Get parameters to roughly fit template to data

    Parameters
    ----------
    wv : numpy.array
        wavelenght of filters in angstroms
    template_to_fit : RectBivariateSpline object
        interpolated template
    filts : numpy.array
        normalized wavelength values for each obseration
    wv_corr : float
        Mean of wavelengths, used in GP pre-processing
    flux : numpy.array
        flux data from observations
    time : numpy.array
        time data from observations
    errs : numpy.array
        errors on flux data
    z : float
        redshift
    output_chi : bool
        If true, function returns chi squared value
    output_params : bool
        If true, function returns optimal parameters

    Output
    ------
    A_opt : float
        multiplicative constant to be applied to template flux values
    t_c_opt : float
        additive constant to line up template and data in time
    t_s_opt : float
        multiplicative constant to scale the template in the time dimention
    chi2 : float
        The chi squared value for the given parameters
    '''

    A_opt = []
    t_c_opt = []
    t_s_opt = []
    chi2 = []

    # Fit the template to the data for each filter used
    # and choose the set of parameters with the lowest total chi2
    for wavelength in wv:
        # A callable function to test chi2 later on
        def model(time, filt, A, t_c, t_s):
            time_sorted = sorted(time)
            time_corr = np.asarray(time_sorted) * 1./t_s + t_c
            mag = template_to_fit(time_corr, filt) + A  # log(flux), not mag
            mag = np.ndarray.flatten(mag)
            return mag

        # curve_fit won't know what to do with the filt param
        # so I need to modify it slightly
        def curve_to_fit(time, A, t_c, t_s):
            mag = model(time, wavelength, A, t_c, t_s)
            return mag

        # Collect the data points coresponding to the current wavelength
        gis = np.where(filts*1000 + wv_corr == wavelength)
        dat_fluxes = flux[gis]
        dat_times = time[gis]
        dat_errs = errs[gis]
        popt, pcov = curve_fit(curve_to_fit, dat_times, dat_fluxes,
                               p0=[20, 0, 1+z], maxfev=8000,
                               bounds=([-np.inf, -np.inf, 0], np.inf))
        A_opt.append(popt[0])
        t_c_opt.append(popt[1])
        t_s_opt.append(popt[2])

        # Test chi2 for this set of parameters over all filters
        param_chi = 0
        for filt in wv:
            m = model(dat_times, filt, popt[0], popt[1], popt[2])
            param_chi += chi_square(dat_fluxes, m, dat_errs)
        chi2.append(param_chi)

    # Choose the template with the minimum chi2
    gi = np.argmin(chi2)
    chi2 = chi2[gi]
    A_opt = A_opt[gi]
    t_c_opt = t_c_opt[gi]
    t_s_opt = t_s_opt[gi]

    if output_chi:
        if output_params:
            return A_opt, t_c_opt, t_s_opt, chi2
        else:
            return chi2
    else:
        if not output_params:
            return 0
        else:
            return A_opt, t_c_opt, t_s_opt


def test(lc, wv_corr, z):
    '''
    Test every available template for the lowest possible chi^2

    Parameters
    ----------
    lc : numpy.array
        LC array
    wv_corr : float
        mean of wavelength, needed to find wavelength in angstroms
    z : float
        redshift

    Output
    ------
    best_temp : string
        The name of the template with the lowest chi squared value
    '''
    lc = lc.T

    # Extract necissary information from lc
    time = lc[:, 0]
    flux = lc[:, 1]
    filts = lc[:, 2]
    errs = lc[:, 3]
    ufilts = np.unique(lc[:, 2])
    ufilts_in_angstrom = ufilts*1000.0 + wv_corr

    # Generate a template for each available supernova type
    # Then fit and test each one for lowest possible chi2
    templates = ['1a', '1bc', '2p', '2l']
    chi2 = []
    for template in templates:
        template_to_fit = generate_template(ufilts_in_angstrom, template)
        chi2.append(fit_template(ufilts_in_angstrom, template_to_fit, filts,
                    wv_corr, flux, time, errs, z, output_chi=True,
                    output_params=False))

    # Chooses the template that yields the lowest chi2
    gi = np.argmin(chi2)
    chi2 = chi2[gi]
    best_temp = templates[gi]

    return best_temp


def interpolate(lc, wv_corr, sn_type, use_mean, z, verbose, filter_mean_function, 
                filter_name_to_effwv, linear_filters = None, cubic_filters = None, 
                template_filters = None):
    '''
    Interpolate the LC using a 2D Gaussian Process (GP)

    Parameters
    ----------
    lc : numpy.array
        LC array
    wv_corr : float
        mean of wavelengths, needed to find wavelenght in angstroms
    sn_type : string
        type of supernova template being used for GP mean function
    use_mean : bool
        If True, use a template for GP mean function
        If False, use 0 as GP mean function
    z : float
        redshift

    Output
    ------
    dense_lc : numpy.array
        GP-interpolated LC and errors
    test_y : numpy.array
        A set of flux and wavelength values from the template, to be plotted
    test_times : numpy.array
        A set of time values to plot the template data against
    '''

    lc = lc.T

    times = lc[: , 0].astype('float64')
    fluxes = lc[: , 1].astype('float64')
    flux_corr = np.min(fluxes) - 1 
    wv_effs = lc[: , 2].astype('float64')
    errs = lc[: , 3].astype('float64') 
    filters = lc[: , 5 ] 
    unique_filter_names = np.unique(filters) 
    
    stacked_data = np.vstack([times, wv_effs]).T
    ufilts = np.unique(wv_effs)
    ufilts_in_angstrom = ufilts*1000.0 + wv_corr
    nfilts = len(ufilts) 
    min_time = np.min(times)
    print('min time:', min_time)
    max_time = int(np.ceil(np.max(times)))
    length_of_times = len(np.arange(np.min(times), np.max(times)))
   
    print('length of times:', length_of_times)
    print(length_of_times)
    x_pred = np.zeros((length_of_times, 2)) 
    print("len x_pred:", len(x_pred))
    dense_fluxes = np.zeros((length_of_times, nfilts)) 
    print('len dense_fluxes', len(dense_fluxes))
    dense_errs = np.zeros((length_of_times, nfilts))

    # test_y is only used if mean = True
    # but I still need it to exist either way
    test_y = []
    test_times = [] 
    linear_results = {} 
    dense_lc_list = [] 
    # Set up gp 
    # kernel = np.var(fluxes) \
    #     * george.kernels.Matern32Kernel([12, 0.1], ndim=2) 
    key_count = 0 
    if not use_mean:
        mean = 0 
        gp = george.GP(kernel, mean = 0)
    else:
        #create linear splines, set up and run GP 
        for i, filt in enumerate(linear_filters):
            if verbose:
                print(f'Using linear spline for {filt}')
                
            idx = np.where(filters == filt)
            x = times[idx]
            y = fluxes[idx]
            yerr = errs[idx] 
            central_wv = wv_effs[idx] * 1000 + wv_corr 
            print('central wv:', central_wv[0])
            kernel = np.var(y) * george.kernels.Matern32Kernel(1e6)
            spline = interp.interp1d(x, y, kind = 'linear')
            class snModel_linear:
                def __init__(self, spline):
                    self.spline = spline
                def get_value(self, x):
                    return self.spline(x)
            mean = snModel_linear(spline)
            gp = george.GP(mean = mean, kernel = kernel)
            gp.compute(x, yerr)
            x_pred = np.arange(np.min(x), np.max(x)) 
            pred, pred_var = gp.predict(y, x_pred, return_var = True) 
            #store interpolation for blackbody fitting and plotting later 
            dense_fluxes = pred 
            dense_errs = np.sqrt(pred_var) 
            central_wv_array = np.repeat(central_wv[0], len(x_pred))
            dense_lc_filtered = np.vstack((x_pred, dense_fluxes, dense_errs, central_wv_array)) 
               
            print('vstack shape:', dense_lc_filtered.shape)
            linear_results[filt] = { 
                'dense_lc_filtered':dense_lc_filtered
            }
                    
        # extract dense_lc for each filter
        for filter in linear_results.values():
            dense_lc_individual = filter.get('dense_lc_filtered')
            dense_lc_list.append(dense_lc_individual)
        print('dense_lc_list len:', len(dense_lc_list))
        # concatenate into one object 
        dense_lc = np.hstack(dense_lc_list)
        #print('dense_lc:', dense_lc)
        print('dense_lc shape:', dense_lc.shape)    
        
        #create cubic splines, set up and run GP         
        for i, filt in enumerate(cubic_filters):
            idx = np.where(filters == filt)
            x = times[idx]
            y = fluxes[idx]
            yerr = errs[idx] 
            sorted_idx = np.argsort(x)
            sorted_time = x[sorted_idx]
            sorted_mag = y[sorted_idx]
            kernel = np.var(y) * george.kernels.Matern32kKernel(1e6)
            cubic_spline = interp.Cubic_spline(sorted_time, sorted_mag)
            class snModel_cubic:
                def __init__(self, cubic_spline):
                    self.cubic_spline = cubic_spline
                def get_value(self, x):
                    return self.cubic_spline(x)
            mean = snModel_cubic(cubic_spline)
            kernel = np.var(y) * george.kernels.Matern32Kernel(1e6)
            gp = george.GP(mean = mean, kernel = kernel)
            gp.compute(x, yerr)
        
        
        #fit templates to data 
        for i, filt in enumerate(template_filters): 
            template = generate_template(ufilts_in_angstrom, sn_type)
            if verbose:
                print(f'Fitting Template for {filt}')
            idx = np.where(filters == filt)
            x = times[idx]
            y = fluxes[idx]
            yerr = errs[idx] 
            kernel = np.var(fluxes) * george.kernels.Matern32Kernel([12, 0.1], ndim=2) 
            f_stretch, t_shift, t_stretch = fit_template(ufilts_in_angstrom,
                                                         template, wv_effs, 
                                                         wv_corr, fluxes, times,
                                                         errs, z) 
            # Geroge needs the mean function to be in this format 
            class snModel(Model):
                def get_value(self, param):
                    t = (param[:,0] * 1./t_stretch) + t_shift 
                    wv = param[:,1]
                    return np.asarray([template(*p)] for p in zip(t, wv)) 
                + f_stretch
            # Get Test data so that the template can be plotted 
            mean = snModel()
            for i in ufilts_in_angstrom:
                test_wv = np.full((1, length_of_times), i)
                test_times = np.arange(int(np.floor(np.min(times))), int(np.ceil(np.max(times))+1))
                test_x = np.vstack((test_times, test_wv)).T 
                test_y.append(mean.get_value(test_x)) 
            test_y = np.asarray(test_y)
            gp = george.GP(kernel = kernel, mean = mean) 
            
            def neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.loglikelihood(fluxes)
            def grad_neg_ln_like(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(fluxes)
            # Optimize gp paramters 
            bnds = ((None, None), (None, None), (None, None))
            result = minimize(neg_ln_like, 
                              gp.get_parameter_vector(),
                              jac = grad_neg_ln_like,
                              bounds = bnds)
            gp.set_parameter_vector(result.x)
            #Populate arrays with time and wavelength values to be fed into gp 
        #     print('arange length of times:', len(np.arange(min_time, max_time)))
        #     print('arange array:', np.arange(min_time, max_time))
        #     x_pred[:, 0] = np.arange(np.min(times), np.max(times)) 
        #     x_pred[:, 1] = ufilts[key_count]
            
        #     # Run gp to estimate interpolation 
        #     pred, pred_var = gp.predict(this_filter_fluxes, x_pred, return_var = True)
            
            # Populate dense_lc with newly gp-predicted values 
        #     gind = np.where(np.abs(x_pred[:,1] - ufilts[key_count]) < epsilon)[0]
        #     dense_fluxes[:, key_count] = pred[gind]
        #     dense_errs[:, key_count] = np.sqrt(pred_var[gind])
        #     key_count += 1 
        # dense_lc = np.dstack((dense_fluxes, dense_errs))
        print('dense_lc:', dense_lc)
                                        
    return dense_lc, test_y, test_times, ufilts, ufilts_in_angstrom


def fit_bb(dense_lc, wvs, use_mcmc, T_max):
    '''
    Fit a series of BBs to the GP LC
    Adapted from superbol, Nicholl, M. 2018, RNAAS)

    Parameters
    ----------
    dense_lc : numpy.array
        GP-interpolated LC
    wvs : numpy.array
        Reference wavelengths in Ang

    Output
    ------
    T_arr : numpy.array
        BB temperature array (K)
    R_arr : numpy.array
        BB radius array (cm)
    Terr_arr : numpy.array
        BB radius error array (K)
    Rerr_arr : numpy.array
        BB temperature error array (cm)
    '''
    print('dense_lc[1]:', dense_lc.shape[1])
    print('wvs', wvs)
    print('len dense_lc:', len(dense_lc))
    T_arr = np.zeros(dense_lc.shape[1])
    print('len Tarr:', len(T_arr))
    R_arr = np.zeros(dense_lc.shape[1])
    Terr_arr = np.zeros(dense_lc.shape[1])
    Rerr_arr = np.zeros(dense_lc.shape[1])
    covar_arr = np.zeros(dense_lc.shape[1])

    prior_fit = (9000, 1e15)
    full_wv = [] 
    full_flux = [] 
    full_err = [] 
    
    #sort by time so blackbody fit is possible 
    idx = np.argsort(dense_lc[0,:])
    dense_lc = dense_lc[:, idx]
    
    
    for i in range(dense_lc.shape[1]):
        datapoint = dense_lc[:,i] 
        x_time = datapoint[0]
        flux = datapoint[1]
        fluxerr = datapoint[2]
        wavelength = datapoint[3]
        full_wv.append(wavelength)
        fnu = 10.**((-flux +48.6) / -2.5) 
        ferr = fluxerr
        fnu = fnu * 4. * np.pi * (3.086e19)**2
        fnu_err = np.abs(0.921034 * 10.**(0.4*flux - 19.44)) \
            * ferr * 4. * np.pi * (3.086e19)**2
        flam = fnu * c / (wavelength * ang_to_cm) ** 2 
        full_flux.append(flam)
        flam_err = fnu_err * c / (wavelength * ang_to_cm) ** 2 
        full_err.append(flam_err)
        
        print('wavelength:', wavelength)
        print('flam:', flam)
        print('flam_err:', flam_err)

        if use_mcmc:
            def log_likelihood(params, lam, f, f_err):
                T, R = params
                model = bbody(lam, T, R)
                return -np.sum((f-model)**2/(f_err**2))

            def log_prior(params):
                T, R = params
                if T > 0 and T < T_max and R > 0:
                    return 0.
                return -np.inf

            def log_probability(params, lam, f, f_err):
                lp = log_prior(params)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(params, lam, f, f_err)

            nwalkers = 16
            ndim = 2
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            args=[wavelength, flam, flam_err])
            T0 = 9000 + 1000*np.random.rand(nwalkers)
            R0 = 1e15 + 1e14*np.random.rand(nwalkers)
            p0 = np.vstack([T0, R0])
            p0 = p0.T
            burn_in_state = sampler.run_mcmc(p0, 100)
            sampler.reset()
            sampler.run_mcmc(burn_in_state, 4000)
            flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)
            covar_arr[i] = np.cov(flat_samples.T)[0,1]
            T_arr[i] = np.median(flat_samples[:, 0])
            R_arr[i] = np.median(flat_samples[:, 1])
            Terr_arr[i] = (np.percentile(flat_samples[:, 0], 84) -
                           np.percentile(flat_samples[:, 0], 16)) / 2.
            Rerr_arr[i] = (np.percentile(flat_samples[:, 1], 84) -
                           np.percentile(flat_samples[:, 1], 16)) / 2.

        else:

            try:
                print('flam:', flam)
                print('flam_err:', flam_err)
                print('wavelength:', wavelength)
                BBparams, covar = curve_fit(bbody, full_wv, full_flux, maxfev=10000,
                                            p0=prior_fit, sigma=full_err,
                                            bounds=(0, [T_max, np.inf])) 
                
                fit_curve = bbody(full_wv, *BBparams)
                
                plt.figure()
                plt.plot(full_wv, full_flux, color = 'k')
                plt.plot(full_wv, fit_curve, label = 'fit')
                plt.show()

                # Get temperature and radius, with errors, from fit
                T_arr[i] = BBparams[0]
                Terr_arr[i] = np.sqrt(np.diag(covar))[0]
                R_arr[i] = np.abs(BBparams[1])
                Rerr_arr[i] = np.sqrt(np.diag(covar))[1]
                covar_arr[i] = covar[0,1]
                prior_fit = BBparams
            except RuntimeWarning:
                T_arr[i] = np.nan
                R_arr[i] = np.nan
                Terr_arr[i] = np.nan
                Rerr_arr[i] = np.nan
                covar_arr[i] = np.nan
                 
                
        print('T_arr:', T_arr)
        np.savetxt('flux', full_flux)
        np.savetxt('wl', full_wv)
        np.savetxt('err', full_err)
    return T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr


def plot_gp(lc, dense_lc, snname, flux_corr, my_filters, wvs, test_data,
            outdir, sn_type, test_times, mean, show_template, filter_name_to_effwv, linear_filters = None, 
            cubic_filters = None):
    '''
    Plot the GP-interpolate LC and save

    Parameters
    ----------
    lc : numpy.array
        Original LC data
    dense_lc : numpy.array
        GP-interpolated LC
    snname : string
        SN Name
    flux_corr : float
        Flux correction factor for GP
    my_filters : list
        List of filters
    wvs : numpy.array
        List of central wavelengths, for colors
    outdir : string
        Output directory
    sn_type : string
        Type of sn template used for GP mean function
    test_times : numpy array
        Time values for sn template to be plotted against
    mean : bool
        Whether or not a non-zero mean function is being used in GP
    show_template : bool
        Whether or not the sn template is plotted

    Output
    ------
    '''
    #dictionary to assign filters a color 
    filter_colors = {}
    
    #plot interpolation + errors 
    plt.figure()
    for i, wavelength in enumerate(wvs):
        cmap = plt.get_cmap('tab10')
        color = cmap(i)
        idx = np.where(dense_lc[3] == wavelength)[0]
        x_pred = dense_lc[0,idx] 
        pred = dense_lc[1, idx]
        pred_var = dense_lc[2, idx]
        filter_colors[wavelength] = color
        plt.plot(x_pred, -pred, color = color, lw = 1.5, alpha = 0.5)
        plt.fill_between(x_pred, -pred - np.sqrt(pred_var), -pred + np.sqrt(pred_var), color = 'k', alpha = 0.2)
        

    #plot original data + errorbars 
    for i, filt in enumerate(linear_filters):
        central_wv = filter_name_to_effwv[filt]
        color = filter_colors[central_wv]
        idx = np.where(lc[:,5] == filt)
        x = lc[:,0].astype('float64')[idx]
        y = -lc[:,1].astype('float64')[idx]
        yerr = lc[:,3].astype('float64')[idx]
        plt.errorbar(x, y, yerr = yerr, fmt = '.', capsize = 0, color = color, label = filt.split('/')[-1])

    if mean:
        plt.title(snname + ' using sn' + sn_type)
    else:
        plt.title(snname + ' Light Curves')
    plt.legend()
    plt.xlabel('Time(days)')
    plt.ylabel('Absolute Magnitudes')
    plt.gca().invert_yaxis()
    plt.savefig(outdir + snname + '_' + str(sn_type) + '_gp.png')
    plt.clf()

    return 1


def plot_bb_ev(lc, dense_lc, Tarr, Rarr, Terr_arr, Rerr_arr, snname, outdir, sn_type):
    '''
    Plot the BB temperature and radius as a function of time

    Parameters
    ----------
    lc : numpy.array
        Original LC data
    T_arr : numpy.array
        BB temperature array (K)
    R_arr : numpy.array
        BB radius array (cm)
    Terr_arr : numpy.array
        BB radius error array (K)
    Rerr_arr : numpy.array
        BB temperature error array (cm)
    snname : string
        SN Name
    outdir : string
        Output directory
    sn_type : string
        The type of sn template used for the gp

    Output
    ------
    '''
    #min_time = np.min(lc[:,0].astype('float64'))
    #max_time = np.max(lc[:,0].astype('float64')) 
    #plot_times = dense_lc[0]
    #plot_times = np.arange(min_time, max_time)
    
    times = dense_lc[0]
    sorted_idx = np.argsort(times)
    sorted_time = times[sorted_idx]
    plot_times = sorted_time

    print('len plot times:', len(plot_times))
    
    fig, axarr = plt.subplots(2, 1, sharex=True)

    axarr[0].plot(plot_times, Tarr / 1.e3, color='k')
    axarr[0].fill_between(plot_times, Tarr/1.e3 - Terr_arr/1.e3,
                          Tarr/1.e3 + Terr_arr/1.e3, color='k', alpha=0.2)
    axarr[0].set_ylabel('Temp. (1000 K)')

    axarr[1].plot(plot_times, Rarr / 1e15, color='k')
    axarr[1].fill_between(plot_times, Rarr/1e15 - Rerr_arr/1e15,
                          Rarr/1e15 + Rerr_arr/1e15, color='k', alpha=0.2)
    axarr[1].set_ylabel(r'Radius ($10^{15}$ cm)')

    axarr[1].set_xlabel('Time (Days)')
    axarr[0].set_title(snname + ' Black Body Evolution')

    plt.savefig(outdir + snname + '_' + str(sn_type) + '_bb_ev.png')
    plt.clf()

    return 1


def plot_bb_bol(lc, dense_lc, bol_lum, bol_err, snname, outdir, sn_type):
    '''
    Plot the BB bolometric luminosity as a function of time

    Parameters
    ----------
    lc : numpy.array
        Original LC data
    bol_lum : numpy.array
        BB bolometric luminosity (erg/s)
    bol_err : numpy.array
        BB bolometric luminosity error (erg/s)
    snname : string
        SN Name
    outdir : string
        Output directory
    sn_type : string
        The type of sn template used in the gp

    Output
    ------
    '''
    #min_time = np.min(lc[:,0].astype('float64'))
    #max_time = np.max(lc[:,0].astype('float64'))
    # time_length = len(bol_lum)
    #plot_times = np.arange(min_time, max_time)
    times = dense_lc[0]
    sorted_idx = np.argsort(times)
    sorted_time = times[sorted_idx]
    plot_times = sorted_time
    
    plt.plot(plot_times, bol_lum, 'k')
    plt.fill_between(plot_times, bol_lum-bol_err, bol_lum+bol_err,
                     color='k', alpha=0.2)

    plt.title(snname + ' Bolometric Luminosity')
    plt.xlabel('Time (Days)')
    plt.ylabel('Bolometric Luminosity')
    plt.yscale('log')
    plt.savefig(outdir + snname + '_' + str(sn_type) + '_bb_bol.png')
    plt.clf()

    return 1


def write_output(lc, dense_lc, Tarr, Terr_arr, Rarr, Rerr_arr,
                 bol_lum, bol_err, my_filters,
                 snname, outdir, sn_type):
    '''
    Write out the interpolated LC and BB information

    Parameters
    ----------
    lc : numpy.array
        Initial light curve
    dense_lc : numpy.array
        GP-interpolated LC
    T_arr : numpy.array
        BB temperature array (K)
    Terr_arr : numpy.array
        BB radius error array (K)
    R_arr : numpy.array
        BB radius array (cm)
    Rerr_arr : numpy.array
        BB temperature error array (cm)
    bol_lum : numpy.array
        BB luminosity (erg/s)
    bol_err : numpy.array
        BB luminosity error (erg/s)
    my_filters : list
        List of filter names
    snname : string
        SN Name
    outdir : string
        Output directory
    sn_type : string
        Type of sn template used for the gp

    Output
    ------
    '''

    min_time = np.min(lc[:,0].astype('float64'))
    max_time = np.max(lc[:,0].astype('float64'))
    times = np.arange(min_time, max_time)
    dense_lc = np.reshape(dense_lc, (len(dense_lc), -1))
    dense_lc = np.hstack((np.reshape(-times, (len(times), 1)), dense_lc))
    tabledata = np.stack((Tarr / 1e3, Terr_arr / 1e3, Rarr / 1e15,
                          Rerr_arr / 1e15, np.log10(bol_lum),
                          np.log10(bol_err))).T
    tabledata = np.hstack((-dense_lc, tabledata)).T

    ufilts = np.unique(my_filters)
    table_header = []
    table_header.append('Time (MJD)')
    for filt in ufilts:
        table_header.append(filt)
        table_header.append(filt + '_err')
    table_header.extend(['Temp./1e3 (K)', 'Temp. Err.',
                         'Radius/1e15 (cm)', 'Radius Err.',
                         'Log10(Bol. Lum)', 'Log10(Bol. Err)'])
    table = Table([*tabledata],
                  names=table_header,
                  meta={'name': 'first table'})

    format_dict = {head: '%0.3f' for head in table_header}
    ascii.write(table, outdir + snname + '_' + str(sn_type) + '.txt',
                formats=format_dict, overwrite=True)

    return 1


def main():
    default_data = importlib_resources.files('extrabol.example') / 'SN2010bc.dat'
    default_data = str(default_data)
    # Define all arguments
    parser = argparse.ArgumentParser(description='extrabol helpers')
    parser.add_argument('snfile', nargs='?',
                        default=default_data,
                        type=str, help='Give name of SN file')
    parser.add_argument('-m', '--mean', dest='mean', type=str, default='0',
                        help="Template function for gp.\
                                Choose \'1a\',\'1bc\', \'2l\', \'2p\', or \
                                \'0\' for no template")
    parser.add_argument('-t', '--show_template', dest='template',
                        action='store_true',
                        help="Shows template function on plots")
    parser.add_argument('-d', '--dist', dest='distance', type=float,
                        help='Object luminosity distance', default=1e-5)
    parser.add_argument('-z', '--redshift', dest='redshift', type=float,
                        help='Object redshift', default=-1.)
    # Redshift can't =-1
    # this is simply a flag to be replaced later
    parser.add_argument('-dm', dest='dm', type=float, default=0,
                        help='Object distance modulus')
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--plot", help="Make plots", dest='plot',
                        action="store_true", default=True)
    parser.add_argument("--outdir", help="Output directory", dest='outdir',
                        type=str, default='./products/')
    parser.add_argument("--ebv", help="MWebv", dest='ebv',
                        type=float, default=-1.)
    # Ebv won't =-1
    # this is another flag to be replaced later
    parser.add_argument("--hostebv", help="Host B-V", dest='hostebv',
                        type=float, default=0.0)
    parser.add_argument('-s', '--start',
                        help='The time of the earliest \
                            data point to be accepted',
                        type=float, default=-50)
    parser.add_argument('-e', '--end',
                        help='The time of the latest \
                            data point to be accepted',
                        type=float, default=200)
    parser.add_argument('-snr',
                        help='The minimum signal to \
                            noise ratio to be accepted',
                        type=float, default=4)
    parser.add_argument('-wc', '--wvcorr',
                        help='Use the redshift-corrected \
                            wavelenghts for extinction calculations',
                        action="store_true")
    parser.add_argument('-mc', '--use_mcmc', dest='mc',
                        help='Use a Markov Chain Monte Carlo \
                              to fit BBs instead of curve_fit. This will \
                              take longer, but gives better error estimates',
                        default=False, action="store_true")
    parser.add_argument('--T_max', dest='T_max',  help='Temperature prior \
                                                        for black body fits',
                        type=float, default=40000.) 
    parser.add_argument('--settings', dest = 'settings', type=str, default ='settings.txt', help = 'Settings file name')

    args = parser.parse_args()

    # We need to know if an sn template is being used for gp
    sn_type = args.mean
    try:
        sn_type = int(sn_type)
        mean = False
        if sn_type != 0:
            print('Template request not valid. Assuming mean function of 0.')
    except ValueError:
        sn_type = sn_type
        mean = True

    # If redshift or ebv aren't specified by the user,
    # we read them in from the file here
    if args.redshift == -1 or args.ebv == -1:
        # Read in redshift and ebv and replace values if not specified
        f = open(args.snfile, 'r')
        if args.redshift == -1:
            args.redshift = float(f.readline())
            if args.ebv == -1:
                args.ebv = float(f.readline())
        if args.ebv == -1:
            args.ebv = float(f.readline())
            args.ebv = float(f.readline())
        f.close

    # Solve for redshift, distance, and/or dm if possible
    # if not, assume that data is already in absolute magnitudes
    if args.redshift != 0 or args.distance != 1e-5 or args.dm != 0:
        if args.redshift != 0:
            args.distance = cosmo.luminosity_distance(args.redshift).value
            args.dm = cosmo.distmod(args.redshift).value
        elif args.distance != 1e-5:
            args.redshift = z_at_value(cosmo.luminosity_distance, distance
                                       * u.Mpc)
            dm = cosmo.distmod(args.redshift).value
        else:
            args.redshift = z_at_value(cosmo.distmod, dm * u.mag)
            distance = cosmo.luminosity_distance(args.redshift).value
    elif args.verbose:
        print('Assuming absolute magnitudes.')

    # Make sure outdir name is formatted correctly
    if args.outdir[-1] != '/':
        args.outdir += '/'

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    snname = ('.').join(args.snfile.split('.')[: -1]).split('/')[-1]

    lc, wv_corr, flux_corr, my_filters, filter_mean_function, filter_name_to_effwv, linear_filters, cubic_filters, template_filters = read_in_photometry(args.snfile,
                                                            args.dm,
                                                            args.redshift,
                                                            args.start,
                                                            args.end, args.snr,
                                                            args.ebv,
                                                            args.wvcorr,
                                                            args.verbose, 
                                                            args.settings)

    # Test which template fits the data best
    if sn_type == 'test':
        sn_type = test(lc, wv_corr, args.redshift)
    if args.verbose:
        print('Using ' + str(sn_type) + ' template.')

    dense_lc, test_data, test_times, ufilts, ufilts_in_angstrom = interpolate(lc, wv_corr, sn_type,
                                                  mean, args.redshift,
                                                  args.verbose, filter_mean_function, filter_name_to_effwv, 
                                                  linear_filters, cubic_filters, template_filters)
    lc = lc.T
    wvs, wvind, wvrev = np.unique(lc[:, 2].astype('float64'), return_index=True, return_inverse = True)
    wvs = wvs*1000.0 + wv_corr 
    un_filts = my_filters[wvind]
    effwv = wvs[wvrev].astype('float64')
    my_filters = np.asarray(my_filters)
    ufilts = my_filters[wvind] 
    print('main ufilts:', ufilts)

    # Converts to AB magnitudes
    # dense_lc[:, :, 0] += flux_corr

    if args.verbose:
        print('Fitting Blackbodies, this may take a few minutes...')
    Tarr, Rarr, Terr_arr, Rerr_arr, covar_arr = fit_bb(dense_lc, wvs, args.mc,
                                                       args.T_max)

    # Calculate bolometric luminosity and error
    bol_lum = 4. * np.pi * Rarr**2 * sigsb * Tarr**4
    covar_err = 2. * (4. * np.pi * sigsb)**2 * (2 * Rarr * Tarr**4) * \
                (4 * Rarr**2 * Tarr**3) * covar_arr
    bol_err = 4. * np.pi * sigsb * np.sqrt(
                (2. * Rarr * Tarr**4 * Rerr_arr)**2
                + (4. * Tarr**3 * Rarr**2 * Terr_arr)**2
                )
    bol_err = np.sqrt(bol_err**2 + covar_err)

    if args.plot:
        if args.verbose:
            print('Making plots in ' + args.outdir)
        plot_gp(lc, dense_lc, snname, flux_corr, ufilts, wvs, test_data,
                args.outdir, sn_type, test_times, mean, args.template, filter_name_to_effwv, 
                linear_filters, cubic_filters)
        plot_bb_ev(lc, dense_lc, Tarr, Rarr, Terr_arr, Rerr_arr, snname,
                   args.outdir, sn_type)
        plot_bb_bol(lc, dense_lc, bol_lum, bol_err, snname, args.outdir, sn_type)

    if args.verbose:
        print('Writing output to ' + args.outdir)
    write_output(lc, dense_lc, Tarr, Terr_arr, Rarr, Rerr_arr,
                 bol_lum, bol_err, my_filters, snname, args.outdir, sn_type)
    print('job completed')


if __name__ == "__main__":
    main()
