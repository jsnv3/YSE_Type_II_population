#!/usr/bin/env python

# Interpolation and data fitting
from scipy import interpolate as interp
from scipy.optimize import minimize, curve_fit
from scipy.integrate import simps
# Astronomy-specific and data importing
from astroquery.svo_fps import SvoFps
from astropy.table import Table, unique
from astropy.io import ascii
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Gaussian processes
import george
from george.modeling import Model
# General options
import numpy as np
import os
import sys
import extinction
import emcee
import importlib_resources
import glob
import copy
# pysynphot dependencies for nebular
import pysynphot as S

# Ignore warnings 
# import warnings
# warnings.filterwarnings('ignore')

# Define a few important constants that will be used later
epsilon = 0.001
c = 2.99792458E10     # cm / s
sigsb = 5.6704e-5     # erg / cm^2 / s / K^4
h = 6.62607E-27       # erg / Hz
ang_to_cm = 1e-8      # angstrom -> cm
k_B = 1.38064852E-16  # cm^2 * g / s^2 / K
dist_10pc = 10.0 * 3.08568025e18 # 10pc in cm

def offset(mag, A):
    return(mag+A)

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

def import_nebular(bandpasses, use_jerkstrand=False,
    use_dessart=True, nebular_normalize=1.0e41, unit='AB',
    nebular_dir='../nebular'):

    if not os.path.exists(nebular_dir):
        raise Exception(f'ERROR: {nebular_dir} does not exist.  '+\
            'Set --nebular-dir to directory with nebular spectra.')
    j_dir = os.path.join(nebular_dir, 'jerkstrand')
    d_dir = os.path.join(nebular_dir, 'dessart')
    if use_jerkstrand and not os.path.exists(j_dir):
        raise Exception(f'ERROR: cannot find jerkstrand models in {nebular_dir}')
    if use_dessart and not os.path.exists(d_dir):
        raise Exception(f'ERROR: cannot find dessart models in {nebular_dir}')

    outdata = []

    if use_dessart:
        for file in sorted(glob.glob(os.path.join(d_dir, '*.fl'))):
            print(f'Importing nebular file: {file}')
            wave, flux = np.loadtxt(file, unpack=True, dtype=float)

            # Normalize flux so pseudo-bolometric luminosity is nebular_normalize
            norm = simps(flux, wave)
            flux = flux/norm * nebular_normalize/(4.0 * np.pi * (dist_10pc)**2)

            sp = S.ArraySpectrum(wave, flux, waveunits='angstrom',
                fluxunits='flam')

            data = {'file': file, 'normalize': nebular_normalize,
                'unit':unit, 'teff': 2500.0}

            for bp in bandpasses:

                obs = S.Observation(sp, bp, binset=wave)
                if unit.lower()=='ab':
                    mag = obs.effstim('abmag')
                elif unit.lower()=='vega':
                    mag = obs.effstim('vegamag')

                data[bp.name]=mag

            outdata.append(data)

    return(outdata)


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

        flux = 10.**(mag/-2.5) * zpts[-1] / (1.+redshift)

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


    my_template_file = os.path.join('template_bank',f'smoothed_sn{sn_type}.npz')
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
                template_filters = None, delta_time = 1.0):
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
    delta_time : float
        Step size in time to interpolate light curves
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
    max_time = int(np.ceil(np.max(times)))

    length_of_times = int(np.round((np.max(times)-np.min(times))/delta_time))
   
    x_pred = np.zeros((length_of_times, 2)) 
    dense_fluxes = np.zeros((length_of_times, nfilts)) 
    dense_errs = np.zeros((length_of_times, nfilts))

    # test_y is only used if mean = True
    # but I still need it to exist either way
    test_y = []
    test_times = [] 
    linear_results = {} 
    dense_lc_list = [] 
    key_count = 0 
    if not use_mean:
        mean = 0 
        gp = george.GP(kernel, mean = 0)
    else:
        #create linear splines, set up and run GP 
        #use the entire LC range 
        for i, filt in enumerate(linear_filters):
            if verbose:
                print(f'Using linear spline for {filt}')
                
            idx = np.where(filters == filt)
            x = times[idx]
            y = fluxes[idx]
            yerr = errs[idx] 
            central_wv = wv_effs[idx] * 1000 + wv_corr 
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
            dense_lc_filtered = np.column_stack((x_pred, dense_fluxes, dense_errs, central_wv_array)) 
            linear_results[filt] = { 
                'dense_lc_filtered':dense_lc_filtered
            }
                 
        # extract dense_lc for each filter
        for filter in linear_results.values():
            dense_lc_individual = filter.get('dense_lc_filtered')
            dense_lc_list.append(dense_lc_individual)
               
        # concatenate into one object 
        #move this to bottom once cubic/template functions incorporated
        
        dense_lc = np.concatenate(dense_lc_list, axis = 0)

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
        
        #sort data by time 
        dense_lc = dense_lc[dense_lc[:,0].argsort()]
                                        
    return dense_lc, test_y, test_times, ufilts, ufilts_in_angstrom


def fit_sed(dense_lc, use_mcmc=False, T_max=40000., nebular_time=None,
    delta_time=1.0, filters=[], wvs=[]):
    """
    Fit model SEDs (either blackbodies or nebular SED) to the GP LC

    Parameters
    ----------
    dense_lc : numpy.array
        GP-interpolated LC

    Optional Parameters
    -------------------
    use_mcmc : boolean
        Use Monte Carlo Markov-Chain to fit SEDs
    T_max : float
        maximum temperature (in Kelvin) to fit to SEDs
    nebular_time : float
        relative time from peak light (in days) to switch from blackbody to nebular SED fitting
    """

    epoch_data = get_epoch_data(dense_lc)
    # Sort by first element in list, which is relative time from peak light
    epoch_data = sorted(epoch_data, key=lambda x: x[0])
    # Get just time data and save as array
    all_times = np.array([x[0] for x in epoch_data])
    print(f'len(all_times):{len(epoch_data)}')
    if nebular_time is None:
        T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr = fit_bb(epoch_data, 
            use_mcmc=use_mcmc, T_max=T_max, delta_time=delta_time)
        return(epoch_data, T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr)
    else:
        nebular_idx = -1
        for i,epoch in enumerate(epoch_data):
            if epoch[0] > nebular_time:
                nebular_idx = i
                break

        if nebular_idx==-1:
            print(f'WARNING: data do not extend to nebular time {nebular_time}')
            T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr = fit_bb(epoch_data, 
                use_mcmc=use_mcmc, T_max=T_max, delta_time=delta_time)
            return(epoch_data, T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr)
        else:
            
            # to prevent discontinuities between models, need to apply apodization 
            # we want window length to be # of days before and after the model switch date 
            window_lower_idx = nebular_idx - 10
            window_upper_idx = nebular_idx + 10
            window_length = window_upper_idx - window_lower_idx 
            window_range = np.linspace(0, 1, window_length)
            window = np.exp(-3.5 * window_range)
            print(f'window:{window}')
            
            bT_arr, bR_arr, bTerr_arr, bRerr_arr, bcovar_arr = fit_bb(epoch_data[:window_upper_idx],
                use_mcmc=use_mcmc, T_max=T_max, delta_time=delta_time)
            nT_arr, nR_arr, nTerr_arr, nRerr_arr, ncovar_arr = fit_nebular(epoch_data[window_lower_idx:],
                use_mcmc=use_mcmc, delta_time=delta_time,
                filters=filters, wvs=wvs) 
            return(epoch_data, bT_arr, bR_arr, bTerr_arr, bRerr_arr, bcovar_arr, nT_arr, nR_arr, nTerr_arr, nRerr_arr, ncovar_arr, window, nebular_idx)

def get_epoch_data(all_data, delta_time=1.0):

    shape = all_data.shape

    # Get data by epoch
    min_time = np.min(all_data[:,0])
    max_time = np.max(all_data[:,0])
    ntimes = int(np.round((max_time-min_time)/delta_time))
    times = np.linspace(min_time, max_time, ntimes)

    epoch_data = []

    for i,t in enumerate(times):
        mask = np.abs(all_data[:,0]-t) < 0.5 * delta_time

        if len(all_data[mask,0])==0:
            continue

        subdata = all_data[mask,:]

        data = [t]
        for datapoint in subdata:
            data.append([datapoint[1],datapoint[2],datapoint[3]])

        epoch_data.append(data)

    return(epoch_data)

def get_flam(datapoint):
    """
    Convenience function to transform magntidues and magnitude errors into
    f_lambda and f_lambda_err

    Parameters
    ----------
    datapoint : list
        A list with three elements - assumed to be magnitude, magnitude error,
        and wavelength

    Output
    ------
    f_lambda : float
        output f_lambda for input magnitude and wavelength
    f_lambda_err : float
        output f_lambda_err for input magnitude, magnitude error, and wavelength
    """

    mag = datapoint[0]
    magerr = datapoint[1]
    wavelength = datapoint[2] 
    fnu = 10.**((-mag + 48.6) / -2.5)
    fnu = fnu * 4. * np.pi * (3.086e19) ** 2
    fnu_err = np.abs(0.921034 * 10. ** (0.4 * mag - 19.44)) \
            * magerr * 4. * np.pi * (3.086e19) ** 2

    flam = fnu*c / (wavelength * ang_to_cm) ** 2
    flam_err = fnu_err * c / (wavelength * ang_to_cm) ** 2 

    return(flam, flam_err)

def fit_bb(epoch_data, use_mcmc=False, T_max=40000., save_chains=False,
    delta_time=1.0):
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

    ntimes = len(epoch_data)
    # initialize arrays for temp, radius, errors, covar 
    T_arr = np.zeros(ntimes)
    R_arr = np.zeros(ntimes)
    Terr_arr = np.zeros(ntimes)
    Rerr_arr = np.zeros(ntimes)
    covar_arr = np.zeros(ntimes)

    prior_fit = (9000., 0.2e15) 

    # loop over unique epoch and apply bbfit 
    for i, epoch in enumerate(epoch_data):

        # Get wavelengths, flux, and flux_err as arrays
        wavelengths = np.array([e[2] for e in epoch[1:]])
        flux_data = np.array([get_flam(e) for e in epoch[1:]])

        flams = flux_data[:,0]
        flamerrs = flux_data[:,1]

        idx_sort = np.argsort(wavelengths)
        wavelengths = wavelengths[idx_sort]
        flams = flams[idx_sort]
        flamerrs = flamerrs[idx_sort]
    
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
                                            args=[wavelengths, flams, flamerrs])
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
                BBparams, covar = curve_fit(bbody, wavelengths, flams, maxfev=15000,
                                            p0=prior_fit, sigma=flamerrs,
                                            bounds=(0, [T_max, np.inf]), absolute_sigma = True)
                      
                # Get temperature and radius, with errors, from fit 
                T_arr[i] = BBparams[0]
                Terr_arr[i] = np.sqrt(np.diag(covar))[0]
                R_arr[i] = np.abs(BBparams[1])
                Rerr_arr[i] = np.sqrt(np.diag(covar))[1]
                covar_arr[i] = covar[0,1]
                prior_fit = BBparams            
                
            except RuntimeWarning:
                print('runtime warning')
                T_arr[i] = np.nan
                R_arr[i] = np.nan
                Terr_arr[i] = np.nan
                Rerr_arr[i] = np.nan
                covar_arr[i] = np.nan 
                
        #debugging
        if save_chains:
            np.savetxt('T_arr', T_arr)
            np.savetxt('Terr', Terr_arr)
            np.savetxt('Rarr', R_arr)
            np.savetxt('Rerr', Rerr_arr)

    return T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr

def fit_nebular(epoch_data, filters, wvs, use_mcmc=False, delta_time=1.0):

    print('Generating nebular models...')
    index = SvoFps.get_filter_index(wavelength_eff_min=100*u.angstrom,
                                    wavelength_eff_max=30000*u.angstrom,
                                    timeout=3600)

    bps = []
    for filt in filters:
        print(filt)
        mask = index['filterID']==filt
        transdata = Table.read(index[mask]['TrasmissionCurve'][0])
        # Make sure transmission data zeros out
        minwave = np.min(transdata['Wavelength'])
        maxwave = np.max(transdata['Wavelength'])

        # Assume wavelength in angstroms
        transdata.add_row([minwave-1.0, 0.0])
        transdata.add_row([maxwave+1.0, 0.0])

        # Correct for multiple entries at same wavelength
        transdata = unique(transdata, keys=['Wavelength'], keep='first')

        transdata.sort('Wavelength')
        transdata = transdata.filled()

        bp = S.ArrayBandpass(transdata['Wavelength'].data, 
            transdata['Transmission'].data, name=filt)
        bps.append(bp)

    nebular_models = import_nebular(bps)

    print('Picking best nebular model...')
    filters = np.array(filters)
    model_results = []
    for i, epoch in enumerate(epoch_data):

        epoch_results = []

        # Get wavelengths, flux, and flux_err as arrays
        wavelengths = np.array([e[2] for e in epoch[1:]])
        mags = np.array([-1.0*e[0] for e in epoch[1:]])
        magerrs = np.array([e[1] for e in epoch[1:]])
        epoch_filts = np.array([filters[wvs==w][0] for w in wavelengths])

        for model in nebular_models:
            model_mags = np.array([model[f] for f in epoch_filts])
            popt, pcov = curve_fit(offset, mags, model_mags, sigma=magerrs,
                p0=(0.0), absolute_sigma = True)
            # If popt is negative, that means model_mags are brighter than 
            # observed mags, hence luminosity needs to be scaled down
            luminosity = model['normalize'] * 10**(0.4 * popt[0])
            dlum = 0.921034 * model['normalize'] * 10**(0.4 * popt[0]) * np.sqrt(pcov[0][0])
            epoch_results.append({'file':model['file'],
                'luminosity':luminosity,'epoch':epoch[0],
                'parameter': popt[0], 'dlum': dlum,
                'covariance':pcov[0][0],'teff':model['teff']})

        model_results.append(epoch_results)

    # Find model with lowest total covariance across all epochs
    best_covariance = np.inf
    best_model = ''
    for i,model in enumerate(nebular_models):

        total_covariance = 0.0
        for models in model_results:
            for m in models:
                if m['file']==model['file']:
                    total_covariance+=m['covariance']

        if best_covariance > total_covariance:
            best_covariance=copy.copy(total_covariance)
            best_model = model['file']

    print(f'Best model is {best_model} with total covariance {best_covariance}') 


    best_idx = [i for i,model in enumerate(nebular_models) if model['file']==best_model][0]

    # Grab luminosities, T_eff, radii from output
    Tarr = [] ; Tarr_err = [] ; Rarr = [] ; Rarr_err = [] ; covar = []
    for i,epoch in enumerate(epoch_data):

        model = model_results[i][best_idx]
        Tarr.append(model['teff'])
        Tarr_err.append(0.0)

        lum = model['luminosity']
        dlum = model['dlum']

        scaling = 1./(4. * np.pi * sigsb * model['teff']**4)**0.5
        radius = lum**0.5 * scaling
        radius_err = dlum/(2 * np.sqrt(lum)) * scaling

        Rarr.append(radius)
        Rarr_err.append(radius_err)

        covar.append(model['covariance'])

    Tarr = np.array(Tarr)
    Rarr = np.array(Rarr)
    Tarr_err = np.array(Tarr_err)
    Rarr_err = np.array(Rarr_err)
    covar = np.array(covar)

    return Tarr, Rarr, Tarr_err, Rarr_err, covar

def plot_gp(epoch_data, lc, wvs, snname, filter_name_to_effwv, sn_type,
    linear_filters=[], mean=True, outdir='.'):
    '''
    Plot the GP-interpolate LC and save

    Parameters
    ----------
    epoch_data : list
        Data sorted by epoch
    lc : numpy.array
        Original LC data
    wvs : numpy.array
        List of central wavelengths, for colors
    snname : string
        SN Name
    sn_type : string
        Type of sn template used for GP mean function
    mean : bool
        Whether or not a non-zero mean function is being used in GP
    outdir : string
        Output directory

    Output
    ------
    '''
    #dictionary to assign filters a color 
    filter_colors = {}
        
    #plot interpolation + errors 
    plt.figure()
    cmap = plt.get_cmap('tab10')

    for i, wavelength in enumerate(wvs):
        color = cmap(i)
        # Sort times, mags, magerrs into lists for this filter
        times = []
        mags = []
        magerrs = []
        for epoch in epoch_data:
            for e in epoch[1:]:
                if e[2]==wavelength:
                    times.append(epoch[0])
                    mags.append(e[0])
                    magerrs.append(e[1])

        times = np.array(times)
        mags = np.array(mags)
        magerrs = np.array(magerrs)

        filter_colors[wavelength] = color
        plt.plot(times, mags, color = color, lw = 1.5, alpha = 0.5)
        plt.fill_between(times, mags-magerrs, mags + magerrs, color = 'k', 
            alpha = 0.2)
        

    #plot original data + errorbars 
    for i, filt in enumerate(linear_filters):
        central_wv = filter_name_to_effwv[filt]
        color = filter_colors[central_wv]
        idx = np.where(lc[:,5] == filt)
        x = lc[:,0].astype('float64')[idx]
        y = lc[:,1].astype('float64')[idx]
        yerr = lc[:,3].astype('float64')[idx]
        plt.errorbar(x, y, yerr=yerr, fmt = '.', capsize = 0, color = color, 
            label = filt.split('/')[-1])

    ylims = [np.max(lc[:,1].astype('float')),np.min(lc[:,1].astype('float'))]
    yran = ylims[0]-ylims[1]
    ylims = [ylims[0]+0.05*yran, ylims[1]-0.05*yran]
    plt.ylim(ylims)

    if mean:
        plt.title(snname + ' using sn' + sn_type)
    else:
        plt.title(snname + ' Light Curves')

    plt.legend()
    plt.xlabel('Time(days)')
    plt.ylabel('Absolute Magnitudes')
    plt.gca().invert_yaxis()
    outfig = os.path.join(outdir, f'{snname}_{sn_type}_gp.png')
    plt.savefig(outfig)
    plt.clf()

    return 1

def plot_bb_ev(epoch_data, Tarr, Rarr, Terr_arr, Rerr_arr, snname, sn_type,
    outdir='.', nTarr = None, nRarr = None, nTerr_arr = None, nRerr_arr = None):
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
    
    plot_times = np.array([e[0] for e in epoch_data])
    len_Tarr = len(Tarr)
    len_nTarr = len(nTarr)
    print(f'len_Tarr: {len_Tarr}, len_nTarr: {len_nTarr}')
    fig, axarr = plt.subplots(2, 1, sharex=True) 
    idx = np.where(Terr_arr != np.inf)
    axarr[0].plot(plot_times[idx], Tarr[idx] / 1.e3, color='b', label = 'blackbody model')
    axarr[0].fill_between(plot_times[idx], Tarr[idx]/1.e3 - Terr_arr[idx]/1.e3,
                          Tarr[idx]/1.e3 + Terr_arr[idx]/1.e3, color='k', alpha=0.2)
    axarr[0].plot(plot_times[-len_nTarr:], nTarr / 1.e3, 'r--', label = 'nebular model')
    axarr[0].set_ylabel('Temp. (1000 K)') 


    axarr[1].plot(plot_times[idx], Rarr[idx] / 1e15, color='b', label = 'blackbody model')
    axarr[1].fill_between(plot_times[idx], Rarr[idx]/1e15 - Rerr_arr[idx]/1e15,
                          Rarr[idx]/1e15 + Rerr_arr[idx]/1e15, color='k', alpha=0.2)
    axarr[1].plot(plot_times[-len(nRarr):], nRarr / 1e15, 'r--', label = 'nebular model')
    axarr[1].set_ylabel(r'Radius ($10^{15}$ cm)')
    axarr[1].set_xlabel('Time (Days)')
    axarr[0].set_title(snname + ' Black Body Evolution')
    axarr[0].legend()
    axarr[1].legend()

    outfig = os.path.join(outdir, f'{snname}_{sn_type}_bb_ev.png')
    plt.savefig(outfig)
    plt.clf()

    return 1


def plot_bb_bol(epoch_data, bol_lum, bol_err, snname, sn_type, outdir='.'):
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

    plot_times = np.array([e[0] for e in epoch_data])
    
    plt.plot(plot_times, bol_lum, 'k')
    plt.fill_between(plot_times, bol_lum-bol_err, bol_lum+bol_err,
                     color='k', alpha=0.2)

    plt.title(snname + ' Bolometric Luminosity')
    plt.xlabel('Time (Days)')
    plt.ylabel('Bolometric Luminosity')
    plt.yscale('log')
    outfig = os.path.join(outdir, f'{snname}_{sn_type}_bb_bol.png')
    plt.savefig(outfig)
    plt.clf()

    return 1


def write_output(epoch_data, Tarr, Terr_arr, Rarr, Rerr_arr,
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

    times = np.array([e[0] for e in epoch_data])

    tabledata = np.stack((times,Tarr / 1e3, Terr_arr / 1e3, Rarr / 1e15,
                          Rerr_arr / 1e15, np.log10(bol_lum),
                          np.log10(bol_err))).T

    table_header = ['Time (MJD)', 'Temp./1e3 (K)', 'Temp. Err.',
                         'Radius/1e15 (cm)', 'Radius Err.',
                         'Log10(Bol. Lum)', 'Log10(Bol. Err)']

    table = Table(rows = tabledata, names = table_header, meta = {'name':'first table'})

    format_dict = {head: '%0.3f' for head in table_header}
    ascii.write(table, outdir + snname + '_' + str(sn_type) + '.txt',
                formats=format_dict, overwrite=True)

    return 1 

def add_options():
    import argparse
    # Define all arguments
    default_data = os.path.join('example','SN2010bc.dat')
    default_data = str(default_data)
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
    parser.add_argument('--delta-time','-dt', type=float, default=1.0,
                        help='Step size in time for light curve interpolation.')
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
    parser.add_argument('--settings', dest = 'settings', type=str, 
        default ='settings.txt', help = 'Settings file name')
    parser.add_argument('--nebular-dir', default='../nebular', type=str,
        help='Directory where nebular models are stored.')
    parser.add_argument('--nebular', type=float, default=None,
        help='Use a nebular model starting from this relative time in the '+\
        'SED fit.')
    parser.add_argument('--nebular-model', type=str, default=None,
        help='Use this nebular model instead of choosing the best joint fit '+\
        'to the photometry across all epochs.')

    args = parser.parse_args()

    return(args)


def main():
    # Import all options
    args = add_options()

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
        print(f'Using {sn_type} template.')

    datavals = interpolate(lc, wv_corr, sn_type, mean, args.redshift,
        args.verbose, filter_mean_function, filter_name_to_effwv, 
        linear_filters, cubic_filters, template_filters,
        delta_time=args.delta_time)
    dense_lc, test_data, test_times, ufilts, ufilts_in_angstrom = datavals

    lc = lc.T
    wvs, wvind, wvrev = np.unique(lc[:, 2].astype('float64'), return_index=True,
        return_inverse = True)
    wvs = wvs*1000.0 + wv_corr 
    un_filts = my_filters[wvind]
    effwv = wvs[wvrev].astype('float64')
    my_filters = np.asarray(my_filters)
    ufilts = my_filters[wvind] 
    print('main ufilts:', ufilts)

    if args.verbose:
        print('Fitting SEDs, this may take a few minutes...')
    if args.nebular is None:
        epoch_data, Tarr, Rarr, Terr_arr, Rerr_arr, covar_arr= fit_sed(dense_lc, use_mcmc=args.mc,
                                                       T_max=args.T_max, 
                                                       delta_time=args.delta_time,
                                                       filters=ufilts, wvs=wvs,
                                                       nebular_time=args.nebular)
        # Calculate bolometric luminosity and error
        
        bol_lum = 4. * np.pi * Rarr**2 * sigsb * Tarr**4
        covar_err = 2. * (4. * np.pi * sigsb)**2 * (2 * Rarr * Tarr**4) * \
                    (4 * Rarr**2 * Tarr**3) * covar_arr
        bol_err = 4. * np.pi * sigsb * np.sqrt(
                    (2. * Rarr * Tarr**4 * Rerr_arr)**2
                    + (4. * Tarr**3 * Rarr**2 * Terr_arr)**2)
        bol_err = np.sqrt(bol_err**2 + covar_err) 
        if args.plot:
            if args.verbose:
                print(f'Making plots in {args.outdir}')    
            plot_gp(epoch_data, lc, wvs, snname, filter_name_to_effwv, sn_type,
            linear_filters=linear_filters, outdir=args.outdir)
            plot_bb_ev(epoch_data, Tarr, Rarr, Terr_arr, Rerr_arr, snname,
                   sn_type, outdir=args.outdir)
            plot_bb_bol(epoch_data, bol_lum, bol_err, snname, sn_type,
            outdir=args.outdir)
        if args.verbose:
            print(f'Writing output to {args.outdir}')
        write_output(epoch_data, Tarr, Terr_arr, Rarr, Rerr_arr, bol_lum, 
            bol_err, my_filters, snname, args.outdir, sn_type)
        print('job completed')
                
    else: 
        epoch_data, bTarr, bRarr, bTerr_arr, bRerr_arr, bcovar_arr, nTarr, nRarr, nTerr_arr, nRerr_arr, ncovar_arr, window, nebular_idx = fit_sed(dense_lc, use_mcmc=args.mc,
                                                                                            T_max = args.T_max,
                                                                                            delta_time = args.delta_time,
                                                                                            filters = ufilts, wvs = wvs,
                                                                                            nebular_time = args.nebular)
        # here we blend the two models using apodization in only the luminoisty space 
        # this should smooth out any disconitinuties 
        # first calculate bolometric luminosity and error for the blackbody model only 
        
        plt.figure(figsize = (10,10))
        len_bTarr = len(bTarr)
        len_nTarr = len(nTarr)
        print(f'len_bTarr :{len_bTarr}, len_nTarr: {len_nTarr}')
        times = [i[0] for i in epoch_data]
        plt.plot(times[:len_bTarr], bTarr, color = 'b', marker = 'o', label = 'bb temp')
        plt.plot(times[-len_nTarr:], nTarr, color = 'r', marker = 'o', label = 'nebular')
        plt.legend()
        plt.show()        
        
        bb_bol_lum = 4. * np.pi * bRarr**2 * sigsb * bTarr**4
        bb_covar_err = 2. * (4. * np.pi * sigsb)**2 * (2 * bRarr * bTarr**4) * \
                    (4 * bRarr**2 * bTarr**3) * bcovar_arr
        bb_bol_err = 4. * np.pi * sigsb * np.sqrt(
                    (2. * bRarr * bTarr**4 * bRerr_arr)**2
                    + (4. * bTarr**3 * bRarr**2 * bTerr_arr)**2)
        bb_bol_err = np.sqrt(bb_bol_err**2 + bb_covar_err) 
        
        #do the same for neb 
        neb_bol_lum = 4. * np.pi * nRarr**2 * sigsb * nTarr**4
        neb_covar_err = 2. * (4. * np.pi * sigsb)**2 * (2 * nRarr * nTarr**4) * \
                    (4 * nRarr**2 * nTarr**3) * ncovar_arr
        neb_bol_err = 4. * np.pi * sigsb * np.sqrt(
                    (2. * nRarr * nTarr**4 * nRerr_arr)**2
                    + (4. * nTarr**3 * nRarr**2 * nTerr_arr)**2)
        neb_bol_err = np.sqrt(neb_bol_err**2 + neb_covar_err) 
        
        # now we use the window function and propagate error 
        window_length = len(window)
        window_bb_bol = window * bb_bol_lum[-window_length:]
        window_neb_bol = (1 - window) * neb_bol_lum[:window_length]
        window_bol = window_bb_bol + window_neb_bol
        
        window_bb_bol_err = bb_bol_err[-window_length:]
        window_neb_bol_err = neb_bol_err[:window_length]
        window_err = np.sqrt((window * (window_bb_bol_err ** 2)) + ((1- window) * (window_neb_bol_err ** 2)))
        
        # now we combine the three bol arrays 
        # first chop off the last n bb_bol_lum/err and first n neb_bol_lum/err 
                #where n is window_length 
        nw_bb_bol_lum = bb_bol_lum[:-window_length]
        nw_bb_bol_err = bb_bol_err[:-window_length]
        nw_neb_bol_lum = neb_bol_lum[window_length:]
        nw_neb_bol_err = neb_bol_err[window_length:]
        
        #concatenate arrays 
        
        bol_lum = np.concatenate([nw_bb_bol_lum, window_bol, nw_neb_bol_lum])
        bol_err = np.concatenate([nw_bb_bol_err, window_err, nw_neb_bol_err])
        print(f'main len_bol_lum{len(bol_lum)}')
        print(f'main len bol_err:{len(bol_err)}')

    
        if args.plot:
            if args.verbose:
                print(f'Making plots in {args.outdir}')
                
            plot_gp(epoch_data, lc, wvs, snname, filter_name_to_effwv, sn_type,
                linear_filters=linear_filters, outdir=args.outdir)
            plot_bb_ev(epoch_data, bTarr, bRarr, bTerr_arr, bRerr_arr, snname,
                    sn_type, outdir=args.outdir, nTarr=nTarr, nRarr=nRarr, nTerr_arr=nTerr_arr, nRerr_arr=nRerr_arr)
            plot_bb_bol(epoch_data, bol_lum, bol_err, snname, sn_type,
                outdir=args.outdir)

        if args.verbose:
            print(f'Writing output to {args.outdir}')
        # write_output(epoch_data, Tarr, Terr_arr, Rarr, Rerr_arr, bol_lum, 
        #     bol_err, my_filters, snname, args.outdir, sn_type)
        print('job completed')


if __name__ == "__main__":
    main()


