#!/usr/bin/env python3
#Written by Kostantin Malanchev & Patrick Aleo

import os
import glob

import dataclasses
import pandas as pd

from astropy.io import ascii
from astropy.table import Table

import numpy as np

@dataclasses.dataclass
class Observation:
    MJD: float
    PASSBAND: str
    FLUX: float
    FLUXERR: float
    MAG: float
    MAGERR: float
    PHOTFLAG: str
    


REDSHIFT_UNKNOWN = -99.0


def read_YSE_ZTF_snana_dir(dir_name, keep_ztf=True):
    """
    file_path : str
        The file path to the combined YSE+ZTF light curve SNANA-style format data file.
    keep_ztf : bool
        True: Plots including ZTF data
        False : Plots not include ZTF data
    
    """
    files = os.listdir(os.getcwd())
    dat_files = []
    for file in files:
        if ".dat" in file:
            dat_files.append(file)
    snid_list = []
    meta_list = []
    yse_ztf_fp_df_list = []
    
    
    for file_path in dat_files:
        #print(file_path)
    
        meta = {}
        lc = []
        with open(file_path) as file:
            for line in file:
                try:
                    # SNID
                    if line.startswith('SNID: '):
                        _, snid = line.split()
                        meta['object_id'] = snid
                        meta['original_object_id'] = snid

                    # RA
                    if line.startswith('RA: '):
                        _, ra, _ = line.split()
                        meta['ra'] = float(ra)

                    # DEC
                    if line.startswith('DECL: '):
                        _, decl, _ = line.split()
                        meta['dec'] = float(decl)

                    # MWEBV
                    if line.startswith('MWEBV: '):
                        _, mwebv, _, _mwebv_error, *_ = line.split()
                        meta['mwebv'] = float(mwebv)

                    # REDSHIFT
                    if line.startswith('REDSHIFT_FINAL: '):
                        try:
                            _, redshift, _, _redshift_error, _z_type, _z_frame = line.split()
                        # 2020roe has empty redshift
                        except ValueError:
                            redshift = -99
                            redshift_err = -99
                            _z_type = 'NaN'
                            _z_frame = 'HELIO'
                        meta['redshift'] = float(redshift)
                        meta['redshift_err'] = float(_redshift_error)
                        meta['redshift_type'] = str(_z_type.split('(')[1].split(',')[0])
                        meta['redshift_frame'] = str(_z_frame.split(')')[0])
                        
                    # PHOTO-Z
                    if line.startswith('PHOTO_Z: '):
                        try:
                            _, photoz, _, _photoz_error, _, _  = line.split()
                        except ValueError:
                            photoz = -99
                            _photoz_error = -99
                        meta['photo_z'] = float(photoz)
                        meta['photoz_err'] = float(_photoz_error)
                        
                    # HOST INFO
                    if line.startswith('SN_OFFSET_TO_VETTED_HOST_GALAXY_CENTER: '):
                        try:
                            _, sn_offset, _ = line.split()         
                        except ValueError:
                            sn_offset = -99.000
                        meta['sn_offset'] = float(sn_offset)
                    
                    
                    if line.startswith('VETTED_HOST_GALAXY_NAME: '):
                        try:
                            _, host_gal_name_cat, host_gal_name_id, host_gal_name_source = line.split()
                            host_gal_name = str(host_gal_name_cat)+' '+str(host_gal_name_id)
                        except ValueError:
                            host_gal_name = 'None (or error)'
                            host_gal_name_source = '(SIMBAD,SDSS)'
                        meta['host_gal_name'] = host_gal_name
                        meta['host_gal_name_source'] = str(host_gal_name_source)
                        
                    if line.startswith('VETTED_HOST_GALAXY_REDSHIFT: '):
                        try:
                            _, hostz, _, _hostz_error, _hostz_type, _hostz_frame = line.split()
                        except ValueError:
                            hostz = -99
                            hostz_err = -99
                            _hostz_type = 'NaN'
                            _hostz_frame = 'HELIO'
                        meta['host_gal_z'] = float(hostz)
                        meta['host_gal_z_err'] = float(_hostz_error)
                        meta['host_gal_z_type'] = str(_hostz_type.split('(')[1].split(',')[0])
                        meta['host_gal_z_frame'] = str(_hostz_frame.split(')')[0])
                       
                    # PEAKMJD
                    if line.startswith('SEARCH_PEAKMJD: '):
                        _, pkmjd = line.split()
                        meta['peakmjd'] = search_peakmjd = float(pkmjd)
                       
                    # HOST LOGMASS
                    if line.startswith('HOST_LOGMASS: '):
                        _, host_logmass, _, host_logmass_error = line.split()
                        meta['host_logmass'] = float(host_logmass)
                     
                    # PEAK ABS MAG
                    if line.startswith('PEAK_ABS_MAG: '):
                        _, pkabsmag = line.split()
                        
                        try:
                            meta['peak_abs_mag'] = peak_abs_mag = float(pkabsmag)
                        except: # For NA
                            meta['peak_abs_mag'] = peak_abs_mag = str(pkabsmag)
                    
                    # SPEC CLASS
                    if line.startswith('SPEC_CLASS: '):
                        try:
                            _, sn, spec_subtype = line.split()
                            meta['transient_spec_class'] = transient_spec_class = str(sn+spec_subtype)
                        except:
                            _, spec_subtype = line.split()
                            meta['transient_spec_class'] = transient_spec_class = str(spec_subtype)
                       
                    # SPEC CLASS BROAD
                    if line.startswith('SPEC_CLASS_BROAD: '):
                        try: 
                            _, sn, subtype = line.split()
                            meta['spectype_3class'] = spectype_3class = str(sn+subtype)
                        except: 
                            _, subtype = line.split() 
                            meta['spectype_3class'] = spectype_3class = str(subtype)
                        
                    # PARSNIP PRED
                    if line.startswith('PARSNIP_PRED: '):
                        try: 
                            _, sn, p_pred = line.split()
                            meta['parsnip_pred_class'] = parsnip_pred_class = str(sn+p_pred)
                        except: 
                            _, p_pred = line.split() # for "NA" Prediction
                            meta['parsnip_pred_class'] = parsnip_pred_class = str(p_pred)
                        
                    # PARSNIP CONF
                    if line.startswith('PARSNIP_CONF: '):
                        _, p_conf = line.split()
                        meta['parsnip_pred_conf'] = parsnip_pred_conf = str(p_conf)
                        
                    # PARSNIP S1
                    if line.startswith('PARSNIP_S1: '):
                        _, s1, _, s1_error = line.split()
                        try: 
                            meta['parsnip_s1'] = float(s1)
                            meta['parsnip_s1_err'] = float(s1_error)
                        except: # NA
                            meta['parsnip_s1'] = str(s1) 
                            meta['parsnip_s1_err'] = str(s1_error)
                        
                    # PARSNIP S2
                    if line.startswith('PARSNIP_S2: '):
                        _, s2, _, s2_error = line.split()
                        try: 
                            meta['parsnip_s2'] = float(s2)
                            meta['parsnip_s2_err'] = float(s2_error)
                        except: # NA
                            meta['parsnip_s2'] = str(s2) 
                            meta['parsnip_s2_err'] = str(s2_error)
                        
                    # PARSNIP S3
                    if line.startswith('PARSNIP_S3: '):
                        _, s3, _, s3_error = line.split()
                        try: 
                            meta['parsnip_s3'] = float(s3)
                            meta['parsnip_s3_err'] = float(s3_error)
                        except: # NA
                            meta['parsnip_s3'] = str(s3) 
                            meta['parsnip_s3_err'] = str(s3_error)
                        
                    # SUPERPHOT PRED
                    if line.startswith('SUPERPHOT_PRED: '):
                        try: 
                            _, sn, s_pred = line.split()
                            meta['superphot_pred_class'] = superphot_pred_class = str(sn+s_pred)
                        except: 
                            _, s_pred = line.split() # for "NA" Prediction
                            meta['superphot_pred_class'] = superphot_pred_class = str(s_pred)
                        
                    # SUPERPHOT CONF
                    if line.startswith('SUPERPHOT_CONF: '):
                        _, s_conf = line.split()
                        meta['superphot_pred_conf'] = superphot_pred_conf = str(s_conf)
                    
                    # SUPERRAENN PRED
                    if line.startswith('SUPERRAENN_PRED: '):
                        try: 
                            _, sn, sr_pred = line.split()
                            meta['superraenn_pred_class'] = superraenn_pred_class = str(sn+sr_pred)
                        except: 
                            _, sr_pred = line.split() # for "NA" Prediction
                            meta['superraenn_pred_class'] = superraenn_pred_class = str(sr_pred)
                            
                    # SUPERRAENN CONF   
                    if line.startswith('SUPERRAENN_CONF: '):
                        _, sr_conf = line.split()
                        meta['superraenn_pred_conf'] = superraenn_pred_conf = str(sr_conf)    
                       
                       
                    # ZTF ZEROPOINT
                    if line.startswith('SET_ZTF_FP: '):
                        _, ztf_fp = line.split()
                        try:
                            meta['ztf_zeropoint'] = float(ztf_fp)
                        except:
                            meta['ztf_zeropoint'] = str(ztf_fp)
                            
                    # PEAKMJD
                    if line.startswith('PEAK_SNR: '):
                        _, pkSNR = line.split()
                        meta['peakSNR'] = float(pkSNR)
                        
                    # MAX MJD GAP
                    if line.startswith('MAX_MJD_GAP(days): '):
                        _, max_mjd_gap = line.split()
                        meta['max_mjd_gap'] = float(max_mjd_gap)
                        
                    # NOBS BEFORE PEAK
                    if line.startswith('NOBS_BEFORE_PEAK: '):
                        _, nobs_before_peak = line.split()
                        meta['nobs_before_peak'] = int(nobs_before_peak) 
                        
                    # NOBS TO THE PEAK OBS (ANY BAND)
                    if line.startswith('NOBS_TO_PEAK: '):
                        _, nobs_to_peak = line.split()
                        meta['nobs_to_peak'] = int(nobs_to_peak)  
                    
                    # NOBS AFTER PEAK 
                    if line.startswith('NOBS_AFTER_PEAK: '):
                        _, nobs_after_peak = line.split()
                        meta['nobs_after_peak'] = int(nobs_after_peak)  
                        
                    # PEAK MAGNITUDE
                    if line.startswith('SEARCH_PEAKMAG: '):
                        _, pkmag = line.split()
                        meta['peakmag'] = search_peakmag = float(pkmag)
                        
                    # PEAK FILTER (PASSBAND OF OBS w/ PEAK MAG OBS)
                    if line.startswith('SEARCH_PEAKFLT: '):
                        _, pkflt = line.split()
                        meta['peakflt'] = search_peakflt = str(pkflt)
                        
                        
                    # PEAK MAGNITUDE YSE-r or ZTF-r (Y) band for mag lim sample!
                    if line.startswith('PEAKMAG_YSE-r/ZTF-r(Y): '):
                        _, pkmag_rY = line.split()
                        meta['peakmag_rY'] = search_peakmag_rY = float(pkmag_rY)
                        
                    # PEAK FILTER of YSE-r or ZTF-r (Y) band peak mag
                    if line.startswith('PEAKFLT_YSE-r/ZTF-r(Y): '):
                        _, pkflt_rY = line.split()
                        meta['peakflt_rY'] = search_peakflt_rY = str(pkflt_rY)
                        
                    # FILTERS/PASSBANDS
                    if line.startswith('FILTERS: '):
                        _, pbs = line.split()
                        meta['passbands'] = passbands = str(pbs)
                        
                    # TOTAL OBS
                    if line.startswith('NOBS_wZTF: ') or line.startswith('NOBS_AFTER_MASK: '):
                        _, desired_nobs = line.split()
                        meta['num_points'] = int(desired_nobs)
                        continue
                        
                        
                except ValueError as e:
                    print(e)
                    print(meta['object_id'])
                    raise e
                    
                    
                if not line.startswith('OBS: '):
                    continue

                _obs, mjd, flt, _field, fluxcal, fluxcalerr, mag, magerr, _flag = line.split()
                lc.append(Observation(
                    MJD=float(mjd),
                    PASSBAND=str(flt),
                    FLUX=float(fluxcal),
                    FLUXERR=float(fluxcalerr),
                    MAG=float(mag),
                    MAGERR=float(magerr),
                    PHOTFLAG=str(_flag))
                )

                
                
                 

        meta.setdefault('mwebv', 0.0)

        #assert len(meta) == 13, f'meta has wrong number of values,\nmeta = {meta}'
        assert len(lc) == meta['num_points']
        table = Table([dataclasses.asdict(obs) for obs in lc if keep_ztf or obs.FLT not in ZTF_BANDS])

        yse_ztf_fp_df = table.to_pandas()
        
        snid_list.append(snid)
        meta_list.append(meta)
        yse_ztf_fp_df_list.append(yse_ztf_fp_df)
        
    return snid_list, meta_list, yse_ztf_fp_df_list
