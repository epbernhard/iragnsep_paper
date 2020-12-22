###################################
# Example of how to use iragnsep. #
# In this example we fit the      # 
# galaxy mrk1066, first including #
# the IRS data and then based on  #
# broadband photometry only.      #
###################################

# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import iragnsep
# import iragnsep

from iragnsep import run_all

import pdb

# redshift of mrk1066
z = 0.011

######
# Fit with IRS spectra
#####

# IRS spectra data
spec = pd.read_csv('./IC5063_IRS.csv')

wavIRS = spec['lambdamic'].values # wavelengths of the IRS [microns]
fluxIRS = spec['fluxJy'].values # fluxes of the IRS [Jy]
efluxIRS = spec['efluxJy'].values # uncertainties on the IRS fluxes [Jy]

# We only retain fluxes that are 3 sigma detected
wavIRS_fit = wavIRS[fluxIRS > 3.* efluxIRS]
fluxIRS_fit = fluxIRS[fluxIRS > 3.* efluxIRS]
efluxIRS_fit = efluxIRS[fluxIRS > 3.* efluxIRS]

# Herschel photometry
phot = pd.read_csv('./IC5063_photo.csv')

wav = np.array([70., 100., 160., 250., 350., 500]) # wavelengths of the available Herschel photometry [microns]
flux = np.array([phot['F70mJy'].values, phot['F100mJy'].values, \
				 phot['F160mJy'].values, phot['F250mJy'].values,\
				 phot['F350mJy'].values, phot['F500mJy']]).flatten()*1e-3 # Array of fluxes for the Herschel photometry [Jy]
eflux = np.array([phot['eF70mJy'].values, phot['eF100mJy'].values, \
				 phot['eF160mJy'].values, phot['eF250mJy'].values,\
				 phot['eF350mJy'].values, phot['eF500mJy']]).flatten()*1e-3 # Array of uncertainties for the Herschel photometry [Jy]
filters = np.array(['PACS70', 'PACS100', 'PACS160', 'SPIRE250ps', 'SPIRE350ps', 'SPIRE500ps']) # Array containing the names of the photometric filters.
UL_phot = np.array([0., 0., 1., 0., 0., 0.]) # Vector containing the UL for the photometry. Here, we set the flux at 160 micron as an UL.
S9p7 = 0.2 # Here, we fix S9p7 as it is low, and iragnsep struggles to infer low values. The SFR will not change significantly if you decide to let iragnsep calculate the S9p7.

# The observed wavelengths, fluxes and uncertainties for the IRS spectra and the Herschel photometry are fed to iragnsep.
# The module run_all contains functions that fits, compare the models, analyse the results and plots.
# In particular, since we have the IRS spectrum of IC5063, we use the function fitSpec of the module run_all.

# Keyword used here:
# z: redshift
# UL: indicates which of the photometric fluxes need to be set as an upper limit for the fits.
# Nmc: numbers of MCMC to perform. The more the better. 10000 is just converged, 50000 is good.
# saveRes: If set to True, save the plots and the tables at pathFig+sourceName_fitRes_spec.pdf and pathFig+sourceName_fitRes_spec.csv, respectively.

res, resBM = run_all.fitSpec(wavIRS_fit, fluxIRS_fit, efluxIRS_fit, \
							 wav, flux, eflux, \
							 filters, \
							 z = z, \
							 Nmc = 50000, \
							 S9p7_fixed = S9p7, \
							 ULPhot = UL_phot, \
							 sourceName = 'IC5063', \
							 saveRes = True, \
							 pathTable = './', \
							 pathFig = './')

# OUTPUT
# res: dataframe containing the results of the fits.
# resBM: same as res but for the Best Model [BM] only.

# Print the best model SFR and the AIC weighted average SFR.
print('***************************')
print('The SFR of IC5063 is ', np.round(np.sum(res['wSFR']),2), 'Msun/yr.') # The sum of the weighted SFR is used (statistically better than just using the best model).
print('***************************')

##################################################################

######
# Fit with photometry only
#####
filters = np.array(['WISE_W3', 'WISE_W4', 'MIPS24','PACS70', 'PACS100', 'PACS160', 'SPIRE250ps', 'SPIRE350ps', 'SPIRE500ps']) # Filters used in the fit.
wav = np.array([12., 22., 24., 70., 100., 160., 250., 350., 500.]) # Observed wavelength to fit.
flux = np.array([31.674 * 10**(-phot['w3mpro']/2.5)*1e3, 8.363 * 10**(-phot['w4mpro']/2.5)*1e3, \
		phot['m1_f_psf']*1e-3, phot['F70mJy'], phot['F100mJy'], phot['F160mJy'], phot['F250mJy'], phot['F350mJy'], phot['F500mJy']])*1e-3 # Fluxes
eflux = np.array([31.674 * 10**(-phot['w3mpro']/2.5)*np.log(10)*phot['w3sigmpro']*1e3, 31.674 * 10**(-phot['w4mpro']/2.5)*np.log(10)*phot['w4sigmpro']*1e3, \
	phot['m1_df_psf']*1e-3, phot['eF70mJy'], abs(phot['eF100mJy']), phot['eF160mJy'], phot['eF250mJy'], phot['eF350mJy'], phot['eF500mJy']])*1e-3 # uncertainties
UL = np.zeros(len(wav)) # UL, here we define 160 micron as un UL
UL[-4] = 1.


# Only retain measured fluxes
wav_data = wav[flux.flatten() > 0.]
flux_data = flux[flux .flatten()> 0.].flatten()
eflux_data = eflux[flux.flatten() > 0.].flatten()
UL_data = UL[flux.flatten() > 0.]
filters_data = filters[flux.flatten() > 0.]

# The observed wavelengths [wav_data], observed fluxes [flux_data], and their uncertainties [eflux_data] are fed to iragnsep (INPUT).
# The module run_all contains functions that fits, compare the models, analyse the results and plots.
# In particular, since we have photometry data alone of mrk1066, we use the function fitPhoto of the module run_all.

# Keyword used here:
# z: redshift
# UL: indicates which of the fluxes need to be set as an upper limit in the fits. UL has to be of the same length as the data.
# 	  Here we define the flux at 160 micron as upper limit.
# Nmc: numbers of MCMC to perform. The more the better. 10000 is good for photometric fits.
# We know the value of S9p7, so we feed it to iragnsep
# saveRes:If set to True, save the plots and the tables at pathFig+sourceName_fitRes_photo.pdf and pathFig+sourceName_fitRes_photo.csv, respectively.
res, resBM = run_all.fitPhoto(wav_data, flux_data, eflux_data, \
							  filters, \
							  z = z, \
							  UL = UL, \
							  Nmc = 10000, \
							  S9p7 = S9p7, \
						  	  sourceName = 'IC5063', \
						  	  pathTable = './', \
						  	  pathFig = './', \
						  	  saveRes = True)

# OUTOUT
# res: dataframe containing the results of the fits.
# resBM: same as res but for the Best Model [BM] only.

# Print the best model SFR and the AIC weighted average SFR.
print('***************************')
print('The SFR of IC5063 is ', np.round(np.sum(res['wSFR']),2), 'Msun/yr.')
print('***************************')


