import os
import pandas as pd
import iragnsep
import numpy as np

from .func import basictests, get_prop
from .SEDanalysis import runSEDspecFit, runSEDphotFit
from .toolplot import plotFitSpec, plotFitPhoto
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy import constants as const

c = const.c.value
Lsun = const.L_sun.value
h = const.h.value
k = const.k_B.value

def fitSpec(wavSpec, fluxSpec, efluxSpec,\
			wavPhot, fluxPhot, efluxPhot, \
			filters, \
			z = -0.01,\
			ULPhot = [],\
			obsCorr = True,\
			S9p7_fixed = -99.,\
			Nmc = 10000, pgrbar = 1, \
			ExtCurve = 'iragnsep',\
			Pdust_sigm = 3., PPAH_sigm = 3., PPL_sigm = 3., PSi_sigm = 3., \
			Pbreak = [40., 10.], Palpha = [0., 1.], \
			sourceName = 'NoName', pathTable = './', pathFig = './', \
			redoFit = True, saveRes = True):


	"""
    This function fits observed SEDs with combined spectra and photometry. The observed wavelengths, fluxes and uncertainties on the fluxes are passed separately for the spectrum and the photometry.
    ------------
    :param wavSpec: observed wavelengths for the spectrum (in microns).
    :param fluxSpec: observed fluxes for the spectrum (in Jansky).
    :param efluxSpec: observed uncertainties on the fluxes for the spectrum (in Jansky).
    :param wavPhot: observed wavelengths for the photometry (in microns).
    :param fluxPhot: observed fluxes for the photometry (in Jansky).
    :param efluxPhot: observed uncertainties on the fluxes for the photometry (in Jansky).
    :param filters: names of the photometric filters to include in the fit.
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword ULPhot: vector of length Nphot, where Nphot is the number of photometric data. If any of the values is set to 1, the corresponding flux is set has an upper limit in the fit. Default = [].
    :keyword obsCorr: if set to True, iragnsep attempt to calculate the total silicate absorption at 9.7micron, and correct for obscuration. Default = True
    :keyword S9p7_fixed: can be used to pass a fixed value for the total silicate absorption at 9.7 micron. Default = -99.
    :keyword Nmc: numer of MCMC run. Default = 10000.
    :keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
    :keyword Pdust_sigm: width of the normal prior for the normalisation (in log) of the dust continuum template. Default = 3.
    :keyword PPAH_sigm: width of the normal prior for the normalisation (in log) of the PAH template. Default = 3.
    :keyword PPL_sigm: width of the normal prior for the normalisation (in log) of the power-law for the AGN model. Default = 3.
    :keyword PSi_sigm: width of the normal prior for the normalisation (in log) of the silicate emission. Default = 3.
	:keyword Pbreak: normal prior on the position of the break. Default = [40., 10.] ([mean, std dev]).
	:keyword Palpha: normal prior on the three slopes alpha of the AGN continuum model. Default = [0., 1.] ([mean, std dev]).
	:keyword sourceName: name of the source. Default = 'NoName'.
	:keyword pathTable: if saveRes is set to True, the tables containing the results of the fits will be saved at the location pathTable. Default = './'.
	:keyword pathFig: if saveRes is set to True, the figues showing the results of the fits will be saved at the location pathFig. Default = './'.
	:keyword redoFit: if set to True, re-performs the fits. Otherwise, finds the table saved at pathTable and reproduces the analysis and the figures only. Default = True.
	:keyword saveRes: if set to True, the tables containing the results of the fits as well as the figures are saved. Default = True.
	------------
    :return res_fit: dataframe containing the results of all the possible fits.
    :return res_fitBM: dataframe containing the results of the best fit only.
    """
	
	# test if the path for the tables exists. If not, raise ValueError, crash.
	if pathTable.endswith('/') == False:
		pathTable = pathTable+'/'
	if os.path.isdir(pathTable) == False:
		raise ValueError('The path '+pathTable+' to save the tables does not exist. Please create it.')

	# test if the path for the figures exists. If not, raise ValueError, crash.
	if pathFig.endswith('/') == False:
		pathFig = pathFig+'/'	
	if os.path.isdir(pathFig) == False:
		raise ValueError('The path '+pathFig+' to save the figures does not exist. Please create it.')

	# test that the name of the source is a string
	if isinstance(sourceName, str) == False:
		raise ValueError('The keyword sourceName should be a string as it will be used for reference to save files.')		
	
	# run basic tests to avoid crashing while fitting (see func.py file). 
	basictests(wavSpec, fluxSpec, efluxSpec, wavPhot, fluxPhot, efluxPhot, filters, z, specOn = True)
	z = abs(z)

	# open the galaxy and AGN templates.
	path = os.path.dirname(iragnsep.__file__)
	templ = pd.read_csv(path+'/iragnsep_templ.csv')

	# Test that the obscuration is not fixed
	if S9p7_fixed != -99.:
		if S9p7_fixed < 1e-3:
			S9p7_fixed = 0.
		if S9p7_fixed > 1e5:
			raise ValueError("S9p7_fixed has been set above to the maximum allowed value (i.e. 1e5). Please check.")

	# Test if the length of the upper-limit vector is that of the length of photometric point. If not, raise ValueError, crash.
	if (len(ULPhot) > 0) & (len(ULPhot) != (len(wavPhot))):
		raise ValueError("UPPER LIMITS ISSUE: Crashed because the vector UL for photometry has been passed but has not the same length as the photometric data.")

	# Calculate the central value of the priors for the normalisations based on the averaged FIR flux
	dMpc = cosmo.luminosity_distance(z).value #Mpc
	dmeter = dMpc*u.Mpc.to(u.m)
	d2z = dmeter**2./(1.+z) # K correction=>https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html

	Lnu = fluxPhot*1e-26 * 4. * np.pi * d2z/Lsun 
	logLtot = np.log10(np.trapz(Lnu, 3e8/wavPhot[::-1]/1e-6, dx = np.gradient(3e8/wavPhot[::-1]/1e-6))) # Lum in the FIR in the data

	Pdust = [logLtot+1, Pdust_sigm]
	PPAH = [0.97 * Pdust[0] - 0.95, PPAH_sigm]
	PPL = [np.log10(np.mean(fluxSpec)), PPL_sigm]
	PSi = [np.log10(np.interp(10., wavSpec, fluxSpec)), PSi_sigm]
	
	# If redoFit is set to True, re-performs the fits. Otherwise, try to open the table results and jump to calculating the IR properties and plotting the results.
	if redoFit == True:
		
		# run the SED fit (see SEDanalysis.py file).
		res_fit = runSEDspecFit(wavSpec, fluxSpec, efluxSpec,\
								wavPhot, fluxPhot, efluxPhot, \
								filters, \
								z = z,\
								ULPhot = ULPhot, \
								obsCorr = obsCorr,\
								S9p7_fixed = S9p7_fixed,\
								Nmc = Nmc, pgrbar = pgrbar, \
								ExtCurve = ExtCurve,\
								Pdust = Pdust, PPAH = PPAH, \
				 				PPL = PPL, Palpha = Palpha, \
								Pbreak = Pbreak, \
								PSi = PSi, \
								templ = templ)
	else:

		# If redoFit is not set to True, attempt to open the table containing the results of the fits. If failed, raise ValueError, crash.
		try:
		
			res_fit = pd.read_csv(pathTable+sourceName+'_fitRes_spec.csv')
		
		except:
		
			raise ValueError('Cannot find the table. Check the name or redo the fit.')

	# Prepare the upper limits. Stick together a vector of zeros if no upper-limits are provided.
	if len(ULPhot) != len(wavPhot):
		ULPhot = np.zeros(len(wavPhot))
	ULSpec = np.zeros(len(wavSpec))
	UL = np.concatenate([ULSpec, ULPhot])

	# Calculate the IR properties of the galaxy and the AGN.
	loglum_hostIR, eloglum_hostIR, \
	loglum_hostMIR, eloglum_hostMIR, \
	loglum_hostFIR, eloglum_hostFIR, \
	loglum_AGNIR, loglum_AGNMIR, loglum_AGNFIR, \
	AGNfrac_IR, AGNfrac_MIR, AGNfrac_FIR, SFR, eSFR, wSFR, ewSFR = get_prop(res_fit, templ = templ, z = z)

	# Flag the models without AGNs
	noAGN = np.where(res_fit['AGNon'].values == 0)[0]

	# Set default values to -99. for the final table.
	logNormGal_dust = res_fit['logNormGal_dust'].values
	logNormGal_dust = np.round(logNormGal_dust,3)
	res_fit['logNormGal_dust'] = logNormGal_dust

	elogNormGal_dust = res_fit['elogNormGal_dust'].values
	elogNormGal_dust = np.round(elogNormGal_dust,3)
	res_fit['elogNormGal_dust'] = elogNormGal_dust

	logNormGal_PAH = res_fit['logNormGal_PAH'].values
	logNormGal_PAH = np.round(logNormGal_PAH,3)
	res_fit['logNormGal_PAH'] = logNormGal_PAH

	elogNormGal_PAH = res_fit['elogNormGal_PAH'].values
	elogNormGal_PAH = np.round(elogNormGal_PAH,3)
	res_fit['elogNormGal_PAH'] = elogNormGal_PAH

	logNormAGN_PL = res_fit['logNormAGN_PL'].values
	logNormAGN_PL[noAGN] = -99.
	logNormAGN_PL = np.round(logNormAGN_PL,3)
	res_fit['logNormAGN_PL'] = logNormAGN_PL

	elogNormAGN_PL = res_fit['elogNormAGN_PL'].values
	elogNormAGN_PL[noAGN] = -99.
	elogNormAGN_PL = np.round(elogNormAGN_PL,3)
	res_fit['elogNormAGN_PL'] = elogNormAGN_PL

	lBreak_PL = res_fit['lBreak_PL'].values
	lBreak_PL[noAGN] = -99.
	lBreak_PL = np.round(lBreak_PL,3)
	res_fit['lBreak_PL'] = lBreak_PL

	elBreak_PL = res_fit['elBreak_PL'].values
	elBreak_PL[noAGN] = -99.
	elBreak_PL = np.round(elBreak_PL,3)
	res_fit['elBreak_PL'] = elBreak_PL

	alpha1 = res_fit['alpha1'].values
	alpha1[noAGN] = -99.
	alpha1 = np.round(alpha1,3)
	res_fit['alpha1'] = alpha1

	ealpha1 = res_fit['ealpha1'].values
	alpha1[noAGN] = -99.
	ealpha1 = np.round(ealpha1,3)
	res_fit['ealpha1'] = ealpha1

	alpha2 = res_fit['alpha2'].values
	alpha2[noAGN] = -99.
	alpha2 = np.round(alpha2,3)
	res_fit['alpha2'] = alpha2

	ealpha2 = res_fit['ealpha2'].values
	ealpha2[noAGN] = -99.
	ealpha2 = np.round(ealpha2,3)
	res_fit['ealpha2'] = ealpha2

	alpha3 = res_fit['alpha3'].values
	alpha3[noAGN] = -99.
	alpha3 = np.round(alpha3,3)
	res_fit['alpha3'] = alpha3

	ealpha3 = res_fit['ealpha3'].values
	ealpha3[noAGN] = -99.
	ealpha3 = np.round(ealpha3,3)
	res_fit['ealpha3'] = ealpha3

	logNorm_Si = res_fit['logNorm_Si'].values
	logNorm_Si[noAGN] = -99.
	logNorm_Si = np.round(logNorm_Si,3)
	res_fit['logNorm_Si'] = logNorm_Si

	elogNorm_Si = res_fit['elogNorm_Si'].values
	elogNorm_Si[noAGN] = -99.
	elogNorm_Si = np.round(elogNorm_Si,3)
	res_fit['elogNorm_Si'] = elogNorm_Si

	dSi = np.round(res_fit['dSi'],3)
	res_fit['dSi'] = dSi
	edSi = np.round(res_fit['edSi'],3)
	res_fit['edSi'] = edSi

	Aw = res_fit['Aw'].values
	o = np.where(Aw < 1e-3)[0]
	if len(o > 0.):
		Aw[o] = 0.0
	Aw = np.round(Aw, 3)
	res_fit['Aw'] = Aw

	logl = res_fit['logl'].values
	logl = np.round(logl,3)
	res_fit['logl'] = logl

	# Generate the final table
	try:
		res_fit['logLumIR_host'] = loglum_hostIR
		res_fit['elogLumIR_host'] = eloglum_hostIR
		res_fit['logLumMIR_host'] = loglum_hostMIR
		res_fit['elogLumMIR_host'] = eloglum_hostMIR
		res_fit['logLumFIR_host'] = loglum_hostFIR
		res_fit['elogLumFIR_host'] = eloglum_hostFIR
		res_fit['logLumIR_AGN'] = loglum_AGNIR
		res_fit['logLumMIR_AGN'] = loglum_AGNMIR
		res_fit['logLumFIR_AGN'] = loglum_AGNFIR
		res_fit['AGNfrac_IR'] = AGNfrac_IR
		res_fit['AGNfrac_MIR'] = AGNfrac_MIR
		res_fit['AGNfrac_FIR'] = AGNfrac_FIR
		res_fit['SFR'] = SFR
		res_fit['eSFR'] = eSFR
		res_fit['wSFR'] = wSFR
		res_fit['ewSFR'] = ewSFR
	except:
		res_fit['logLumIR_host'] = pd.Series(loglum_hostIR, index=res_fit.index)
		res_fit['elogLumIR_host'] = pd.Series(eloglum_hostIR, index=res_fit.index)
		res_fit['logLumMIR_host'] = pd.Series(loglum_hostMIR, index=res_fit.index)
		res_fit['elogLumMIR_host'] = pd.Series(eloglum_hostMIR, index=res_fit.index)
		res_fit['logLumFIR_host'] = pd.Series(loglum_hostFIR, index=res_fit.index)
		res_fit['elogLumFIR_host'] = pd.Series(eloglum_hostFIR, index=res_fit.index)
		res_fit['logLumIR_AGN'] = pd.Series(loglum_AGNIR, index=res_fit.index)
		res_fit['logLumMIR_AGN'] = pd.Series(loglum_AGNMIR, index=res_fit.index)
		res_fit['logLumFIR_AGN'] = pd.Series(loglum_AGNFIR, index=res_fit.index)
		res_fit['AGNfrac_IR'] = pd.Series(AGNfrac_IR, index=res_fit.index)
		res_fit['AGNfrac_MIR'] = pd.Series(AGNfrac_MIR, index=res_fit.index)
		res_fit['AGNfrac_FIR'] = pd.Series(AGNfrac_FIR, index=res_fit.index)
		res_fit['SFR'] = pd.Series(SFR, index=res_fit.index)
		res_fit['eSFR'] = pd.Series(eSFR, index=res_fit.index)
		res_fit['wSFR'] = pd.Series(wSFR, index=res_fit.index)
		res_fit['ewSFR'] = pd.Series(ewSFR, index=res_fit.index)

	# If saveRes is set to True, save the table
	if saveRes == True:
		order = ['tplName', 'AGNon', 'logNormGal_dust', 'elogNormGal_dust', 'logNormGal_PAH','elogNormGal_PAH', 'logNormAGN_PL', 'elogNormAGN_PL', 'lBreak_PL',\
				 'elBreak_PL', 'alpha1', 'ealpha1', 'alpha2', 'ealpha2', 'alpha3', 'ealpha3',\
				 'logNorm_Si', 'elogNorm_Si', 'dSi', 'edSi', \
				 'logLumIR_host', 'elogLumIR_host', 'logLumMIR_host', 'elogLumMIR_host', 'logLumFIR_host', 'elogLumFIR_host', 'logLumIR_AGN', \
				 'logLumMIR_AGN', 'logLumFIR_AGN', 'AGNfrac_IR', 'AGNfrac_MIR', 'AGNfrac_FIR', 'SFR','eSFR', 'wSFR', 'ewSFR', 'logl', \
				 'Aw', 'S9p7', 'bestModelFlag']

		res_fit.to_csv(pathTable+sourceName+'_fitRes_spec.csv', index = False, columns = order)

	print('#########################')
	print('# Generating the plots. #')
	print('#########################')
	# Plot all the fits
	wav = np.concatenate([wavSpec, wavPhot])
	flux = np.concatenate([fluxSpec, fluxPhot])
	eflux = np.concatenate([efluxSpec, efluxPhot])
	plotFitSpec(res_fit, wavSpec, fluxSpec, efluxSpec,\
				wavPhot, fluxPhot, efluxPhot,\
				UL = ULPhot, pathFig = pathFig, sourceName = sourceName, \
				templ = templ, z = z, saveRes = saveRes, ExtCurve = ExtCurve)

	# Select the best model
	o = np.where(res_fit['bestModelFlag'] == 1)[0]
	res_fitBM = res_fit.iloc[o]

	return res_fit, res_fitBM

def fitPhoto(wav, flux, eflux,\
			 filters, \
			 z = -0.01,\
			 UL = [], \
			 ExtCurve = 'iragnsep', \
			 S9p7 = -99.,\
			 Nmc = 10000, pgrbar = 1, \
			 NoSiem = False, \
			 Pdust = [10., 3.], PPAH = [9., 3.], PnormAGN = [10., 3.], PSiEm = [10., 3.], \
			 sourceName = 'NoName', pathTable = './', pathFig = './', \
			 redoFit = True, saveRes = True, \
			 NOAGN = False):

	"""
    This function fits the observed photometric SED.
    ------------
    :param wav: observed wavelengths (in microns).
    :param fluxSpec: observed fluxes (in Jansky).
    :param efluxSpec: observed uncertainties on the fluxes (in Jansky).
    :param filters: name of the photometric filters to include in the fit.
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword UL: vector of length Nphot, where Nphot is the number of photometric data. If any of the value is set to 1, the corresponding flux is set has an upper limit in the fit. Default = [].
	:keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
    :keyword S9p7: can be used to pass a fixed value for the total silicate absorption at 9.7 micron. Default = -99.
    :keyword Nmc: numer of MCMC run. Default = 10000.
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
	:keyword NoSiem: if set to True, no silicate emission template is included in the fit. Default = False.
    :keyword Pdust: normal prior on the log-normalisation of the galaxy dust continuum template. Default = [10., 3.] ([mean, std dev]).
    :keyword PPAH: normal prior on the log-normalisation of the PAH template. Default = [9., 3.] ([mean, std dev]).
    :keyword PnormAGN: normal prior on the log-normalisation of the AGN template. Default = [10., 3.] ([mean, std dev]).
    :keyword PSiem: normal prior on the log-normalisation of the silicate emission template. Default = [10., 3.] ([mean, std dev]).
    :keyword sourceName: name of the source. Default = 'NoName'.
	:keyword pathTable: if saveRes is set to True, the tables containing the results of the fits will be saved at the location pathTable. Default = './'.
	:keyword pathFig: if saveRes is set to True, the figues showing the results of the fits will be saved at the location pathFig. Default = './'.
	:keyword redoFit: if set to True, re-performs the fits. Otherwise, find the table saved at pathTable and reproduces the analysis and the figures only. Default = True.
	:keyword saveRes: if set to True, the tables containing the results of the fits as well as the figures are saved. Default = True.
	:keyword NOAGN: if set to True, fits are ran with SF templates only (i.e. no AGN emission is accounted for). Default = False.
	------------
    :return res_fit: dataframe containing the results of all the possible fits.
    :return res_fitBM: dataframe containing the results of the best fit only.
    """

	# test if the path for the tables exists. If not, raise ValueError, crash.
	if pathTable.endswith('/') == False:
		pathTable = pathTable+'/'	
	if os.path.isdir(pathTable) == False:
		raise ValueError('The path specified to save the tables does not exist. Please create it.')

	# test if the path for the figures exists. If not, raise ValueError, crash.
	if pathFig.endswith('/') == False:
		pathFig = pathFig+'/'	
	if os.path.isdir(pathFig) == False:
		raise ValueError('The path specified to save the figures does not exist. Please create it.')

	# test on the length of upper limits.
	if (len(UL)>0) & (len(UL) != len(wav)):
		raise ValueError('UPPER LIMITS: The length of the vector for the upper limits passed to fitPhoto does not match that of the number of photometric points.')		

	# if no UL vector is passed, then defined a vector of zeros
	if len(UL) != len(wav):
		UL = np.zeros(len(wav))

	# test if any of the photometry is detected:
	# if less than 3 photometry is used, or that only UL are defined, remove the AGN contribution (i.e. SF only).
	o = np.where(UL == 0.)[0]
	if (len(wav) < 3.) | (len(o) == 0.):
		NOAGN = True
		NoSiem = True
		
	# Test if any FIR data are included (rest wavelength > 50 microns). If not AGN contribution removed.
	o = np.where(wav/(1.+z) > 50.)[0]
	if len(o) == 0.:
		NOAGN = True
		NoSiem = True

	# if the AGN contribution is included run some basic tests.
	if NOAGN != True:
		# run basic tests to avoid crashing while fitting (see func.py file).
		basictests([], [], [], wav, flux, eflux, filters, z, specOn = False)
	z = abs(z)

	# If no silicate emission template are set for the fit, jump straight to the fit.
	if NoSiem == True:
		pass
	else:
		# If the silicate emission template is considered in the fit, test that there are enough photometric data to constrain it.
		# If not, remove the siliate emission template by setting NoSiem to True, and display a warning.
		SiRange = [int(9.*(1.+z)), int(20.*(1.+z))]
		o = np.where((wav>SiRange[0]) & (wav<SiRange[1]))[0]
		if len(o) == 0.:
			NoSiem = True
			print('The silicate emission template is excluded from the fit due to the lack of data points around the silicate emission features.')
			pass

		o = np.where(wav < 70.)[0]
		if (len(o) < 4) & (NoSiem == False):
			NoSiem = True
			print('The silicate emission template is excluded from the fit due to the lack of data points at shorter wavelengths.')
			pass

	# open the galaxy and AGN templates.
	path = os.path.dirname(iragnsep.__file__)
	templ = pd.read_csv(path+'/iragnsep_templ.csv')

	# If redoFit is set to True, re-performs the fits. Otherwise, try to open the table results and jump to calculating the IR properties and plotting the results.
	if redoFit == True:
		# run the SED fit (see SEDanalysis.py file).
		res_fit = runSEDphotFit(wav, flux, eflux,\
								z = z,\
								filters = filters, \
								UL = UL, \
								S9p7 = S9p7,\
								ExtCurve = ExtCurve, \
								Nmc = Nmc, pgrbar = pgrbar, \
								NoSiem = NoSiem, \
								Pdust = Pdust, PPAH = PPAH, PnormAGN = PnormAGN, PSiEm = PSiEm,\
								templ = templ, \
								NOAGN = NOAGN)
	else:
		try:
			# If redoFit is not set to True, attempt to open the table containing the results of the fits. If failed, raise ValueError, crash.
			res_fit = pd.read_csv(pathTable+sourceName+'_fitRes_photo.csv')
		except:
			raise ValueError('Cannot find the table. Check the name or redo the fit.')

	# Calculate the IR properties of the galaxy and the AGN.
	loglum_hostIR, eloglum_hostIR, \
	loglum_hostMIR, eloglum_hostMIR, \
	loglum_hostFIR, eloglum_hostFIR, \
	loglum_AGNIR, loglum_AGNMIR, loglum_AGNFIR, \
	AGNfrac_IR, AGNfrac_MIR, AGNfrac_FIR, SFR, eSFR, wSFR, ewSFR = get_prop(res_fit, templ = templ, z = z, specOn = False)

	# Set default values to -99. for the final table.
	logNormGal_dust = res_fit['logNormGal_dust'].values
	logNormGal_dust = np.round(logNormGal_dust,3)
	res_fit['logNormGal_dust'] = logNormGal_dust

	elogNormGal_dust = res_fit['elogNormGal_dust'].values
	elogNormGal_dust = np.round(elogNormGal_dust,3)
	res_fit['elogNormGal_dust'] = elogNormGal_dust

	logNormGal_PAH = res_fit['logNormGal_PAH'].values
	o = np.where(logNormGal_PAH == -11.)[0]
	logNormGal_PAH[o] = -99.
	logNormGal_PAH = np.round(logNormGal_PAH,3)
	res_fit['logNormGal_PAH'] = logNormGal_PAH

	elogNormGal_PAH = res_fit['elogNormGal_PAH'].values
	o = np.where(elogNormGal_PAH == 0.)[0]
	elogNormGal_PAH[o] = -99.
	elogNormGal_PAH = np.round(elogNormGal_PAH,3)
	res_fit['elogNormGal_PAH'] = elogNormGal_PAH

	logNormAGN = res_fit['logNormAGN'].values
	o = np.where(logNormAGN == -89.)[0]
	logNormAGN[o] = -99.
	logNormAGN = np.round(logNormAGN,3)
	res_fit['logNormAGN'] = logNormAGN

	elogNormAGN = res_fit['elogNormAGN'].values
	elogNormAGN = np.round(elogNormAGN,3)
	res_fit['elogNormAGN'] = elogNormAGN

	logNormSiem = res_fit['logNormSiem'].values
	o = np.where(logNormSiem == -89.)[0]
	logNormSiem[o] = -99.
	o = np.where(logNormSiem == -10.)[0]
	logNormSiem[o] = -99.
	logNormSiem = np.round(logNormSiem,3)
	res_fit['logNormSiem'] = logNormSiem

	elogNormSiem = res_fit['elogNormSiem'].values
	o = np.where(elogNormSiem == 0.)[0]
	elogNormSiem[o] = -99.
	elogNormSiem = np.round(elogNormSiem,3)
	res_fit['elogNormSiem'] = elogNormSiem

	Aw = res_fit['Aw'].values
	o = np.where(Aw < 1e-3)[0]
	Aw[o] = 0.0
	Aw = np.round(Aw, 3)
	res_fit['Aw'] = Aw

	# Generate the final table
	try:
		res_fit['logLumIR_host'] = loglum_hostIR
		res_fit['elogLumIR_host'] = eloglum_hostIR
		res_fit['logLumMIR_host'] = loglum_hostMIR
		res_fit['elogLumMIR_host'] = eloglum_hostMIR
		res_fit['logLumFIR_host'] = loglum_hostFIR
		res_fit['elogLumFIR_host'] = eloglum_hostFIR
		res_fit['logLumIR_AGN'] = loglum_AGNIR
		res_fit['logLumMIR_AGN'] = loglum_AGNMIR
		res_fit['logLumFIR_AGN'] = loglum_AGNFIR
		res_fit['AGNfrac_IR'] = AGNfrac_IR
		res_fit['AGNfrac_MIR'] = AGNfrac_MIR
		res_fit['AGNfrac_FIR'] = AGNfrac_FIR
		res_fit['SFR'] = SFR
		res_fit['eSFR'] = eSFR
		res_fit['wSFR'] = wSFR
		res_fit['ewSFR'] = ewSFR
	except:
		res_fit['logLumIR_host'] = pd.Series(loglum_hostIR, index=res_fit.index)
		res_fit['elogLumIR_host'] = pd.Series(eloglum_hostIR, index=res_fit.index)
		res_fit['logLumMIR_host'] = pd.Series(loglum_hostMIR, index=res_fit.index)
		res_fit['elogLumMIR_host'] = pd.Series(eloglum_hostMIR, index=res_fit.index)
		res_fit['logLumFIR_host'] = pd.Series(loglum_hostFIR, index=res_fit.index)
		res_fit['elogLumFIR_host'] = pd.Series(eloglum_hostFIR, index=res_fit.index)
		res_fit['logLumIR_AGN'] = pd.Series(loglum_AGNIR, index=res_fit.index)
		res_fit['logLumMIR_AGN'] = pd.Series(loglum_AGNMIR, index=res_fit.index)
		res_fit['logLumFIR_AGN'] = pd.Series(loglum_AGNFIR, index=res_fit.index)
		res_fit['AGNfrac_IR'] = pd.Series(AGNfrac_IR, index=res_fit.index)
		res_fit['AGNfrac_MIR'] = pd.Series(AGNfrac_MIR, index=res_fit.index)
		res_fit['AGNfrac_FIR'] = pd.Series(AGNfrac_FIR, index=res_fit.index)
		res_fit['SFR'] = pd.Series(SFR, index=res_fit.index)
		res_fit['eSFR'] = pd.Series(eSFR, index=res_fit.index)
		res_fit['wSFR'] = pd.Series(wSFR, index=res_fit.index)
		res_fit['ewSFR'] = pd.Series(ewSFR, index=res_fit.index)

	# If saveRes is set to True, save the table
	if saveRes == True:
		order = ['tplName_gal', 'AGNon', 'tplName_AGN', 'logNormGal_dust', 'elogNormGal_dust', 'logNormGal_PAH','elogNormGal_PAH', 'logNormAGN', 'elogNormAGN',\
				 'logNormSiem', 'elogNormSiem', 'logLumIR_host', 'elogLumIR_host', 'logLumMIR_host', 'elogLumMIR_host', 'logLumFIR_host', \
				 'elogLumFIR_host', 'logLumIR_AGN', 'logLumMIR_AGN', 'logLumFIR_AGN', 'AGNfrac_IR', 'AGNfrac_MIR', 'AGNfrac_FIR', 'SFR','eSFR', \
				 'wSFR', 'ewSFR', 'logl', 'Aw', 'S9p7', 'bestModelFlag']
		res_fit.to_csv(pathTable+sourceName+'_fitRes_photo.csv', index = False, columns = order)

	print('#########################')
	print('# Generating the plots. #')
	print('#########################')
	# Plot all the fits
	plotFitPhoto(res_fit, wav, flux, eflux, UL = UL, pathFig = pathFig, sourceName = sourceName, templ = templ, z = z, saveRes = saveRes, NOAGN = NOAGN)

	# Select the best model
	o = np.where(res_fit['bestModelFlag'] == 1)[0]
	res_fitBM = res_fit.iloc[o]

	return res_fit, res_fitBM


