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
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
    :keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
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

	# Test if fluxes at wavelengths longer than 100 microns are only constrained by upper limits.
	# If yes, returns SFR as an upper limit.
	SFR_UL = False
	fluxPhot_fit = fluxPhot.copy()
	efluxPhot_fit = efluxPhot.copy()
	ULPhot_fit = ULPhot.copy()

	o = np.where(wavPhot/(1.+z) >= 100.)[0]
	if len(o) == 0:
		print('Not enough detected FIR data. Deriving upper-limits only.')
		SFR_UL = True
		ULPhot_fit[-1] = 0.
	elif ULPhot[o].sum() == len(o):
		print('Not enough detected FIR data. Deriving upper-limits only.')
		o = np.where(wavPhot/(1.+z) >= 70.)[0]
		ULPhot_fit[o[fluxPhot_fit[o] == fluxPhot_fit[o].min()]] = 0.
		efluxPhot_fit[o[fluxPhot_fit[o] == fluxPhot_fit[o].min()]] = fluxPhot[o[fluxPhot_fit[o] == fluxPhot_fit[o].min()]] * 0.1
		SFR_UL = True

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
								wavPhot, fluxPhot_fit, efluxPhot_fit, \
								filters, \
								z = z,\
								ULPhot = ULPhot_fit, \
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
	AGNfrac_IR, AGNfrac_MIR, AGNfrac_FIR, SFR, eSFR = get_prop(res_fit, templ = templ, z = z)

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

	keys = res_fit.keys()
	for i, key in enumerate(keys):
		if key.startswith('e'):
			parms = res_fit[keys[i-1]].values
			eparms = res_fit[key].values

			parm_r = []
			eparm_r = []

			for parm, eparm in zip(parms,eparms):
				if eparm == -99.:
					parm_r.append(-99.)
					eparm_r.append(-99.)
					continue
				else:
					r_i = 0
					eparm_round = 0.
					while eparm_round == 0.:
						eparm_round = np.round(eparm,r_i)
						r_i += 1

					r_i += 2
					parm_r.append(np.round(parm,r_i))
					eparm_r.append(np.round(eparm,r_i))

			res_fit[keys[i-1]] = parm_r
			res_fit[key] = eparm_r


	logl = res_fit['logl']
	logl = np.round(logl, 3)
	res_fit['logl'] = logl

	Aw = res_fit['Aw'].values
	o = np.where(Aw < 1e-3)[0]
	Aw[o] = 0.0
	Aw = np.round(Aw, 3)
	res_fit['Aw'] = Aw

	S9p7 = res_fit['S9p7']
	S9p7 = np.round(S9p7, 3)
	res_fit['S9p7'] = S9p7

	if SFR_UL == True:
		SFR = res_fit['SFR'].values * -1.
		eSFR = np.zeros(len(SFR)) - 99.
		res_fit['SFR'] = SFR
		res_fit['eSFR'] = eSFR


	# If saveRes is set to True, save the table
	if saveRes == True:
		order = ['tplName', 'AGNon', 'logNormGal_dust', 'elogNormGal_dust', 'logNormGal_PAH','elogNormGal_PAH', 'logNormAGN_PL', 'elogNormAGN_PL', 'lBreak_PL',\
				 'elBreak_PL', 'alpha1', 'ealpha1', 'alpha2', 'ealpha2', 'alpha3', 'ealpha3',\
				 'logNorm_Si', 'elogNorm_Si', 'dSi', 'edSi', \
				 'logLumIR_host', 'elogLumIR_host', 'logLumMIR_host', 'elogLumMIR_host', 'logLumFIR_host', 'elogLumFIR_host', 'logLumIR_AGN', \
				 'logLumMIR_AGN', 'logLumFIR_AGN', 'AGNfrac_IR', 'AGNfrac_MIR', 'AGNfrac_FIR', 'SFR','eSFR', 'logl', \
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
				templ = templ, z = z, saveRes = saveRes, ExtCurve = ExtCurve, SFR_UL = SFR_UL)

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
			 Pdust_sigm = 1., PPAH_sigm = 1., PnormAGN_sigm = 1., PSiem_sigm = 1., \
			 sourceName = 'NoName', pathTable = './', pathFig = './', \
			 redoFit = True, saveRes = True, \
			 NOAGN = False):

	"""
    This function fits the observed photometric SED.
    ------------
    :param wav: observed wavelengths (in microns).
    :param flux: observed fluxes (in Jansky).
    :param eflux: observed uncertainties on the fluxes (in Jansky).
    :param filters: name of the photometric filters to include in the fit.
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword UL: vector of length Nphot, where Nphot is the number of photometric data. If any of the value is set to 1, the corresponding flux is set has an upper limit in the fit. Default = [].
	:keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
    :keyword S9p7: can be used to pass a fixed value for the total silicate absorption at 9.7 micron. Default = -99.
    :keyword Nmc: numer of MCMC run. Default = 10000.
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
	:keyword NoSiem: if set to True, no silicate emission template is included in the fit. Default = False.
    :keyword Pdust_sigm: width of the normal prior for the normalisation (in log) of the dust continuum template. Default = 1.
    :keyword PPAH_sigm: width of the normal prior for the normalisation (in log) of the PAH template. Default = 1.
    :keyword PnormAGN_sigm: width of the normal prior for the normalisation (in log) of the AGN continuum template. Default = 1.
    :keyword PSiem_sigm: width of the normal prior for the normalisation (in log) of the silicate emission template. Default = 1.
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

	SFR_UL = False
	flux_fit = flux.copy()
	eflux_fit = eflux.copy()
	UL_fit = UL.copy()
 
 	# Check the IR bands to remove the AGN contribution when necessary, and return SFRs are upper limits.
	o = np.where(wav/(1.+z) <= 40.)[0]
	if len(o) == 0:
		print('Not enough MIR data. Ignoring AGN and deriving upper-limits only.')
		UL_fit[wav/(1.+z) < 70.] = 1.
		NOAGN = True
		SFR_UL = True

	o = np.where(wav/(1.+z) >= 100.)[0]
	if len(o) == 0:
		print('Not enough detected FIR data. Deriving upper-limits only.')
		SFR_UL = True
		UL_fit[-1] = 0.
	elif UL[o].sum() == len(o):
		print('Not enough detected FIR data. Deriving upper-limits only.')
		o = np.where(wav/(1.+z) >= 70.)[0]
		UL_fit[o[flux_fit[o] == flux_fit[o].min()]] = 0.
		eflux_fit[o[flux_fit[o] == flux_fit[o].min()]] = flux[o[flux_fit[o] == flux_fit[o].min()]] * 0.1
		SFR_UL = True

	o = np.where(wav/(1.+z) <= 70.)[0]
	if (UL[o].sum() == len(o)) & (NOAGN == False):
		print('Not enough detected MIR data. Ignoring AGN and deriving upper-limits only.')
		NOAGN = True
		SFR_UL = True

	if (len(wav) < 4) & (SFR_UL == False):
		print('Not enough detected MIR data. Ignoring AGN and deriving upper-limits only.')
		NOAGN = True
		SFR_UL = True

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

	# Test if it remains some non valid values
	if any(eflux_fit[UL_fit == 0.] <= 0.) | any(flux_fit[UL_fit == 0.] <= 0.):
		raise ValueError("PHOTOMETRY ISSUE: Crashed because one or some of the detected fluxes or uncertainties have non-physical values (i.e. negative or 0).")

	# open the galaxy and AGN templates.
	path = os.path.dirname(iragnsep.__file__)
	templ = pd.read_csv(path+'/iragnsep_templ.csv')

	# Calculate the central value of the priors for the normalisations based on the averaged FIR flux
	dMpc = cosmo.luminosity_distance(z).value #Mpc
	dmeter = dMpc*u.Mpc.to(u.m)
	d2z = dmeter**2./(1.+z) # K correction=>https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html

	Lnu = flux*1e-26 * 4. * np.pi * d2z/Lsun
	logLtot = np.log10(np.trapz(Lnu, 3e8/wav[::-1]/1e-6, dx = np.gradient(3e8/wav[::-1]/1e-6))) # Lum in the FIR in the data

	Pdust = [logLtot, Pdust_sigm]
	PPAH = [0.97 * Pdust[0] - 0.95, PPAH_sigm]
	PnormAGN = [logLtot, PnormAGN_sigm]
	PSiEm = [logLtot, PnormAGN_sigm]

	# If redoFit is set to True, re-performs the fits. Otherwise, try to open the table results and jump to calculating the IR properties and plotting the results.
	if redoFit == True:
		# run the SED fit (see SEDanalysis.py file).
		res_fit = runSEDphotFit(wav, flux_fit, eflux_fit,\
								z = z,\
								filters = filters, \
								UL = UL_fit, \
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

	if NOAGN == False:
		if np.sum(res_fit['logNormSiem'] - PSiEm[0]) == -99. * len(res_fit):
			logNormSiem = res_fit['logNormSiem'].values*0. -99.
			res_fit['logNormSiem'] = logNormSiem

	# Calculate the IR properties of the galaxy and the AGN.
	loglum_hostIR, eloglum_hostIR, \
	loglum_hostMIR, eloglum_hostMIR, \
	loglum_hostFIR, eloglum_hostFIR, \
	loglum_AGNIR, loglum_AGNMIR, loglum_AGNFIR, \
	AGNfrac_IR, AGNfrac_MIR, AGNfrac_FIR, SFR, eSFR = get_prop(res_fit, templ = templ, z = z, specOn = False, NOAGN = NOAGN)

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

	keys = res_fit.keys()
	for i, key in enumerate(keys):
		if key.startswith('e'):
			parms = res_fit[keys[i-1]].values
			eparms = res_fit[key].values

			parm_r = []
			eparm_r = []

			for parm, eparm in zip(parms,eparms):
				if eparm == -99.:
					parm_r.append(-99.)
					eparm_r.append(-99.)
					continue
				else:
					r_i = 0
					eparm_round = 0.
					while eparm_round == 0.:
						eparm_round = np.round(eparm,r_i)
						r_i += 1

					r_i += 2
					parm_r.append(np.round(parm,r_i))
					eparm_r.append(np.round(eparm,r_i))

			res_fit[keys[i-1]] = parm_r
			res_fit[key] = eparm_r


	logl = res_fit['logl']
	logl = np.round(logl, 3)
	res_fit['logl'] = logl

	Aw = res_fit['Aw'].values
	o = np.where(Aw < 1e-3)[0]
	Aw[o] = 0.0
	Aw = np.round(Aw, 3)
	res_fit['Aw'] = Aw

	S9p7 = res_fit['S9p7']
	S9p7 = np.round(S9p7, 3)
	res_fit['S9p7'] = S9p7

	if SFR_UL == True:
		SFR = res_fit['SFR'].values * -1.
		eSFR = np.zeros(len(SFR)) - 99.
		res_fit['SFR'] = SFR
		res_fit['eSFR'] = eSFR

	# If saveRes is set to True, save the table
	if (saveRes == True) & (NOAGN == False):
		order = ['tplName_gal', 'AGNon', 'tplName_AGN', 'logNormGal_dust', 'elogNormGal_dust', 'logNormGal_PAH','elogNormGal_PAH', 'logNormAGN', 'elogNormAGN',\
				 'logNormSiem', 'elogNormSiem', 'logLumIR_host', 'elogLumIR_host', 'logLumMIR_host', 'elogLumMIR_host', 'logLumFIR_host', \
				 'elogLumFIR_host', 'logLumIR_AGN', 'logLumMIR_AGN', 'logLumFIR_AGN', 'AGNfrac_IR', 'AGNfrac_MIR', 'AGNfrac_FIR', 'SFR','eSFR', \
				 'logl', 'Aw', 'S9p7', 'bestModelFlag']
		res_fit.to_csv(pathTable+sourceName+'_fitRes_photo.csv', index = False, columns = order)
	if (saveRes == True) & (NOAGN == True):
		order = ['tplName_gal', 'logNormGal_dust', 'elogNormGal_dust', 'logNormGal_PAH','elogNormGal_PAH', \
				 'logLumIR_host', 'elogLumIR_host', 'logLumMIR_host', 'elogLumMIR_host', 'logLumFIR_host', \
				 'elogLumFIR_host', 'SFR','eSFR', 'logl', 'Aw', 'S9p7', 'bestModelFlag']
		res_fit.to_csv(pathTable+sourceName+'_fitRes_photo.csv', index = False, columns = order)

	print('#########################')
	print('# Generating the plots. #')
	print('#########################')
	# Plot all the fits
	plotFitPhoto(res_fit, wav, flux, eflux, UL = UL, pathFig = pathFig, sourceName = sourceName, templ = templ, z = z, saveRes = saveRes, NOAGN = NOAGN, SFR_UL = SFR_UL)

	# Select the best model
	o = np.where(res_fit['bestModelFlag'] == 1)[0]
	res_fitBM = res_fit.iloc[o]

	return res_fit, res_fitBM