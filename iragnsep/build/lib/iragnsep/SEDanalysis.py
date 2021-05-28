import numpy as np
import pandas as pd
import os
import emcee
import math
import numba
import iragnsep
import astropy

from emcee import EnsembleSampler
from .func import S9p7toTau9p7, getExtCurve, calc_S9p7, contmodel, Simodel, nuLnuToFnu, exctractBestModel
from .classes import modelToSED
from numba import njit, vectorize
from astropy import stats
from scipy.stats import truncnorm

import matplotlib.pyplot as plt

#########################################################
#														#
#		SPECTROPHOTOMETRIC VERSION OF THE FITTING		#
#														#
#########################################################
@vectorize(nopython=True)
def nberf(x):
	"""
    Returns the erf of x in the  Numba environment.
    ------------
    :param x: x-value
    ------------
    :return erf(x): speaks for itself.
    """
	return math.erf(x)

# Numba wrapper
@njit(fastmath=True)
def lnpostfn_spec_noAGN(theta, P, modelDust, modelPAH, y, ey, UL, wei):
	
	"""
    This function calculates the log-likelihood between spectral + photometric data and model without AGN contribution.
    ------------
    :param theta: vector containing the parameters.
    :param P: vector containing the priors for each of the parameters.
    :param modelDust: model dust continuum template.
    :param modelPAH: model PAH template.
    :param y: observed fluxes.
    :param ey: uncertainties on the observed fluxes.
    :param UL: vector contaning the upper-limits, if any.
    :param wei: weighting of the data points.
    ------------
    :return logl: log-likelihood of the model knowing the parameters theta.
    """

    # Set a variable for the loglikelihood
	logl = 0

	# Prior constraint on the dust normalisation
	logl += -0.5*(theta[0]/P[0][1])**2.

	# Prior constraint on the PAH emission
	logl += -0.5*(theta[1]/P[1][1])**2.

	# Constraint of 1 dex between the PAH and the dust continuum
	logl += -0.5*((theta[1] + P[1][0] - (0.97 * (theta[0] + P[0][0]) - 0.95))/0.3)**2.

	# Define the model knowing the parameters theta
	ym = 10**(theta[0] + P[0][0]) * modelDust + 10**(theta[1] + P[1][0]) * modelPAH
	
	# Upper Limits
	x = (ym[UL == 1.] - y[UL == 1.])/y[UL == 1.]/np.sqrt(2) * 3.
	cdf = np.prod((1. - (0.5 * (1. + nberf(x))))*wei[UL == 1.])

	if cdf == 0.:	
		logl += -1e10
	else:
		logl += np.log(cdf)

	# Detected fluxes
	logl += np.sum(-0.5 * ((y[UL == 0.] - ym[UL == 0.])/(ey[UL == 0.]))**2. * wei[UL == 0.])

	return logl

# Numba wrapper
@njit(fastmath=True)
def lnpostfn_spec_wAGN(theta, P, modelDust, modelPAH, modelSi, y, ey, UL, wei, z, wavFit, obsC, x_red):

	"""
    This function calculates the log-likelihood between spectral + photometric data and model that includes AGN contribution.
    ------------
    :param theta: vector containing the parameters.
    :param P: vector containing the priors for each of the parameters.
    :param modelDust: dust continuum template.
    :param modelPAH: PAH template.
    :param modelSi: Silicate emission template.
    :param y: observed fluxes.
    :param ey: uncertainties on the observed fluxes.
    :param UL: vector contaning the upper-limits, if any.
    :param wei: weighting of the data points.
    :param z: redshift.
    :param wavFit: observed wavelengths.
    :param obsC: vector containing the values of the extinction for the AGN.
    :param x_red: reduced vector for the observed wavelengths.
    ------------
    :return logl: log-likelihood of the model knowing the parameters theta.
    """

	# set a variable for the log-likelihood
	logl = 0

	# Normal prior on the norm of the dust
	logl += -0.5*(theta[0]/P[0][1])**2.

	# Normal prior on the norm of the PAH
	logl += -0.5*(theta[1]/P[1][1])**2.

	# Constraint of 1 dex between the PAH and the dust continuum
	logl += -0.5*((theta[1] + P[1][0] - (0.97 * (theta[0] + P[0][0]) - 0.95))/0.3)**2.

	# Normal prior on the AGN power law
	logl += -0.5*(theta[2]/P[2][1])**2.

	# Normal prior on alpha 1
	if -3.5 < theta[3] + P[3][0] < 3.5:
		logl += -0.5*(theta[3]/P[3][1])**2.
	else:
		return -np.inf

	# Normal prior on alpha 2
	if -3.5 < theta[4] + P[3][0] < 3.5:
		logl += -0.5*(theta[4]/P[3][1])**2.
	else:
		return -np.inf

	# # Normal prior on alpha 3
	if -3.5 < theta[5] + P[3][0] < 1.:
		logl += -0.5*(theta[5]/P[3][1])**2.
	else:
		return -np.inf

	# Normal Prior on the normalisation of the Si emission
	logl += -0.5*(theta[6]/P[4][1])**2.

	# Normal Prior on the Si emission shift
	if -1. < theta[7] < 1.:
		logl += -0.5*(theta[7]/1.)**2.
	else:
		return -np.inf

	# Prior on the position of the breaks
	if 30. <= theta[8]  + P[5][0] <= 500.:
		logl += -0.5*(theta[8]/P[5][1])**2.
	else:
		return -np.inf

	# Calculate the AGN model given the parameters theta
	# To increase the speed, the AGN model is calculated on a reduced wavelength vector which only probes the main features (i.e. breaks and peaks).
	# The linear parts are interpolated.
	lbreak_red = 10**np.arange(np.log10(max(30., theta[8] + P[5][0] - 10.)), np.log10(max(30., theta[8] + P[5][0] + 20.)), 0.05)
	lbreak_red[-1] = wavFit[-1]
	x_red_wlbreak = np.zeros(len(x_red)+len(lbreak_red))

	x_red_wlbreak[:len(x_red)] = x_red
	x_red_wlbreak[len(x_red):] = lbreak_red

	modelPL = 10**(theta[2] + P[2][0]) * contmodel(np.array([theta[8] + P[5][0], theta[3] + P[3][0], theta[4] + P[3][0], theta[5] + P[3][0]]) , x_red_wlbreak)

	# Shifted silicate emission
	modelSi_shift = 10**(theta[6] + P[4][0]) * np.interp(wavFit - theta[7], wavFit, modelSi)

	# Interpolate at the observed wavelengths
	modelAGNFit = 10**np.interp(np.log10(wavFit/(1.+z)), np.log10(x_red_wlbreak), np.log10(modelPL)) + modelSi_shift

	# Calculate the model for the AGN+galaxy given parameters theta.
	ym = modelAGNFit * obsC + 10**(theta[0] + P[0][0]) * modelDust + 10**(theta[1] + P[1][0]) * modelPAH

	# Upper Limits
	x = (ym[UL == 1.] - y[UL == 1.])/y[UL == 1.]/np.sqrt(2) * 3.
	cdf = np.prod((1. - (0.5 * (1. + nberf(x))))*wei[UL == 1.])

	if cdf == 0.:	
		logl += -1e10
	else:
		logl += np.log(cdf)

	# Detected fluxes
	logl += np.sum(-0.5 * ((y[UL == 0.] - ym[UL == 0.])/(ey[UL == 0.]))**2. * wei[UL == 0.])

	return logl

def runSEDspecFit(wavSpec, fluxSpec, efluxSpec,\
				  wavPhot, fluxPhot, efluxPhot, \
				  filters, \
				  z = -0.01,\
				  ULPhot = [], \
				  obsCorr = True,\
				  S9p7_fixed = -99., \
				  ExtCurve = 'iragnsep',\
				  Nmc = 10000, pgrbar = 1, \
				  Pdust = [11., 3.], PPAH = [9.7, 3.], \
				  PPL = [-1., 3.], Palpha = [0., 1.], \
				  Pbreak = [40., 10.], \
				  PSi = [-1., 3.], \
				  templ = ''):
	"""
    This function fits the observed SED when a spectrum is combined to photometric data. The observed wavelengths, 
    fluxes and uncertainties on the fluxes are passed separately for the spectrum and the photometric data.
    ------------
    :param wavSpec: observed wavelengths for the spectrum (in microns).
    :param fluxSpec: observed fluxes for the spectrum (in Jansky).
    :param efluxSpec: observed uncertainties on the fluxes for the spectrum (in Jansky).
    :param wavPhot: observed wavelengths for the photometry (in microns).
    :param fluxPhot: observed fluxes for the photometry (in Jansky).
    :param efluxPhot: observed uncertainties on the fluxes for the photometry (in Jansky).
    :param filters: name of the photometric filters to include in the fit.
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword ULPhot: vector of length Nphot, where Nphot is the number of photometric data. If any of the value is set to 1, 
    the corresponding flux is set has an upper limit in the fit. Default = [].
    :keyword obsCorr: if set to True, iragnsep attempt to calculate the total silicate absorption at 9.7micron, 
    and to correct the observed fluxes for obscuration. Default = True
    :keyword S9p7_fixed: can be used to pass a fixed value for the total silicate absorption at 9.7 micron. Default = -99.
    :keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
    :keyword Nmc: numer of MCMC run. Default = 10000.
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
    :keyword Pdust: normal prior on the log-normalisation of the galaxy dust continuum template. Default = [11., 3.] ([mean, std dev]).
    :keyword PPAH: normal prior on the log-normalisation of the PAH template. Default = [9.7, 3.] ([mean, std dev]).
	:keyword Ppl: normal prior on the log-normalisation AGN continuum (defined at 10 micron). Default = [-1., 3.] ([mean, std dev]).
	:keyword Palpha: normal prior on the three slopes alpha of the AGN continuum model. Default = [0., 1.] ([mean, std dev]).
	:keyword Pbreak: prior on lbreak, the position of the break. Default = [40., 10.] ([mean, std dev]).
	:keyword PSi: prior on silicate emission. Default = [-1., 3.] ([mean, std dev]).
	:keyword templ: the templates used for the fits. Default: iragnsep templates.
	------------
    :return res_fit: dataframe containing the results of all the possible fits.
    :return res_fitBM: dataframe containing the results of the best fit only.
    """

	path_iragnsep = os.path.dirname(iragnsep.__file__)

	# Extract the names of the templates
	keys = templ.keys().values
	nameTempl_gal = []
	nameTempl_PAH = []
	for key in keys:
		if str(key).startswith('gal'):
			if str(key).endswith('PAH') == False:
				nameTempl_gal.append(key)
			else:
				nameTempl_PAH.append(key)

	# Test that we have template for everything (if no galaxy then it crashes)
	if len(nameTempl_gal) == 0:
		raise ValueError('The galaxy template does not exist. The name of the column defining nuLnu for the galaxy template needs to start with "gal".')

	# define the wavelengths
	try:
		wavTempl = templ['lambda_mic'].values
	except:
		raise ValueError('Rename the wavelengths column of the template "lambda_mic".')

	# Define the rest wavelengths for the photometric and the spectral data
	wavSpec_rest = wavSpec/(1. + z)
	wavPhot_rest = wavPhot/(1. + z)

	# Prepare the upper limits. Set a vector of zeros for the spectra. Set a vector fo zeros for the photometry if UL is underfined.
	if len(ULPhot) != len(wavPhot):
		ULPhot = np.zeros(len(wavPhot))
	ULSpec = np.zeros(len(wavSpec))
	UL = np.concatenate([ULSpec, ULPhot])

	# Concatenate the spectral and the photometric data.
	wavFit = np.concatenate([wavSpec, wavPhot])
	fluxFit = np.concatenate([fluxSpec, fluxPhot])
	efluxFit = np.concatenate([efluxSpec, efluxPhot])

	# Calculate the weights
	e_av_phot = np.mean(efluxPhot[ULPhot == 0.]/fluxPhot[ULPhot == 0.])
	e_av_spec = np.mean(efluxSpec[ULSpec == 0.]/fluxSpec[ULSpec == 0.])
	e_av_rat = e_av_spec/e_av_phot

	xrat = [1., 2., 4., 6.]
	yrat = [1., 7., 19., 30.]
	fit = np.polyfit(xrat, yrat, 1)

	multFact = max((fit[0] * e_av_rat + fit[1]), 1.)

	Specwei = np.zeros(len(wavSpec)) + (multFact)/len(wavSpec)
	Photwei = np.ones(len(wavPhot))
	wei = np.concatenate([Specwei, Photwei])

	xe = [0.011, 0.017, 0.034, 0.069]
	ye = [0.5, 1., 5., 15.]
	fit = np.polyfit(xe, ye, 2)

	wei *= max(fit[0]*e_av_phot**2. + fit[1]*e_av_phot + fit[2], 1.)

	# Correct for absorption, using the total 9.7micron absorption feature.
	if S9p7_fixed == -99.:
		# Test if there are actually data in the wavelength range, if not, continue the fit without correcting for obscuration
		o = np.where((wavSpec_rest > 9.) & (wavSpec_rest < 10.))[0]
		if (len(o) == 0.) & (obsCorr == True):
			print('*******************')
			print('It has failed to correct for obscuration. There is no data in the range required to correct for obscuration.' + \
				  ' The fit is continued without correcting for obscuration.')
			print('*******************')
			obsCorr = False

		# Correct for obscuration. If it somehow fails, continue the fit without correcting for obscuration.
		if obsCorr == True:
			try:
				S9p7 = calc_S9p7(wavSpec_rest, fluxSpec)
			except:
				S9p7 = -99.
				print('*******************')
				print('It has failed to correct the IRS spectrum for obscuration. The most likely explanation is '+ \
					  'redshift since it needs the restframe anchor ' + \
					  'wavelengths to measure the strength of the silicate absorption. The fit is continued without correcting for obscuration.')
				print('*******************')
				pass
		else:
			S9p7 = -99.
	else:
		S9p7 = S9p7_fixed

	# Open the extincrtion curve
	EC_wav, EC_tau = getExtCurve(ExtCurve)
	EC_wav_AGN, EC_tau_AGN = getExtCurve('PAHfit')
	if S9p7 > 0.:
		#Calculate tau9p7 for the galaxy
		tau9p7_gal = S9p7toTau9p7(S9p7, 'gal')
		tau9p7_AGN = S9p7toTau9p7(S9p7, 'AGN')

		# Galaxy
		tau = np.interp(wavFit/(1.+z), EC_wav, EC_tau) * tau9p7_gal
		obsPerWav_gal = ((1. - np.exp(-tau))/tau)

		#AGN
		tau = np.interp(wavFit/(1.+z), EC_wav_AGN, EC_tau_AGN) * tau9p7_AGN
		obsPerWav_AGN = np.exp(-tau)
	else:
		obsPerWav_gal = 1.
		obsPerWav_AGN = 1.

	# Define the free parameters
	logNorm_Dust_perTempl = [] #dust continuum
	elogNorm_Dust_perTempl = []
	logNorm_PAH_perTempl = [] #PAH emission
	elogNorm_PAH_perTempl = []
	
	logNorm_Si_perTempl = [] #silicate at 18 micron
	elogNorm_Si_perTempl = []
	logNormAGN_PL_perTempl = [] #norm AGN continuum
	elogNormAGN_PL_perTempl = []
	l_break_PL_perTempl = [] # position of the break for the PL
	el_break_PL_perTempl = []
	alpha1_perTempl = [] #slope of the first power law
	ealpha1_perTempl = []
	alpha2_perTempl = [] #slope of the second power law
	ealpha2_perTempl = []
	alpha3_perTempl = [] 
	ealpha3_perTempl = []
	dSi_perTempl = []
	edSi_perTempl = []

	logl_perTempl = [] #log-likelihood of the mode
	tplName_perTempl = [] #Name of the template 
	S9p7_save = [] #total absorption at 9.7 micron
	AGNon = [] #flag for the use of the AGN in the fit
	nParms = [] # number of parameters

	x_red = 10**np.arange(np.log10(7), np.log10(30), 0.05)
	x_red[0] = wavFit.min()/(1.+z)

	# Fit looping over every of our galaxy templates
	for name_i in nameTempl_gal:
		assert isinstance(name_i, str), "The list nameTempl requests strings as it corresponds to the names" + \
										" given to the various templates of galaxies to use for the fit."

		if pgrbar == 1:
			print("****************************************")
			print("  Fit of "+name_i+" as galaxy template  ")
			print("****************************************")

		# Define synthetic fluxes for the dust continuum model at the observed wavelength to match the observed fluxes
		nuLnuBGTempl = templ[name_i].values

		Fnu = nuLnuToFnu(wavTempl, nuLnuBGTempl, z)
		fluxSpec_model = np.interp(wavSpec_rest, wavTempl, Fnu)

		SEDgen = modelToSED(wavTempl, nuLnuBGTempl, z)
		fluxPhot_model = []
		for filt in filters:
		 	fluxPhot_model.append(getattr(SEDgen, filt)())

		modelDust = np.concatenate([fluxSpec_model, fluxPhot_model]) * obsPerWav_gal

		# Define synthetic fluxes for the PAH model at the observed wavelength to match the observed fluxes.
		# When an empirical template is used, define a vector of zeros so that no PAH emission is accounted for.
		nuLnuSGTempl = templ[nameTempl_PAH[0]].values

		Fnu = nuLnuToFnu(wavTempl, nuLnuSGTempl, z)
		fluxSpec_model = np.interp(wavSpec_rest, wavTempl, Fnu)

		SEDgen = modelToSED(wavTempl, nuLnuSGTempl, z)
		fluxPhot_model = []
		for filt in filters:
			fluxPhot_model.append(getattr(SEDgen, filt)())

		modelPAH = np.concatenate([fluxSpec_model, fluxPhot_model])

		# fit without the AGN
		ndim = 2 #Number of paraneter
		nwalkers = int(2. * ndim) #number of walkers

		# Define the starting point of each of the parameters. Flat distrib. between -1 and 1. Parameters are normalised with a zero mean to ease convergence.
		parms = np.zeros(shape=(nwalkers, ndim))
		parms[:,0] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # norm Dust
		parms[:,1] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # norm PAH

		# Set the ensemble sampler of Goodman&Weare and run the MCMC for Nmc steps.
		sampler = EnsembleSampler(nwalkers, ndim, lnpostfn_spec_noAGN, \
								  args = (np.array([Pdust, PPAH]), modelDust, modelPAH, fluxFit, efluxFit, \
								  UL, wei))
		sampler.run_mcmc(parms, Nmc, progress=bool(pgrbar))
		
		# Build the flat chain after burning 20% of it and thinning to every 10 values.
		NburnIn = int(0.2 * Nmc)
		chain = sampler.get_chain(discard=NburnIn, thin=10, flat=True)
		dfChain = pd.DataFrame(chain)
		dfChain.columns = ['logNormDust', 'logNormPAH']

		# Save the optimised parameters. Median of the posterior as best fitting value and std dev as 1sigma uncertainties.
		# Dust continuum
		logNorm_Dust_perTempl.append(dfChain['logNormDust'].median())
		elogNorm_Dust_perTempl.append(dfChain['logNormDust'].std())

		# PAH
		logNorm_PAH_perTempl.append(dfChain['logNormPAH'].median())
		elogNorm_PAH_perTempl.append(dfChain['logNormPAH'].std())

		# Norm AGN, here -99. since no AGN accounted for
		logNormAGN_PL_perTempl.append(-99.)
		elogNormAGN_PL_perTempl.append(-99.)

		# slope of the first power law, here -99. since no AGN accounted for
		alpha1_perTempl.append(-99.)
		ealpha1_perTempl.append(-99.)

		# slope of the second power law, here -99. since no AGN accounted for
		alpha2_perTempl.append(-99.)
		ealpha2_perTempl.append(-99.)

		alpha3_perTempl.append(-99.)
		ealpha3_perTempl.append(-99.)

		# cut off or position of the break, here -99. since no AGN accounted for
		l_break_PL_perTempl.append(-99.)
		el_break_PL_perTempl.append(-99.)

		# silicate emisison at 10 microns, here -99. since no AGN accounted for
		logNorm_Si_perTempl.append(-99.)
		elogNorm_Si_perTempl.append(-99.)

		# Shift of the peak for the siicate emission
		dSi_perTempl.append(-99.)
		edSi_perTempl.append(-99.)

		# Calculate the logl of the model
		logl = lnpostfn_spec_noAGN(np.array([logNorm_Dust_perTempl[-1], logNorm_PAH_perTempl[-1]]), \
								   np.array([Pdust, PPAH]), \
								   modelDust, modelPAH, \
								   fluxFit, efluxFit, \
								   UL, wei)

		logl_perTempl.append(logl)

		# Flag for the use of an AGN. Here 0., since no AGN is accounted for.
		AGNon.append(0.)

		# save the number of parameters
		nParms.append(ndim)

		# save the name of the template
		tplName_perTempl.append(name_i)
		
		# Save teh total absorption at 9.7 microns.
		S9p7_save.append(round(S9p7,3))

		# Fit including the full AGN model.
		ndim = 9 # Number of parms
		nwalkers = int(2.* ndim) # Number of walkers

		# Define the starting parms. Each of them has been normalised to a mean of zero to ease convergence.
		parms = np.zeros(shape=(nwalkers, ndim))
		parms[:,0] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # norm Dust
		parms[:,1] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # norm PAH
		parms[:,2] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # norm PL AGN
		parms[:,3] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # alpha1
		parms[:,4] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # alpha2
		parms[:,5] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # alpha3
		parms[:,6] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # Siem
		parms[:,7] = np.random.uniform(low = -0.3, high = 0.3, size=nwalkers) # Si shift
		parms[:,8] = np.random.uniform(low = -5., high = 5., size=nwalkers) # Break

		modelSi11 = Simodel(wavFit/(1.+z), 11., 0.)
		modelSi18 = Simodel(wavFit/(1.+z), 18., 0.)
		modelSi = modelSi18 + modelSi11

		# Set the ensemble sample of Goodman&Weare and run the MCMC for Nmc steps
		sampler = EnsembleSampler(nwalkers, ndim, lnpostfn_spec_wAGN, \
								  args = (np.array([Pdust, PPAH, PPL, Palpha, PSi, Pbreak]), \
								  modelDust, modelPAH, modelSi, fluxFit, efluxFit, UL, wei, z, wavFit, obsPerWav_AGN, x_red))
		sampler.run_mcmc(parms, Nmc, progress=bool(pgrbar))

		# Build the flat chain after burning 20% of the chain and thinning to every 10 values.
		chain = sampler.get_chain(discard=NburnIn, thin=10, flat=True)
		dfChain = pd.DataFrame(chain)
		dfChain.columns = ['logNormDust', 'logNormPAH', 'logNormAGN', 'alpha1', 'alpha2', 'alpha3', 'logNormSi', 'dSi', 'lBreak']

		# Save the optimised parameters. Median of the posterior as best fitting value and std dev as 1sigma uncertainties.
		# Dust continuum
		logNorm_Dust_perTempl.append(dfChain['logNormDust'].median())
		elogNorm_Dust_perTempl.append(dfChain['logNormDust'].std())

		# PAH
		logNorm_PAH_perTempl.append(dfChain['logNormPAH'].median())
		elogNorm_PAH_perTempl.append(dfChain['logNormPAH'].std())
		
		# Norm AGN
		logNormAGN_PL_perTempl.append(dfChain['logNormAGN'].median())
		elogNormAGN_PL_perTempl.append(dfChain['logNormAGN'].std())

		# slope of the first power law
		alpha1_perTempl.append(dfChain['alpha1'].median())
		ealpha1_perTempl.append(dfChain['alpha1'].std())

		# slope of the second power law
		alpha2_perTempl.append(dfChain['alpha2'].median())
		ealpha2_perTempl.append(dfChain['alpha2'].std())

		# slope of the third power law
		alpha3_perTempl.append(dfChain['alpha3'].median())
		ealpha3_perTempl.append(dfChain['alpha3'].std())

		# silicate emisison
		logNorm_Si_perTempl.append(dfChain['logNormSi'].median())
		elogNorm_Si_perTempl.append(dfChain['logNormSi'].std())

		# dSi
		dSi_perTempl.append(dfChain['dSi'].median())
		edSi_perTempl.append(dfChain['dSi'].std())

		# cut off or position of the break
		l_break_PL_perTempl.append(dfChain['lBreak'].median())
		el_break_PL_perTempl.append(dfChain['lBreak'].std())

		# Clauclate the logl of the model
		theta = np.array([logNorm_Dust_perTempl[-1],\
						  logNorm_PAH_perTempl[-1],\
						  logNormAGN_PL_perTempl[-1],\
						  alpha1_perTempl[-1], \
						  alpha2_perTempl[-1], \
						  alpha3_perTempl[-1], \
						  logNorm_Si_perTempl[-1], \
						  dSi_perTempl[-1], \
						  l_break_PL_perTempl[-1],\
						  ])
		Pr = np.array([Pdust, PPAH, PPL, Palpha, PSi, Pbreak])
		logl = lnpostfn_spec_wAGN(theta, Pr, modelDust, modelPAH, modelSi, fluxFit, efluxFit, UL, wei, z, wavFit, obsPerWav_AGN, x_red)
		logl_perTempl.append(logl)

		# AGNon = 1 since AGN is accounted for
		AGNon.append(1.)

		# Save the number of parameters
		nParms.append(ndim)

		# Save the name of the galaxy template
		tplName_perTempl.append(name_i)

		# Save the total obscuration at 9.7 microns
		S9p7_save.append(round(S9p7,3))

	# Find the best model and the Akaike weight
	bestModelInd, Awi = exctractBestModel(logl_perTempl, nParms, len(wavFit), corrected = False)
	bestModelFlag = np.zeros(len(AGNon))
	bestModelFlag[bestModelInd] = 1

	# Save the results in a table
	resDict = {'logNormGal_dust': np.array(logNorm_Dust_perTempl) + Pdust[0], 'elogNormGal_dust': np.abs(np.array(elogNorm_Dust_perTempl)), \
			   'logNormGal_PAH': np.array(logNorm_PAH_perTempl) + PPAH[0], 'elogNormGal_PAH': np.abs(np.array(elogNorm_PAH_perTempl)), \
			   'logNormAGN_PL': np.array(logNormAGN_PL_perTempl) + PPL[0], 'elogNormAGN_PL': np.abs(np.array(elogNormAGN_PL_perTempl)), \
			   'lBreak_PL': np.array(l_break_PL_perTempl) + Pbreak[0], 'elBreak_PL': np.abs(np.array(el_break_PL_perTempl )), \
			   'alpha1': np.array(alpha1_perTempl) + Palpha[0], 'ealpha1': np.abs(np.array(ealpha1_perTempl)), \
			   'alpha2': np.array(alpha2_perTempl) + Palpha[0], 'ealpha2': np.abs(np.array(ealpha2_perTempl)), \
			   'alpha3': np.array(alpha3_perTempl) + Palpha[0], 'ealpha3': np.abs(np.array(ealpha3_perTempl)), \
			   'logNorm_Si': np.array(logNorm_Si_perTempl) + PSi[0], 'elogNorm_Si': np.abs(np.array(elogNorm_Si_perTempl)), \
			   'dSi': np.array(dSi_perTempl), 'edSi': np.abs(np.array(edSi_perTempl)), \
			   'logl': logl_perTempl, 'AGNon': AGNon, 'tplName': tplName_perTempl,\
			   'bestModelFlag': bestModelFlag, 'Aw': Awi, 'S9p7': S9p7_save}

	dfRes = pd.DataFrame(resDict)
	return dfRes


#################################################
#												#
#		PHOTOMETRIC VERSION OF THE FITTING		#
#												#
#################################################
@njit(fastmath=True)
def lnpostfn_photo_noAGN(theta, P, modelDust, modelPAH, UL, y, ey):

	"""
    This function calculates the log-likelihood between photometric data and model without AGN contribution.
    ------------
    :param theta: vector containing the parameters.
    :param P: vector containing the priors for each of the parameters.
    :param modelDust: model dust continuum template.
    :param modelPAH: model PAH template.
    :param UL: vector contaning the upper-limits, if any.
    :param y: observed fluxes.
    :param ey: uncertainties on the observed fluxes.
    ------------
    :return logl: log-likelihood of the model knowing the parameters theta.
    """
	# set the log-likelihood to zero
	logl = 0

	# prior constrain on the normalisation of the dust continuum for galaxy
	logl += -0.5*(theta[0]/P[0][1])**2.

	# prior constrain on the normalisation of the PAH
	logl += -0.5*(theta[1]/P[1][1])**2.

	# Constraint on the 0.1 dex between the PAH and the dust continuum
	logl += -0.5*(((theta[1] + P[1][0]) - (0.97 * (theta[0] + P[0][0]) - 0.95))/0.1)**2.

	# define the full model as PAH + dust continuum
	ym = 10**(theta[0] + P[0][0]) * modelDust + 10**(theta[1] + P[1][0]) * modelPAH

	# Upper Limits
	x = (ym[UL == 1.] - y[UL == 1.])/y[UL == 1.]/np.sqrt(2) * 3.
	cdf = np.prod(1. - (0.5 * (1. + nberf(x))))

	if cdf == 0.:	
		logl += -1e10
	else:
		logl += np.log(cdf)

	# Detected fluxes
	logl += np.sum(-0.5 * ((y[UL == 0.] - ym[UL == 0.])/(ey[UL == 0.]))**2.)

	return logl

@njit(fastmath=True)
def lnpostfn_photo_wAGN(theta, P, modelDust, modelPAH, modelAGN, modelSi, UL, y, ey):
	
	"""
    This function calculates the log-likelihood between photometric data and model that includes AGN contribution.
    ------------
    :param theta: vector containing the parameters.
    :param P: vector containing the priors for each of the parameters.
    :param modelDust: model dust continuum template.
    :param modelPAH: model PAH template.
    :param modelAGN: model AGN continuum template.
    :param modelSi: model silicate emission template.
    :param UL: vector contaning the upper-limits, if any.
    :param y: observed fluxes.
    :param ey: uncertainties on the observed fluxes.
    ------------
    :return logl: log-likelihood of the model knowing the parameters theta.
    """
	# set the log-likelihood to zero
	logl = 0

	# prior constrain on the normalisation of the dust continuum for galaxy
	logl += -0.5*(theta[0]/P[0][1])**2.
		
	# prior constrain on the normalisation of the PAH
	logl += -0.5*(theta[1]/P[1][1])**2.

	# constrain on 0.3 dex between the normalisation of the dust to that of PAHs
	logl += -0.5*((theta[1] + P[1][0] - (0.97 * (theta[0] + P[0][0]) - 0.95))/0.01)**2.

	# prior constrain on the normalisation of the AGN continuum
	logl += -0.5*(theta[2]/P[2][1])**2.

	# prior constraint on the normalisation of the silicate emission
	if np.sum(modelSi) != 0.:
		logl += -0.5*(theta[3]/P[3][1])**2.
	else:
		theta[3] = 0.

	# define the full model as PAH + dust continuum + AGN continuum + silicate emission
	ym = 10**(theta[0] + P[0][0]) * modelDust + 10**(theta[1] + P[1][0]) * modelPAH + 10**(theta[2] + P[2][0]) * modelAGN + 10**(theta[3] + P[3][0]) * modelSi

	# Upper Limits
	x = (ym[UL == 1.] - y[UL == 1.])/y[UL == 1.]/np.sqrt(2) * 3.
	cdf = np.prod(1. - (0.5 * (1. + nberf(x))))

	if cdf == 0.:	
		logl += -1e20
	else:
		logl += np.log(cdf)

	# Detected fluxes
	logl += np.sum(-0.5 * ((y[UL == 0.] - ym[UL == 0.])/(ey[UL == 0.]))**2.)

	return logl

def runSEDphotFit(lambdaObs, fluxObs, efluxObs, \
				  filters, \
				  z = 0.01, \
				  UL = [], \
				  S9p7 = -99.,\
				  ExtCurve = 'iragnsep', \
				  Nmc = 10000, pgrbar = 1, \
				  NoSiem = False, \
				  Pdust = [10., 1.], PPAH = [9., 1.], PnormAGN = [10., 1.], PSiEm = [10., 1.], \
				  templ = '', \
				  NOAGN = False):

	"""
    This function fits the observed photometric SED.
    ------------
    :param lambdaObs: observed wavelengths (in microns).
    :param fluxObs: observed fluxes (in Jansky).
    :param efluxObs: observed uncertainties on the fluxes (in Jansky).
    :param filters: name of the photometric filters to include in the fit.
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword UL: vector of length Nphot, where Nphot is the number of photometric data. If any of the value is set to 1, 
    the corresponding flux is set has an upper limit in the fit. Default = [].
    :keyword S9p7: can be used to pass a fixed value for the total silicate absorption at 9.7 micron. Default = -99.
    :keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
    :keyword Nmc: numer of MCMC run. Default = 10000.
    :keyword pgrbar: if set to 1, display a progress bar while fitting the SED. Default = 1.
	:keyword NoSiem: if set to True, no silicate emission template is included in the fit. Default = False.
    :keyword Pdust: normal prior on the log-normalisation of the galaxy dust continuum template. Default = [10., 1.] ([mean, std dev]).
    :keyword PPAH: normal prior on the log-normalisation of the PAH template. Default = [9., 1.] ([mean, std dev]).
    :keyword PnormAGN: normal prior on the log-normalisation of the AGN template. Default = [10., 1.] ([mean, std dev]).
    :keyword PSiem: normal prior on the log-normalisation of the silicate emission template. Default = [10., 1.] ([mean, std dev]).
    :keyword templ: the templates used for the fits. Default: iragnsep templates.
    :keyword NOAGN: if set to True, fits are ran with SF templates only (i.e. no AGN emission is accounted for). Default = False.
	------------
    :return dfRes: dataframe containing the results of all the possible fits.
    """

	path_iragnsep = os.path.dirname(iragnsep.__file__)

	# If no templates are passed, open the Bernhard+20 templates.
	if len(templ) == 0:
		templ = pd.read_csv(path_iragnsep+'/iragnsep_templ.csv')

	# Extract the name of the templates
	keys = templ.keys().values
	nameTempl_gal = []
	nameTempl_PAH = []
	nameTempl_AGN = []
	nameTempl_Siem = []
	for key in keys:
		if str(key).startswith('gal'):
			if str(key).endswith('PAH') == False:
				nameTempl_gal.append(key)
			else:
				nameTempl_PAH.append(key)
		if str(key).startswith('AGN'):
			if str(key).endswith('Siem'):
				nameTempl_Siem.append(key)
			else:
				nameTempl_AGN.append(key)

	# Test that we have template for everything (if no galaxy then it crashes)
	if len(nameTempl_gal) == 0:
		raise ValueError('The galaxy template does not exist. The name of the column defining nuLnu for the galaxy template needs to start with "gal".')

	# define the wavelengths
	try:
		wavTempl = templ['lambda_mic'].values
	except:
		raise ValueError('Rename the wavelengths column of the template "lambda_mic".')

	# Increase the uncertainties if too small. Otherwise walkers get lost.
	SNR = efluxObs/fluxObs
	efluxObs[SNR < 1e-2] = fluxObs[SNR < 1e-2] * 1e-2

	# Open the extinction curve
	EC_wav, EC_tau = getExtCurve(ExtCurve)
	EC_wav_AGN, EC_tau_AGN = getExtCurve('PAHfit')
	if S9p7 > 0.:
		#Calculate tau9p7 for the galaxy
		tau9p7_gal = S9p7toTau9p7(S9p7, 'gal')
		tau9p7_AGN = S9p7toTau9p7(S9p7, 'AGN')

		# Galaxy
		tau = np.interp(lambdaObs/(1.+z), EC_wav, EC_tau) * tau9p7_gal
		obsPerWav_gal = ((1. - np.exp(-tau))/tau)

		#AGN
		tau = np.interp(lambdaObs/(1.+z), EC_wav_AGN, EC_tau_AGN) * tau9p7_AGN
		obsPerWav_AGN = np.exp(-tau)
	else:
		obsPerWav_gal = 1.
		obsPerWav_AGN = 1.

	# Define a vectors of zeros if no upper limts are passed.
	if len(UL) != len(lambdaObs):
		UL = np.zeros(len(lambdaObs))

	o = np.where(UL == 0.)[0]
	if len(o) == 0.:
		UL = np.zeros(len(lambdaObs))
		efluxObs = fluxObs*0.1

	# Define the free parameters
	# Norm AGN continuum
	lnAGN_perTempl = []
	elnAGN_perTempl= []

	# Norm silicate emission
	lnSi_perTempl = []
	elnSi_perTempl = []

	# Norm dust continuum
	lnDust_perTempl = []
	elnDust_perTempl = []

	# Norm PAHs
	lnPAH_perTempl = []
	elnPAH_perTempl = []

	# final loglikelihood of the model
	logl_perTempl = []

	# name of the AGN and galaxy template
	tplNameGal_perTempl = []
	tplNameAGN_perTempl = []

	# if AGN is accounted for (1) or not (0)
	AGNon = []

	# Number of parameters in the model
	nParms = []

	# We loop over the 6 galaxy templates, first fitting only galaxy templates and then including the AGN templates.
	for name_i in nameTempl_gal:
		# if name_i != 'gal7_dust':
		# 	continue
		assert isinstance(name_i, str), "The list nameTempl requests strings as it corresponds to the names" + \
										" given to the various templates of galaxies to use for the fit."

		if pgrbar == 1:
			print("****************************************")
			print("Fit of "+name_i+" as galaxy template")


		# Define synthetic fluxes for the dust continuum model at the observed wavelength to match the observed fluxes
		nuLnuBGTempl = templ[name_i].values

		SEDgen = modelToSED(wavTempl, nuLnuBGTempl, z)
		fluxPhot_model = []
		for filt in filters:
		 	fluxPhot_model.append(getattr(SEDgen, filt)())

		modelDust = np.array(fluxPhot_model) * obsPerWav_gal

		# Define synthetic fluxes for the PAH model at the observed wavelength to match the observed fluxes.
		# When an empirical template is used, define a vector of zeros so that no PAH emission is accounted for.
		nuLnuSGTempl = templ[nameTempl_PAH[0]].values

		SEDgen = modelToSED(wavTempl, nuLnuSGTempl, z)
		fluxPhot_model = []
		for filt in filters:
			fluxPhot_model.append(getattr(SEDgen, filt)())

		modelPAH = np.array(fluxPhot_model)

		# Perform the fit without the AGN contribution
		ndim = 2 # Number of free params
		nwalkers = int(10. * ndim) # Number of walkers

		# Define the parameters as flat distributions between -3. and 3. (We normalised to zero each parameters to ease convergence).
		parms = np.zeros(shape=(nwalkers, ndim))
		parms[:,0] = np.random.uniform(low = -1. * Pdust[1], high = Pdust[1], size=nwalkers) # norm Dust
		parms[:,1] = np.random.uniform(low = -1. * PPAH[1], high = PPAH[1], size=nwalkers) # norm PAH

		# Set the ensemble sampler of Goodman&Weare and run the MCMC for Nmc steps
		if pgrbar == 1:
			print(" -- NO AGN --")
		sampler = EnsembleSampler(nwalkers, ndim, lnpostfn_photo_noAGN, \
								  moves=[emcee.moves.StretchMove()], \
								  args = (np.array([Pdust, PPAH]), modelDust, modelPAH, UL, fluxObs, efluxObs))
		sampler.run_mcmc(parms, int(Nmc), progress=bool(pgrbar))

		# Build the flat chain, after burning 20% of the chain and thinning to every 10 values.
		NburnIn = int(0.2 * Nmc)
		chain = sampler.get_chain(discard=NburnIn, thin=10, flat=True)

		# Save the best fit parameters. Median of the posterior is taken as the best fit parameter and the standard deviation as 1sigma uncertainties.
		# Norm dust continuum
		lnDust_perTempl.append(np.median(chain[:,0]))
		elnDust_perTempl.append(np.std(chain[:,0]))

		# Norm PAH emission (or -20. when the empirical template is used).
		lnPAH_perTempl.append(np.median(chain[:,1]))
		elnPAH_perTempl.append(np.std(chain[:,1]))

		# Calulate the final loglikelihood of the model, using the best fit parameters.
		logl_perTempl.append(lnpostfn_photo_noAGN(np.array([lnDust_perTempl[-1], lnPAH_perTempl[-1]]), np.array([Pdust, PPAH]), modelDust, modelPAH, UL, fluxObs, efluxObs))
		
		# Norm on the silicate emission. -99. here since no AGN is accounted for.
		lnSi_perTempl.append(-99.)
		elnSi_perTempl.append(-99.)

		# Norm on the AGN template. -99. here since no AGN is accounted for.
		lnAGN_perTempl.append(-99.)
		elnAGN_perTempl.append(-99.)

		# No AGN in this fits, so AGNon set to zero
		AGNon.append(0.)
		
		# Save the numbers of parameters used in the fit, i.e. 2.
		nParms.append(2.)
		
		# Save the name of the template for the galaxy.
		tplNameGal_perTempl.append(name_i)
		
		# No AGN templates used in this fit, so name of the AGN template is set to 'N/A'.
		tplNameAGN_perTempl.append('N/A')

		if NOAGN != True:

			# Fit including the AGN. Loop over the two templates AGN A and AGN B.
			for AGN_i in nameTempl_AGN:

				# calculate the synthetic photometry of the AGN template at wavelengths lambdaObs.
				nuLnu_AGN = templ[AGN_i].values
				# generate the photometry
				SEDgen = modelToSED(wavTempl, nuLnu_AGN, z)
				modelAGN = []
				for filt in filters:
					modelAGN.append(getattr(SEDgen, filt)())

				modelAGN = np.array(modelAGN) * obsPerWav_AGN

				# calculate the synthetic photometry of the silicate template at wavelengths lambdaObs.
				# If no silicate emission is considered, set a vector of zeros so that no silicate emisison is account for.
				if NoSiem == False:
					modelSiem = []
					nuLnu_Siem = templ[nameTempl_Siem].values.flatten()
					SEDgen = modelToSED(wavTempl, nuLnu_Siem, z)
					for filt in filters:
						modelSiem.append(getattr(SEDgen, filt)())

					modelSiem = np.array(modelSiem) * obsPerWav_AGN
				else:
					modelSiem = modelAGN * 0.

				# Perform the fit with the AGN contribution
				ndim = 4 # Number of parmameters
				nwalkers = int(10. * ndim) # Number of walkers

				# Define the starting parameters as flat distributions between -1 and 1. We normalised each parameter to zero to each convergence.
				parms = np.zeros(shape=(nwalkers, ndim))
				parms[:,0] = np.random.uniform(low = -1. * Pdust[1], high = Pdust[1], size=nwalkers) # norm Dust
				parms[:,1] = np.random.uniform(low = -1. * PPAH[1], high = PPAH[1], size=nwalkers) # norm PAH
				parms[:,2] = np.random.uniform(low = -1. * PnormAGN[1], high = PnormAGN[1], size=nwalkers) # normAGN
				parms[:,3] = np.random.uniform(low = -1. * PSiEm[1], high = PSiEm[1], size=nwalkers) # normSi

				# Set the ensemble sampler of Goodman&Weare and run the MCMC for Nmc steps
				print('-- ' + AGN_i + ' --')
				sampler = EnsembleSampler(nwalkers, ndim, lnpostfn_photo_wAGN, \
										  moves=[emcee.moves.StretchMove()],\
										  args = (np.array([Pdust, PPAH, PnormAGN, PSiEm]), \
										  modelDust, modelPAH, modelAGN, modelSiem, UL, fluxObs, efluxObs))
				sampler.run_mcmc(parms, int(Nmc), progress=bool(pgrbar))

				# Build the flat chain, after burning 20% of the chain and thinning to every 10 values in the chain.
				chain = sampler.get_chain(discard=NburnIn, thin=10, flat=True)

				# Save the best fit parameters. Median of the posterior is taken as the best fit parameter and the standard
				# deviation as 1sigma uncertainties.
				# Dust Normalisation
				lnDust_perTempl.append(np.median(chain[:,0]))
				elnDust_perTempl.append(np.std(chain[:,0]))

				# PAH normalisation
				lnPAH_perTempl.append(np.median(chain[:,1]))
				elnPAH_perTempl.append(np.std(chain[:,1]))
				
				# AGN continuum Norm
				lnAGN_perTempl.append(np.median(chain[:,2]))
				elnAGN_perTempl.append(np.std(chain[:,2]))

				if NoSiem == True:
					
					# Si emission Norm
					lnSi_perTempl.append(-99.)
					elnSi_perTempl.append(-99.)

					# Numbers of params in the model
					nParms.append(3.)

				else:

					# Si emission Norm
					lnSi_perTempl.append(np.median(chain[:,3]))
					elnSi_perTempl.append(np.std(chain[:,3]))

					# Numbers of params in the model
					nParms.append(4.)

				# AGN accounted for in this case, so AGNon = 1
				AGNon.append(1.)

				# Name of the galaxy template
				tplNameGal_perTempl.append(name_i)

				# Name of the AGN template
				tplNameAGN_perTempl.append(AGN_i)

				# loglikelihood of the model
				logl_perTempl.append(lnpostfn_photo_wAGN(np.array([lnDust_perTempl[-1], lnPAH_perTempl[-1], lnAGN_perTempl[-1], \
										   lnSi_perTempl[-1]]), np.array([Pdust, PPAH, PnormAGN, PSiEm]), modelDust, modelPAH, modelAGN, \
										   modelSiem, UL, fluxObs, efluxObs))

				if NoSiem == True:
					lnSi_perTempl[-1] = -99.

	# Find the best model and the Akaike weight amongst all the 18 possible fits by comparing their final loglikelihood
	bestModelInd, Awi = exctractBestModel(logl_perTempl, nParms, len(lambdaObs), corrected = True)
	bestModelFlag = np.zeros(len(AGNon))
	bestModelFlag[bestModelInd] = 1

	# Save the results in a table
	resDict = {'logNormGal_dust': np.array(lnDust_perTempl) + Pdust[0], 'elogNormGal_dust': np.array(elnDust_perTempl), \
			   'logNormGal_PAH': np.array(lnPAH_perTempl) + PPAH[0], 'elogNormGal_PAH': np.array(elnPAH_perTempl), \
			   'logNormAGN': np.array(lnAGN_perTempl) + PnormAGN[0], 'elogNormAGN': np.array(elnAGN_perTempl), \
			   'logNormSiem': np.array(lnSi_perTempl) + PSiEm[0], 'elogNormSiem': np.array(elnSi_perTempl), \
			   'logl': logl_perTempl, 'AGNon': AGNon, 'tplName_gal': tplNameGal_perTempl, 'tplName_AGN': tplNameAGN_perTempl,\
			   'bestModelFlag': bestModelFlag, 'Aw': Awi, 'S9p7':S9p7}

	dfRes = pd.DataFrame(resDict)

	return dfRes