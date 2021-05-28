import numpy as np
import os
import iragnsep
import glob
import pandas as pd
import numba

from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy import constants as const
from scipy import integrate
from scipy.interpolate import UnivariateSpline, interp1d
from numba import njit

c = const.c.value
Lsun = const.L_sun.value
h = const.h.value
k = const.k_B.value

path_iragnsep = os.path.dirname(iragnsep.__file__)

def basictests(wavSpec, fluxSpec, efluxSpec, wavPhot, fluxPhot, efluxPhot, filters, z, specOn = True):

	"""
    This function runs some basic tests prior to run the main fitting code.
    ------------
    :param wavSpec: observed wavelengths for the spectrum (in microns).
    :param fluxSpec: observed fluxes for the spectrum (in Jansky).
    :param efluxSpec: observed uncertainties on the fluxes for the spectrum (in Jansky).
    :param wavPhot: observed wavelengths for the photometry (in microns).
    :param fluxPhot: observed fluxes for the photometry (in Jansky).
    :param efluxPhot: observed uncertainties on the fluxes for the photometry (in Jansky).
    :param filters: name of the photometric filters to include in the fit.
    :param z: redshift.
    ------------
	:keyword specOn: set to True if the data contain a spectrum in addition to the photometry. Default = True
    ------------
    :return 0
    """

	if (len(wavPhot) != len(fluxPhot)) or (len(wavPhot) != len(efluxPhot)):
		raise ValueError("PHOTOMETRY ISSUE: Crashed because wavelengths, fluxes and uncertainties on the fluxes have different lengths.")
	if len(filters) != len(wavPhot):
		raise ValueError("FILTERS ISSUE: Crashed because the number of filters provided does not correspond to the number of photometry points.")
	if (any(fluxPhot<0) == True) or (any(wavPhot<0) == True):
		raise ValueError("PHOTOMETRY ISSUE: Crash caused by some negative values in the wavelengths, fluxes or uncertainties on the fluxes.")
	if (any(fluxPhot != fluxPhot) == True) or (any(efluxPhot != efluxPhot) == True) or (any(wavPhot != wavPhot) == True):
		raise ValueError("PHOTOMETRY ISSUE: Crash caused by some non-numerical values in the wavelengths, fluxes or uncertainties on the fluxes.")
	if specOn == True:
		#test that the length of wavelength is the same as the data
		if (len(wavSpec) != len(fluxSpec)) or (len(wavSpec) != len(efluxSpec)):
			raise ValueError("SPECTRUM ISSUE: Crashed because wavelengths, fluxes and uncertainties on the fluxes have different lengths.")
		#test that there are no negative values
		if (any(fluxSpec<0) == True) or (any(efluxSpec<0) == True) or (any(wavSpec<0) == True):
			raise ValueError("SPECTRUM ISSUE: Crash caused by some negative values in the wavelengths, fluxes or uncertainties on the fluxes.")
		#test that there are NAN
		if (any(fluxSpec != fluxSpec) == True) or (any(efluxSpec != efluxSpec) == True) or (any(wavSpec != wavSpec) == True):
			raise ValueError("SPECTRUM ISSUE: Crash caused by some non-numerical values in the wavelengths, fluxes or uncertainties on the fluxes.")
		#test that no photometry overlap with the spectral data
		if len(wavPhot[(wavPhot > wavSpec.min()) & (wavPhot < wavSpec.max())]) > 0.:
			raise ValueError("SPECTRUM/PHOTOMETRY ISSUE: Crash caused because there are some photometric data overlapping with the spectral data.")
	
	#test if the filter exists
	path = os.path.dirname(iragnsep.__file__) + '/Filters/'
	files = [f for f in glob.glob(path + "*.csv")]
	count = -1
	for f in [path+f+"Filter.csv" for f in filters]:
		count += 1
		if (f in files) == False:
			raise ValueError(" \n The filter "+ str(filters[count]) + " does not exist. This version does not allow you to add some filters." + \
							 " Please get in touch with us to add the required filters (manu.p.bernhard[at]gmail.com). Available filters are:" +\
							 " IRAC1 , IRAC2 , IRAC3 , IRAC4 , WISE_W1 , WISE_W2 , WISE_W3 , WISE_W4 , IRAS12, IRAS60, IRAS100 , MIPS24, MIPS70," +\
							 " MIPS160 , PACS70, PACS100 , PACS160, SPIRE250ps , SPIRE350ps , SPIRE500ps, ISOPHTC100")

	#Test if the redshift has been given by the user
	if z<0:
		zdefault = input('Warning: The redshift is set to the default value of 0.01. The keyword "z" allows you to indicate the redshift of the source.\n '+\
						 ' Press enter to continue, or type "exit" to abort.\n')
		if zdefault == "exit":
			exit()
		else:
			pass

	# Test that the wavelengths are in ascening order
	if len(wavPhot) > 1:
		dlambda = np.gradient(wavPhot)
		o = np.where(dlambda < 0.)[0]
		if len(o) > 0.:
			raise ValueError('PHOTOMETRY ISSUE: Wavelenghts need to be in ascending order.')

	if (specOn == True):
		dlambda = np.gradient(wavSpec)
		o = np.where(dlambda < 0.)[0]
		if len(o) > 0.:
			raise ValueError('SPECTRUM ISSUE: Wavelenghts need to be in ascending order.')

	if specOn == True:
		# Test if it can concatenate the Spectra aand the photometry
		try:
			wav = np.concatenate([wavSpec, wavPhot])
		except:
			raise ValueError("WAVELENGTHS: Spectral data cannot be concatenated to photometric data. Please check data.")
		try:
			flux = np.concatenate([fluxSpec, fluxPhot])
		except:
			raise ValueError("FLUXES: Spectral data cannot be concatenated to photometric data. Please check data.")
		try:
			eflux = np.concatenate([efluxSpec, efluxPhot])
		except:
			raise ValueError("UNCERTAINTIES ON THE FLUXES: Spectral data cannot be concatenated to photometric data. Please check data.")
	else:
		# Break as not enough data points anyway,
		if len(wavPhot) < 4:
			if len(wavPhot) == 3:
				restWav = wavPhot/(1.+z)
				NFIR = len(np.where(restWav>50.)[0])
				if (NFIR > 0.):
					pass
				else:
					raise ValueError('There is not enough data points to fit the model when compared to the number of degrees of freedom. It needs a minimum of' + \
								 ' 3 photometric points, with at least 1 FIR (i.e. rest-wavelength>60micron) flux, including upper-limits.')
	
			else:
				raise ValueError('There is not enough data points to fit the model when compared to the number of degrees of freedom. It needs a minimum of' + \
								 ' 3 photometric points, with at least 1 FIR (i.e. rest-wavelength>60micron) flux, including upper-limits.')

def get_prop(df, z = 0.01, specOn = True, templ = '', NOAGN = False):

	"""
    This function calculates the IR properties of the AGN and their hosts.
    ------------
    :param df: data-frame of the results from the fits (i.e. optimised parameters) as returned by iragnsep.
    ------------
    :keyword z: redshift. Default = 0.01
	:keyword specOn: set to True if the data contain a spectrum in addition to the photometry. Default = True
    :keyword templ: set the templates that have been used in the fits. Default = ''
    ------------
    :return loglum_hostIR: the host IR (8--1000microns) log-luminosity free of AGN contamination (Lsun).
    :return eloglum_hostIR: uncertainties on loglum_hostIR.
	:return loglum_hostMIR: the host MIR (5--35microns) log-luminosity free of AGN contamination (Lsun).
    :return eloglum_hostIR: uncertainties on loglum_hostMIR.
	:return loglum_hostFIR: the host FIR (40--1000microns) log-luminosity free of AGN contamination (Lsun).
    :return eloglum_hosFIR: uncertainties on loglum_hostFIR.
	:return loglum_AGNIR: the AGN IR log-luminosity free of host contamination (Lsun).
	:return loglum_AGNMIR: the AGN MIR log-luminosity free of host contamination (Lsun).
	:return loglum_AGNFIR: the AGN FIR log-luminosity free of host contamination (Lsun).
	:return AGNfrac_IR: the AGN fraction in the IR.
	:return AGNfrac_MIR: the AGN fraction in the MIR.
	:return AGNfrac_FIR: the AGN fraction in the FIR.
	:return SFR: the SFR of the galaxy free of AGN contamination.
	:return eSFR: the uncertainties on SFR.
    """
	if len(templ) == 0:
		path = os.path.dirname(iragnsep.__file__)
		templ = pd.read_csv(path+'/iragnsep_templ.csv')

	# Extract the name of the templates
	keys = templ.keys().values
	nameTempl_gal = []
	nameTempl_AGN = []
	nameTempl_PAH = []
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
	if len(nameTempl_AGN) == 0:
		print('Warning: The template for AGN is empty. The name of the column defining nuLnu for the AGN templates needs to start with "AGN".')

	# define the wavelengths
	try:
		wavTempl = templ['lambda_mic'].values
	except:
		raise ValueError('Rename the wavelengths column of the template "lambda_mic".')

	# Calculate luminosities for each of the model
	nu = c/wavTempl/1e-6 #Hz
	o_IR = np.where((nu > c/1000./1e-6) & (nu < c/8./1e-6))[0][::-1]
	o_MIR = np.where((nu > c/35./1e-6) & (nu < c/5./1e-6))[0][::-1]
	o_FIR = np.where((nu > c/1000./1e-6) & (nu < c/40./1e-6))[0][::-1]
	dMpc = cosmo.luminosity_distance(z).value
	dmeter = dMpc*u.Mpc.to(u.m)
	d2z = dmeter**2./(1.+z) # K correction=>https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html
	JyToLsun = 1e-26 * 4. * np.pi * d2z/Lsun

	loglum_hostIR = []
	eloglum_hostIR = []
	loglum_hostMIR = []
	eloglum_hostMIR = []
	loglum_hostFIR = []
	eloglum_hostFIR = []
	loglum_AGNIR = []
	loglum_AGNMIR = []
	loglum_AGNFIR = []

	for i in range(0, len(df)):

		obj = df.iloc[i]

		normDust = 10**obj['logNormGal_dust']
		enormDust = normDust * np.log(10) * obj['elogNormGal_dust']
	
		try:
			nuLnuDust = normDust * templ[obj['tplName']].values
			enuLnuDust = np.sqrt((enormDust * templ[obj['tplName']].values)**2. + (normDust * templ['e'+obj['tplName']].values)**2.)
		except:
			nuLnuDust = normDust * templ[obj['tplName_gal']].values
			enuLnuDust = np.sqrt((enormDust * templ[obj['tplName_gal']].values)**2. + (normDust * templ['e'+obj['tplName_gal']].values)**2.)

		normPAH = 10**obj['logNormGal_PAH']
		enormPAH = normPAH * np.log(10) * obj['elogNormGal_PAH']
		
		nuLnuPAH = normPAH * templ['gal_PAH'].values
		enuLnuPAH = np.sqrt((enormPAH * templ['gal_PAH'].values)**2. + (normPAH * templ['egal_PAH'].values)**2.)

		LnuGal = (nuLnuDust + nuLnuPAH)/nu
		eLnuGal = np.sqrt(enuLnuDust**2. + enuLnuPAH**2.)/nu

		lum_hostIR_k = []
		lum_hostMIR_k = []
		lum_hostFIR_k = []

		loglum_hostIR.append(np.log10(np.sum(LnuGal[o_IR]*np.gradient(nu[o_IR]))))
		eloglum_hostIR.append(np.sqrt(np.sum((np.gradient(nu[o_IR])*eLnuGal[o_IR])**2.))/10**loglum_hostIR[-1] * 0.434)

		loglum_hostMIR.append(np.log10(np.sum(LnuGal[o_MIR]*np.gradient(nu[o_MIR]))))
		eloglum_hostMIR.append(np.sqrt(np.sum((np.gradient(nu[o_MIR])*eLnuGal[o_MIR])**2.))/10**loglum_hostMIR[-1] * 0.434)

		loglum_hostFIR.append(np.log10(np.sum(LnuGal[o_FIR]*np.gradient(nu[o_FIR]))))
		eloglum_hostFIR.append(np.sqrt(np.sum((np.gradient(nu[o_FIR])*eLnuGal[o_FIR])**2.))/10**loglum_hostFIR[-1] * 0.434)

		if NOAGN == False:
			if obj['AGNon'] == 1:
				if specOn == True:

					modelPL = 10**(obj['logNormAGN_PL']) * contmodel(np.array([obj['lBreak_PL'], obj['alpha1'], obj['alpha2'], obj['alpha3']]), wavTempl)
					modelSi11 = 10**(obj['logNorm_Si']) * Simodel(wavTempl, 11., obj['dSi'])
					modelSi18 = 10**(obj['logNorm_Si']) * Simodel(wavTempl, 18., obj['dSi'])

					LnuAGN = (modelPL + modelSi11 + modelSi18) * JyToLsun
					
					loglum_AGNIR.append(np.round(np.log10(np.trapz(LnuAGN[o_IR], nu[o_IR], dx = np.gradient(nu[o_IR]))),3)) #Lsun
					loglum_AGNMIR.append(np.round(np.log10(np.trapz(LnuAGN[o_MIR], nu[o_MIR], dx = np.gradient(nu[o_MIR]))),3)) #Lsun
					loglum_AGNFIR.append(np.round(np.log10(np.trapz(LnuAGN[o_FIR], nu[o_FIR], dx = np.gradient(nu[o_FIR]))),3)) #Lsun

				else:
					#AGN IR luminosity
					normAGN = 10**obj['logNormAGN']
					nuLnuAGN = normAGN * templ[obj['tplName_AGN']].values

					normSi = 10**obj['logNormSiem']
					nuLnuSi = normSi * templ[nameTempl_Siem].values.flatten()

					LnuAGN = (nuLnuAGN + nuLnuSi)/nu
					
					loglum_AGNIR.append(np.round(np.log10(np.trapz(LnuAGN[o_IR], nu[o_IR], dx = np.gradient(nu[o_IR]))),3)) #Lsun
					loglum_AGNMIR.append(np.round(np.log10(np.trapz(LnuAGN[o_MIR], nu[o_MIR], dx = np.gradient(nu[o_MIR]))),3)) #Lsun
					loglum_AGNFIR.append(np.round(np.log10(np.trapz(LnuAGN[o_FIR], nu[o_FIR], dx = np.gradient(nu[o_FIR]))),3)) #Lsun	

			else:
				loglum_AGNIR.append(-99.)
				loglum_AGNMIR.append(-99.)
				loglum_AGNFIR.append(-99.)
		else:
			loglum_AGNIR.append(-99.)
			loglum_AGNMIR.append(-99.)
			loglum_AGNFIR.append(-99.)

	loglum_hostIR = np.array(loglum_hostIR)
	eloglum_hostIR = np.array(eloglum_hostIR)
	loglum_hostMIR = np.array(loglum_hostMIR)
	eloglum_hostMIR = np.array(eloglum_hostMIR)
	loglum_hostFIR = np.array(loglum_hostFIR)
	eloglum_hostFIR = np.array(eloglum_hostFIR)
	
	loglum_AGNIR = np.array(loglum_AGNIR)
	loglum_AGNMIR = np.array(loglum_AGNMIR)
	loglum_AGNFIR = np.array(loglum_AGNFIR)

	#AGN fractions
	AGNfrac_IR = np.round(10**loglum_AGNIR/(10**loglum_hostIR + 10**loglum_AGNIR),2)
	o = np.where(loglum_AGNIR <= 0.)[0]
	AGNfrac_IR[o] = -99.

	AGNfrac_MIR = np.round(10**loglum_AGNMIR/(10**loglum_hostMIR + 10**loglum_AGNMIR),2)
	o = np.where(loglum_AGNMIR <= 0.)[0]
	AGNfrac_MIR[o] = -99.

	AGNfrac_FIR = np.round(10**loglum_AGNFIR/(10**loglum_hostFIR + 10**loglum_AGNFIR),2)
	o = np.where(loglum_AGNFIR <= 0.)[0]
	AGNfrac_FIR[o] = -99.

	# SFR
	SFR = 1.09e-10 * 10**loglum_hostIR
	eSFR = SFR * np.log(10) * eloglum_hostIR
	
	return loglum_hostIR, eloglum_hostIR, \
		   loglum_hostMIR, eloglum_hostMIR, \
		   loglum_hostFIR, eloglum_hostFIR, \
		   loglum_AGNIR, loglum_AGNMIR, loglum_AGNFIR, \
		   AGNfrac_IR, AGNfrac_MIR, AGNfrac_FIR, SFR, eSFR

def exctractBestModel(logl, k, n, corrected = True):

	"""
    This function extracts the best model, and calculates the Akaike weights based on the log-likelihood returned by the fits.
    ------------
    :param logl: log-likelihood returned by the fits.
    :param k: number of free parameters.
    :param n: number of data points.
    ------------
	:keyword corrected: if set to True, calculates the corrected AIC for small number of data points. Default = True
    ------------
    :return bestModelInd: the index of the best model fit.
    :return Awi: Akaike weights of each of the models, with respect to the best model.
    """

	nkdif = np.array(n)-np.array(k)
	o = np.where(nkdif == 1)[0]
	if len(o) > 0:
		corrected = False

	AIC = 2*(np.array(k) - np.array(logl))
	if corrected == True:
		AIC += (2.*np.array(k)**2. + 2.*np.array(k))/(np.array(n)-np.array(k)-1.)

	bestModelInd = np.where(AIC == np.min(AIC))[0]
	AICmin = AIC[bestModelInd][0]
	AwiNorm = np.sum(np.exp(-0.5 * (AIC-AICmin)))

	Awi = np.exp(-0.5 * (AIC-AICmin))/AwiNorm

	return bestModelInd, Awi

def nuLnuToFnu(wav, nuLnu, z):

	"""
    This function calculates the observed flux from nuLnu.
    ------------
    :param wav: rest-wavelengths (in microns).
    :param nuLnu: nuLnu.
    :param z: redshift.
    ------------
    :return Fnu: observed flux on Earth of the source located at redshift z (in Jansky).
    """

	dMpc = cosmo.luminosity_distance(z).value #Mpc
	dmeter = dMpc*u.Mpc.to(u.m)
	d2z = dmeter**2./(1.+z) # K correction=>https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html

	# Derive the observed flux
	nu = c/wav/1e-6 #Hz
	Lnu = nuLnu/nu*Lsun #W/Hz
	Fnu = Lnu/4./np.pi/d2z #W/Hz/m2

	return Fnu * 1e26 # Jy

def getFluxInFilt(filt_wav, filt_QE, wav_rest, nuLnu, z, Fnu = False):

	"""
    This function calculates the synthetic flux in a given filter and at a given redshift from a source with luminosity nuLnu.
    ------------
    :param filt_wav: passband of the filter.
	:param filt_QE: quantum efficient of the filter.
 	:param wav: rest-wavelengths (in microns).
    :param nuLnu: nuLnu.
    :param z: redshift.
    ------------
    :keyword Fnu: if set to True, do not convert nuLnu to Fnu.
    ------------
    :return flux_Obs: observed flux on Earth of the source located at redshift z with luminosity nuLnu (in Jansky).
    """

	norm = integrate.trapz(filt_QE, x=c/filt_wav/1e-6)

	if Fnu == False:
		Fnu_0 = nuLnuToFnu(wav_rest, nuLnu, z) #Flux received on Earth:
	else:
		Fnu_0 = nuLnu

	lambda_0 = wav_rest * (1. + z) # Wavelength of emission
	Fnu_0filt = np.interp(filt_wav, lambda_0, Fnu_0) # Move the template to grab the redshifted flux
	flux_Obs = integrate.trapz(Fnu_0filt*filt_QE, x = c/filt_wav/1e-6)/norm # This is the flux received on Earth

	return flux_Obs

@njit(fastmath = True)
def contmodel(theta, x):
	"""
    This function returns the model for the AGN continuum.
    ------------
    :param theta: parameters of the model (lambdab, alpha1, alpha2, alpha3).
	:param x: wavelengths.
    ------------
    :return Bnu: the model flux density.
    """

	lambdab, alpha1, alpha2, alpha3 = theta

	Bnu = np.zeros(len(x))
	Bnu[x < 14.5] = PLmodel(x[x < 14.5], 11., alpha1, alpha2, 10., 14.5)
	Bnu[(x >= 14.5)] = PLmodel(x[(x >= 14.5)], 18., alpha2, alpha3, 10., 14.5)
	Bnu[x >= 24.] = PLmodel(x[x >= 24.], lambdab, alpha3, -3.5, 2., 24.)*np.interp(24., x, Bnu)

	return Bnu

@njit(fastmath = True)
def PLmodel(x, lambdab, alpha1, alpha2, s, norm):

	"""
    This function calculates the broken power-laws of the AGN model.
    ------------
    :param x: x-values.
	:param lambdab: position of the break.
 	:param alpha1: slope before the break.
 	:param alpha2: slope after the break.
 	:param s: smoothing factor.
 	:param norm: the wvalength of normalisation.
    ------------
    :return Bnu: y-values of a double-broken power law evaluated at x.
    """

	aTerm = (x/norm)**alpha1
	apow = abs(alpha2 - alpha1)*s
	Bnu = (aTerm * (1. + (x/lambdab)**apow)**(np.sign(alpha2 - alpha1)/s))/((1. + (norm/lambdab)**apow)**(np.sign(alpha2-alpha1)/s))

	return Bnu

@njit(fastmath = True)
def Simodel(x, p_w, dSi):

	"""
    This function calculates silicate emission peaking at p_w micron.
    ------------
    :param x: x-values.
	:param p_w: peak wavelength.
	:param dSi: shift from the peak.
    ------------
    :return Bnu: y-values of the silicate emission evaluated at x and peaking at p_w
    """

	alpha1 = 18.
	alpha2 = -5.
	if p_w < 15.:
		lambdab = p_w + dSi - 1.
	if p_w > 15.:
		lambdab = p_w + dSi - 1.4
	s = 0.6

	Bnu = x**alpha1*(1. + (x/lambdab)**(abs(alpha2-alpha1)*s))**(np.sign(alpha2-alpha1)/s)

	return Bnu/Bnu.max()

def drude(x, gamma_r, lambda_r, normed = True):
	"""
    This function calculates a Drude profile.
    ------------
    :param x: x-values.
    :param gamma_r: central wavelengths.
    :param lambda_r: fractional FWHM.
    ------------
    :keyword normed: if set to True, normalise to the maximum value.
    ------------
    :return drudeVal: the Drude profile evaluated at x.
    """
	numerateur = gamma_r**2.
	denominateur = (x/lambda_r - lambda_r/x)**2. + gamma_r**2.
	drudeVal = numerateur/denominateur

	if normed == True:
		return drudeVal/np.max(drudeVal)
	else:
		return np.max(drudeVal), drudeVal

def calc_S9p7(wavRest, flux):
	"""
    This function calculates the total obscuration at 9.7micron.
    ------------
    :param wavRest: rest-wavelength.
    :param flux: observed fluxes.
    ------------
    :return S9p7: the total obscuration at 9.7micron.
    """

	loc1 = np.where((wavRest >= 6.7) & (wavRest<=6.9))[0]
	loc2 = np.where((wavRest >= 29.) & (wavRest<=31.))[0]
	if len(loc2) == 0.:
		loc2 = np.where((wavRest >= 14.7) & (wavRest<=15.4))[0]

	if (len(loc1) <1) & (len(loc2) < 1):
	 	raise Exception()
	k = 1

	# Get the flux in the anchored wavelengths
	wavAbs = np.concatenate((wavRest[loc1], wavRest[loc2]))
	fluxAbs = np.concatenate((flux[loc1], flux[loc2]))

	# Calculate the continuum flux (unabsorbed)
	o = np.where(np.diff(wavAbs) > 0.)[0]
	spl = UnivariateSpline(np.log10(wavAbs[o]), np.log10(fluxAbs[o]), k=k)
	contFlux9p7 = np.interp(9.7,wavRest, 10**spl(np.log10(wavRest)))

	# Get the observed flux (absorbed)
	obsFlux9p7 = np.interp(9.6,wavRest, flux)

	# Ratio of the observed to continuum
	S9p7 = -np.log(obsFlux9p7/contFlux9p7)

	if (S9p7 <= 0.):
		return -99.
	else:
		return S9p7

def getExtCurve(ExtCurve):
	"""
    This function opens the right extinctin curve.
    ------------
    :param ExtCurve: contain the name of the extinction curve.
    ------------
    :return wav, tau: the wavelength and the extinction curve.
    """

	# Open the right extinction curve
	if ExtCurve == 'iragnsep':
		EC = pd.read_csv(path_iragnsep+'/ExtCurves/iragnsep_extCurve.csv')
	elif ExtCurve == 'PAHfit':
		EC = pd.read_csv(path_iragnsep+'/ExtCurves/PAHfit_extCurve.csv')
	elif ExtCurve == 'Min07':
		EC = pd.read_csv(path_iragnsep+'/ExtCurves/Min+07.csv')
	elif ExtCurve == 'CT06':
		EC = pd.read_csv(path_iragnsep+'/ExtCurves/CT06_extCurve.csv')

	return EC['lambda_mic'].values, EC['tau'].values

def S9p7toTau9p7(S9p7, source):
	"""
    This function transforms the total obscuration at 9.7 micron to the optical depth at 9.7.
    ------------
    :param S9p7: the total extinction at 9.7 micron.
    :param source: if 'gal' uses the assumption of a uniformly mixed media, if 'AGN' uses a screen of dust.
    ------------
    :return tau9.7: optical depth at 9.7 micron.
    """

	if source == 'gal':
		tauvec = 10**np.arange(-5., 5., 0.01)
		S9p7vec = (1. - np.exp(-tauvec))/tauvec
		f = interp1d(S9p7vec, tauvec)

		fluxRatio = np.exp(-S9p7)

		return np.round(np.array([f(fluxRatio)])[0],3)

	if source == 'AGN':

		return S9p7