import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import iragnsep

from .func import S9p7toTau9p7, getExtCurve, contmodel, Simodel, nuLnuToFnu
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

font0 = FontProperties(family = 'serif', variant = 'small-caps', size = 22)


############################
## Plot all the results for all the possible fit. Tag the best model in blue.
############################
def plotFitSpec(df, wavSpec, fluxSpec, efluxSpec,\
				wavPhot, fluxPhot, efluxPhot, \
				UL = np.array([]), pathFig = './', \
				sourceName = 'NoName', templ = '', z = 0.01, saveRes = True, ExtCurve = 'iragnsep'):
	

	"""
    This function plots the results of the fits for data which contain spectral + photometry fluxes.
    ------------
    :param df: data-frame contaning the results of the fits (i.e. optimised parameters) as returned by the function iragnsep.
    :param wavSpec: observed wavelengthts for the spectrum (in microns).
    :param fluxSpec: observed fluxes for the spectrum (in Jansky).
    :param efluxSpec: observed uncertainties on the fluxes for the spectrum (in Jansky).
    :param wavPhot: observed wavelengths for the photometry (in microns).
    :param fluxPhot: observed fluxes for the photometry (in Jansky).
    :param efluxPhot: observed uncertainties on the fluxes for the photometry (in Jansky).
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword UL: vector of length Nphot, where Nphot is the number of photometric data. If any of the values is set to 1, the corresponding flux is set has an upper limit in the fit. Default = [].
  	:keyword sourceName: name of the source. Default = 'NoName'.
	:keyword pathFig: if saveRes is set to True, the figues showing the results of the fits will be saved at the location pathFig. Default = './'.
	:keyword saveRes: if set to True, the tables containing the results of the fits as well as the figures are saved. Default = True.
	:keyword templ: set the templates that have been used in the fits.
	:keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
	------------
    :return 0
    """
	
	path_iragnsep = os.path.dirname(iragnsep.__file__)

    # concatenate the spectral and photo fluxes
	flux = np.concatenate([fluxSpec, fluxPhot])

	# define the upper limits if not.
	if len(UL) == 0.:
		UL = np.zeros(len(wavPhot))

	# define the templates to Bernhard+20 if not.
	if len(templ) == 0:
		path = os.path.dirname(iragnsep.__file__)
		templ = pd.read_csv(path+'/iragnsep_templ.csv')

	# Extract the name of the templates
	keys = templ.keys().values
	nameTempl_gal = []
	nameTempl_PAH = []
	for key in keys:
		if str(key).startswith('gal'):
			if str(key).endswith('PAH') == False:
				nameTempl_gal.append(key)
			else:
				nameTempl_PAH.append(key)

	# Test that we have templates for everything (if no galaxy then it crashes)
	if len(nameTempl_gal) == 0:
		raise ValueError('The galaxy template does not exist. The name of the column defining nuLnu for the galaxy template needs to start with "gal".')
	
	# define the wavelengths
	try:
		wavTempl = templ['lambda_mic'].values
	except:
		raise ValueError('Rename the wavelengths column of the template "lambda_mic".')

	# Open 2 figures, 1 for all the possible fits and 1 for the best weighted average model.
	fig1, axs1 = plt.subplots(4,4, sharey = True, figsize = (20, 18))
	fig1.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.08, right = 0.99, bottom = 0.06, top = 0.98)
	axs1 = axs1.ravel()
	axs1[14].axis("off")
	axs1[15].axis("off")

	# legend on the first figure
	line1 = Line2D([], [], label=r'$Herschel$ photometry', color='k', marker='o', markeredgecolor='k', markeredgewidth=1.5, markersize=10, mfc = 'None')
	line2 = Line2D([], [], label=r'$Spitzer$-IRS spectrum', color='k', marker='o', markeredgecolor='k', markeredgewidth=0., markersize=5, mfc = 'None')
	line3 = Line2D([1], [1], label='Best fit (Total)', color='k', lw = 2)
	line4 = Line2D([1], [1], label='Galaxy', ls = '--', color='#66757F', lw = 2)
	line5 = Line2D([1], [1], label='AGN', ls = '-.', color='#66757F', lw = 2)
	axs1[13].legend(frameon = False, handles=[line1, line2, line3, line4, line5], numpoints=1, bbox_to_anchor=(2.9, 0.1), loc='lower right', fontsize = 25, ncol = 2)

	# Open a cmap for the AW
	cmap = cm.get_cmap('magma', 101)
	cmap = ListedColormap(cmap(np.linspace(0.05, 0.7, 101)))

	fig2, axs2 = plt.subplots(figsize = (11, 8))
	fig2.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.15, right = 0.95, bottom = 0.15, top = 0.95)
	axs2.tick_params(axis='both', labelcolor='k', labelsize = 25, width = 1, size = 15, which = 'major', direction = 'inout')
	axs2.tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')

	# Open the extincrtion curve
	S9p7 = df['S9p7'].values[0]
	if S9p7 > 0.:
		EC_wav, EC_tau = getExtCurve(ExtCurve)
		EC_wav_AGN, EC_tau_AGN = getExtCurve('PAHfit')

		#Calculate tau9p7 for the galaxy
		tau9p7_gal = S9p7toTau9p7(S9p7, 'gal')
		tau9p7_AGN = S9p7toTau9p7(S9p7, 'AGN')

		# Galaxy
		tau = np.interp(wavTempl, EC_wav, EC_tau) * tau9p7_gal
		extCorr_gal = ((1. - np.exp(-tau))/tau)

		#AGN
		tau = np.interp(wavTempl, EC_wav_AGN, EC_tau_AGN) * tau9p7_AGN
		extCorr_AGN = np.exp(-tau)
	else:
		extCorr_gal = 1.
		extCorr_AGN = 1.

	# Loop on all the possible combinations of models
	count = 0
	for i in range(0, len(df)):

		# Herschel fluxes
		o = np.where(UL < 1.)[0]
		axs1[count].errorbar(wavPhot[o]/(1.+z), fluxPhot[o], yerr = efluxPhot[o], fmt = 'o', color = 'k', alpha = 1.0, mew = 1, mfc = 'None', ms = 10, zorder = 1)
		o = np.where(UL > 0.)[0]
		axs1[count].errorbar(wavPhot[o]/(1.+z), fluxPhot[o], yerr = fluxPhot[o]/3., fmt = 'o', color = 'k', alpha = 1.0, uplims = True, mew = 1, mfc = 'None', ms = 10)
		axs1[count].errorbar(wavSpec/(1.+z), fluxSpec, yerr = efluxSpec, fmt = 'o', color = 'k', alpha = 0.7, mew = 0.5, mfc = 'None', ms = 5, elinewidth = 0.3, zorder = 1)

		obj = df.iloc[i]

		# Extract the optimised best parameters and their uncertainties
		normDust = 10**(obj['logNormGal_dust'])
		nuLnuDust = normDust * templ[obj['tplName']].values
		enuLnuDust = np.sqrt((nuLnuDust * np.log(10) * obj['elogNormGal_dust'])**2. + (normDust * templ['e'+obj['tplName']].values)**2.)

		normPAH = 10**obj['logNormGal_PAH']
		nuLnuPAH = normPAH * templ['gal_PAH'].values
		enuLnuPAH = np.sqrt((nuLnuPAH * np.log(10) * obj['elogNormGal_PAH'])**2. + (normPAH * templ['egal_PAH'].values)**2.)

		# Differential obscured galaxy emission
		nuLnuGal = nuLnuDust * extCorr_gal + nuLnuPAH
		enuLnuGal = np.sqrt((enuLnuDust * extCorr_gal)**2. + enuLnuPAH**2.)

		# Calculate the observed flux, affected by attenuation
		FnuGal = nuLnuToFnu(wavTempl, nuLnuGal, z)
		FnuGal_i = []
		for MCi in range(0, 1000):
			FnuGal_i.append(nuLnuToFnu(wavTempl, np.random.normal(nuLnuGal, enuLnuGal), z))		
		eFnuGal = np.nanstd(FnuGal_i, axis = 0)

		# Define the AGN contribution
		if obj['AGNon'] == 1:
			
			modelPL = 10**(obj['logNormAGN_PL']) * contmodel(np.array([obj['lBreak_PL'], obj['alpha1'], obj['alpha2'], obj['alpha3']]), wavTempl)
			modelPL_i = []
			for MCi in range(0, 1000):
				modelPL_i.append(10**(np.random.normal(obj['logNormAGN_PL'],obj['elogNormAGN_PL'])) * contmodel(np.array([np.random.normal(obj['lBreak_PL'], obj['elBreak_PL']), \
					np.random.normal(obj['alpha1'], obj['ealpha1']), np.random.normal(obj['alpha2'], obj['ealpha2']), np.random.normal(obj['alpha3'],obj['ealpha3'])]), wavTempl))

			emodelPL = np.std(modelPL_i, axis=0)
		
			modelSi11 = 10**(obj['logNorm_Si']) * Simodel(wavTempl, 11., obj['dSi'])
			emodelSi11 = modelSi11 * np.log(10) * obj['elogNorm_Si']

			modelSi18 = 10**(obj['logNorm_Si']) * Simodel(wavTempl, 18., obj['dSi'])
			emodelSi18 = modelSi18 * np.log(10) * obj['elogNorm_Si']

			FnuAGN = (modelPL + modelSi11 + modelSi18) * extCorr_AGN
			eFnuAGN = np.sqrt(emodelSi11**2. + emodelSi18**2. + emodelPL**2.) * extCorr_AGN

			FnuTot = FnuGal + FnuAGN
			eFnuTot = np.sqrt(eFnuGal**2. + eFnuAGN**2.)
		else:
			FnuTot = FnuGal
			eFnuTot = eFnuGal

		axs1[count].plot(wavTempl, FnuTot, '-', c = cmap.colors[int(obj['Aw']*100.)], linewidth = 3, alpha = 0.7, zorder = 2)

		if obj['AGNon'] == 1:
			axs1[count].plot(wavTempl, FnuGal, '--', color = '#66757F')
			axs1[count].plot(wavTempl, FnuAGN, '-.', color = '#66757F')

		axs1[count].fill_between(wavTempl, FnuTot - eFnuTot, FnuTot + eFnuTot, color = '#8c8888', alpha = 1., zorder=0)

		axs1[count].text(15./(1.+z), min(flux)/3., 'Aw = '+str(round(obj['Aw']*100.))+'%', fontsize = 25, c = cmap.colors[int(obj['Aw']*100.)])
		if obj['AGNon'] == 1:
			axs1[count].text(5./(1.+z), max(flux)*2.5, obj['tplName']+' + AGN', fontsize = 25)
		else:
			axs1[count].text(5./(1.+z), max(flux)*2.5, obj['tplName'], fontsize = 25)

		axs1[count].tick_params(axis='both', labelcolor='k', labelsize = 25, width = 1, size = 15, which = 'major', direction = 'inout')
		axs1[count].tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')
		axs1[count].set_xscale('log')
		axs1[count].set_yscale('log')
		axs1[count].set_xlim([3./(1.+z), 800./(1.+z)])
		axs1[count].set_ylim([min(flux)/5., max(flux)*5.])
		axs1[count].set_xlabel(r'$\lambda_{\rm rest}\ (\mu {\rm m})$', fontsize = 28)
		if (count == 0) or (count == 4) or (count == 8) or (count == 12):
			axs1[count].set_ylabel(r'Flux (Jy)', fontsize = 28)
		count += 1

		# Best model Figure
		if obj['Aw'] > 0.01:
			if obj['AGNon'] == 1:
				axs2.plot(wavTempl, FnuTot, '-', linewidth = 1, label = obj['tplName']+' + AGN ('+str(round(obj['Aw']*100.))+'%)', alpha = 0.3, c = cmap.colors[int(obj['Aw']*100.)])				
			else:
				axs2.plot(wavTempl, FnuTot, '-', linewidth = 1, label = obj['tplName']+' ('+str(round(obj['Aw']*100.))+'%)', alpha = 0.3, c = cmap.colors[int(obj['Aw']*100.)])

		try:
			FnuTot_Aw += FnuTot * obj['Aw']
			eFnuTot_Aw += (eFnuTot*obj['Aw'])**2.
			FnuGal_Aw += FnuGal * obj['Aw']
			if obj['AGNon'] == 1:
				FnuAGN_Aw += FnuAGN * obj['Aw']
		except:
			FnuTot_Aw = FnuTot * obj['Aw']
			eFnuTot_Aw = (eFnuTot * obj['Aw'])**2.
			FnuGal_Aw = FnuGal * obj['Aw']
			if obj['AGNon'] == 1:
				FnuAGN_Aw = FnuAGN * obj['Aw']
			else:
				FnuAGN_Aw = FnuGal_Aw * 0.

	axs2.plot(wavTempl, FnuTot_Aw, ls = '-', color = 'k', linewidth = 3, label = 'Best weighted fit [Total]', alpha = 0.8, zorder = 2)
	axs2.plot(wavTempl, FnuGal_Aw, '--', color = '#E94B3C', linewidth = 2, label = 'Best weighted fit [Galaxy]', alpha = 0.7)
	axs2.plot(wavTempl, FnuAGN_Aw, '-.', color = '#6395F2', linewidth = 2, label = 'Best weighted fit [AGN]', alpha = 0.7)

	o = np.where(UL < 1.)[0]
	axs2.errorbar(wavPhot[o]/(1.+z), fluxPhot[o], yerr = efluxPhot[o], fmt = 'o', color = 'k', alpha = 1.0, mew = 1, mfc = 'None', ms = 15, label = 'Observed Photometry')
	o = np.where(UL > 0.)[0]
	axs2.errorbar(wavPhot[o]/(1.+z), fluxPhot[o], yerr = fluxPhot[o]/3., fmt = 'o', color = 'k', alpha = 1.0, uplims = True, mew = 1, mfc = 'None', ms = 10)
	axs2.errorbar(wavSpec/(1.+z), fluxSpec, yerr = efluxSpec, fmt = 'o', color = 'k', alpha = 0.7, mew = 0.5, mfc = 'None', ms = 5, elinewidth = 0.3, zorder = 1, label = 'Observed Spectrum')

	axs2.fill_between(wavTempl, FnuTot_Aw - np.sqrt(eFnuTot_Aw), FnuTot_Aw + np.sqrt(eFnuTot_Aw), color = '#8c8888', alpha = 0.8, zorder=0)

	axs2.set_xscale('log')
	axs2.set_yscale('log')
	axs2.set_xlim([3./(1.+z), 800./(1.+z)])
	axs2.set_ylim([min(flux)/5., max(flux)*10.])
	axs2.set_xlabel(r'$\lambda_{\rm rest}\ (\mu {\rm m})$', fontproperties = font0)
	axs2.set_ylabel(r'Flux (Jy)', fontproperties = font0)
	axs2.legend(frameon = False, fontsize = 16, ncol = 2)

	# If saveRes is True, then save the figures to the location specified in pathFig.
	if saveRes == True:
		fig1.savefig(pathFig+sourceName+'_fitResAll_spec.pdf')
		fig2.savefig(pathFig+sourceName+'_fitResBM_spec.pdf')
	else:
		plt.show()
	plt.close('all')


############################
## Plot all the results for all the possible fit. Tag the best model in blue.
############################
def plotFitPhoto(df, wav, flux, eflux, UL = np.array([]), pathFig = './', sourceName = 'NoName', templ = '', z = 0.01, saveRes = True, NOAGN = False, \
				 ExtCurve = 'iragnsep'):
	
	"""
    This function plots the results of the fits for photometric data.
    ------------
    :param df: data-frame contaning the results of the fits (i.e. optimised parameters) as returned by the function SEDanalysis.runSEDspecFit.
    :param wav: observed wavelengthts (in microns).
    :param flux: observed fluxes (in Jansky).
    :param eflux: observed uncertainties on the fluxes (in Jansky).
    ------------
    :keyword z: redshift of the source. Default = 0.01.
    :keyword UL: vector of length Nphot, where Nphot is the number of photometric data. If any of the values is set to 1, the corresponding flux is set has an upper limit in the fit. Default = [].
  	:keyword sourceName: name of the source. Default = 'NoName'.
	:keyword pathFig: if saveRes is set to True, the figues showing the results of the fits will be saved at the location pathFig. Default = './'.
	:keyword saveRes: if set to True, the tables containing the results of the fits as well as the figures are saved. Default = True.
	:keyword templ: set the templates that have been used in the fits.
    :keyword ExtCurve: pass the name of the extinction curve to use. Default = 'iragnsep'.
	------------
    :return 0
    """

    # Open the Bernhard+20 templates if not defined
	if len(templ) == 0:
		path = os.path.dirname(iragnsep.__file__)
		templ = pd.read_csv(path+'/iragnsep_templ.csv')

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
		if str(key).startswith('AGN') == True:
			if str(key).endswith('Siem') == True:
				nameTempl_Siem.append(key)
			else:
				nameTempl_AGN.append(key)

	# Test that we have template for everything (if no galaxy then it crashes)
	if len(nameTempl_gal) == 0:
		raise ValueError('The galaxy template does not exist. The name of the column defining nuLnu for the galaxy template needs to start with "gal".')
	if len(nameTempl_AGN) == 0:
		raise ValueError('The AGN template does not exist. The name of the column defining nuLnu for the AGN template needs to start with "AGN".')

	# define the wavelengths
	try:
		wavTempl = templ['lambda_mic']
	except:
		raise ValueError('Rename the wavelengths column of the template "lambda_mic".')

	# Open 2 figures, 1 for all the possible fits and 1 for the best weighted average model.
	fig1, axs1 = plt.subplots(4,6, sharex = False, sharey = True, figsize = (27, 18))
	fig1.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.06, right = 0.99, bottom = 0.10, top = 0.95)
	axs1 = axs1.ravel()
	axs1[21].axis("off")
	axs1[22].axis("off")
	axs1[23].axis("off")

	# legend on the first figure
	line1 = Line2D([], [], label='Observed SED', color='k', marker='o', markeredgecolor='k', markeredgewidth=1.5, markersize=10, mfc = 'None')
	line2 = Line2D([1], [1], label='Best fit (Total)', color='k', lw = 2)
	line3 = Line2D([1], [1], label='Galaxy', ls = '--', color='#66757F', lw = 2)
	line4 = Line2D([1], [1], label='AGN', ls = '-.', color='#66757F', lw = 2)
	axs1[20].legend(frameon = False, handles=[line1, line2, line3, line4], numpoints=1, bbox_to_anchor=(3., 0.1), loc='lower right', fontsize = 25, ncol = 2)

	fig2, axs2 = plt.subplots(figsize = (11, 8))
	fig2.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.15, right = 0.95, bottom = 0.15, top = 0.95)
	axs2.tick_params(axis='both', labelcolor='k', labelsize = 25, width = 1, size = 15, which = 'major', direction = 'inout')
	axs2.tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')

	# Open a cmap for the AW
	cmap = cm.get_cmap('magma', 101)
	cmap = ListedColormap(cmap(np.linspace(0.05, 0.7, 101)))

	# Loop over all the model combinations of templates.
	count = 0
	for i in range(0, len(df)):
		# Plot with all the possible models.
		o = np.where(UL < 1.)[0]
		axs1[count].errorbar(wav[o]/(1.+z), flux[o], yerr = eflux[o], fmt = 'o', color = 'k', alpha = 1.0, label = 'Observed SED', mew = 1, mfc = 'None', ms = 10)
		o = np.where(UL > 0.)[0]
		axs1[count].errorbar(wav[o]/(1.+z), flux[o], yerr = flux[o]/3., fmt = 'o', color = 'k', alpha = 1.0, uplims = True, mew = 1, mfc = 'None', ms = 10)

		obj = df.iloc[i]

		# Open the extincrtion curve
		EC_wav, EC_tau = getExtCurve(ExtCurve)
		EC_wav_AGN, EC_tau_AGN = getExtCurve('PAHfit')

		S9p7 = obj['S9p7']
		if S9p7 > 0.:
			#Calculate tau9p7 for the galaxy
			tau9p7_gal = S9p7toTau9p7(S9p7, 'gal')
			tau9p7_AGN = S9p7toTau9p7(S9p7, 'AGN')

			# Galaxy
			tau = np.interp(wavTempl, EC_wav, EC_tau) * tau9p7_gal
			obsPerWav_gal = ((1. - np.exp(-tau))/tau)

			#AGN
			tau = np.interp(wavTempl, EC_wav_AGN, EC_tau_AGN) * tau9p7_AGN
			obsPerWav_AGN = np.exp(-tau)
		else:
			obsPerWav_gal = 1.
			obsPerWav_AGN = 1.

		# Exrract the optimised parameters and their uncertainties
		normDust = 10**obj['logNormGal_dust']
		nuLnuDust = (normDust * templ[obj['tplName_gal']].values)*obsPerWav_gal
		enuLnuDust = (normDust * templ['e'+obj['tplName_gal']].values)*obsPerWav_gal

		normPAH = 10**obj['logNormGal_PAH']
		nuLnuPAH = normPAH * templ['gal_PAH'].values
		enuLnuPAH = normPAH * templ['egal_PAH'].values

		nuLnuGal = nuLnuDust + nuLnuPAH
		enuLnuGal = np.sqrt(enuLnuDust**2. + enuLnuPAH**2.)

		# Calculate the observed model fluxes
		FnuGal = nuLnuToFnu(wavTempl, nuLnuGal, z).values
		eFnuGal = nuLnuToFnu(wavTempl, enuLnuGal, z).values

		# Define the optimised model AGN template
		if obj['AGNon'] == 1:
			normAGN = 10**obj['logNormAGN']
			nuLnuAGN = normAGN * templ[obj['tplName_AGN']].values
			enuLnuAGN_up = normAGN * templ['e'+obj['tplName_AGN']+'_up'].values
			enuLnuAGN_down = normAGN * templ['e'+obj['tplName_AGN']+'_down'].values
			
			normSi = 10**obj['logNormSiem']
			nuLnuSi = normSi * templ[nameTempl_Siem[0]].values
			enuLnuSi_up = normSi * templ['e'+nameTempl_Siem[0]+'_up'].values
			enuLnuSi_down = normSi * templ['e'+nameTempl_Siem[0]+'_down'].values
		
			FnuAGN =  nuLnuToFnu(wavTempl, nuLnuAGN + nuLnuSi, z) * obsPerWav_AGN

			FnuTot = FnuGal + FnuAGN

			enuLnuTot_up = np.sqrt(enuLnuDust**2. + enuLnuPAH**2. + enuLnuAGN_up**2. + enuLnuSi_up**2.)
			eFnuTot_up = nuLnuToFnu(wavTempl, enuLnuTot_up, z)
			enuLnuTot_down = np.sqrt(enuLnuDust**2. + enuLnuPAH**2. + enuLnuAGN_down**2. + enuLnuSi_down**2.)
			eFnuTot_down = nuLnuToFnu(wavTempl, enuLnuTot_down, z)

			axs1[count].plot(wavTempl, FnuGal, '--', color = '#66757F')
			axs1[count].plot(wavTempl, FnuAGN, '-.', color = '#66757F')
			axs1[count].plot(wavTempl, FnuTot, '-', c = cmap.colors[int(obj['Aw']*100.)], linewidth = 3, alpha = 0.7)

		else:
			FnuTot = FnuGal
			eFnuTot_up = eFnuGal
			eFnuTot_down = eFnuGal

			axs1[count].plot(wavTempl, FnuTot, 'k-', c = cmap.colors[int(obj['Aw']*100.)], linewidth = 3, alpha = 0.7)

		axs1[count].set_xscale('log')
		axs1[count].set_yscale('log')
		axs1[count].set_xlim([3./(1.+z), 800./(1.+z)])
		
		if NOAGN != True:
			axs1[count].set_ylim([min(flux)/5., max(flux)*5.])
		else:
			axs1[count].set_ylim([min(flux)/5., max(FnuTot)*5.])
		
		axs1[count].set_xlabel(r'$\lambda_{\rm rest}\ (\mu {\rm m})$', fontsize = 28)
		
		if (count == 0) or (count == 6) or (count == 12) or (count == 18):
			axs1[count].set_ylabel(r'Flux (Jy)', fontsize = 28)
		
		axs1[count].text(10., min(flux)/4., 'Aw = '+str(round(obj['Aw']*100.))+'%', fontsize = 25, c = cmap.colors[int(obj['Aw']*100.)])
		if obj['AGNon'] == 1:
			axs1[count].text(5./(1.+z), max(flux)*2.5, obj['tplName_gal']+'+'+obj['tplName_AGN'], fontsize = 25)
		else:
			axs1[count].text(5./(1.+z), max(flux)*2.5, obj['tplName_gal'], fontsize = 25)
		axs1[count].tick_params(axis='both', labelcolor='k', labelsize = 25, width = 1, size = 15, which = 'major', direction = 'inout')
		axs1[count].tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')

		axs1[count].fill_between(wavTempl, FnuTot - eFnuTot_down, FnuTot + eFnuTot_up, color = '#bab8b1', alpha = 0.6)

		count += 1

		# Plots with the best model only.
		if obj['Aw'] > 0.05:
			if obj['AGNon'] == 1:
				axs2.plot(wavTempl, FnuTot, '-', color = 'k', \
						 linewidth = 1, label = obj['tplName_gal']+' + '+ obj['tplName_AGN'] + ' ('+str(round(obj['Aw']*100.))+'%)', alpha = obj['Aw']/1.5)
			else:
				axs2.plot(wavTempl, FnuTot, '-', color = 'k', \
						 linewidth = 1, label = obj['tplName_gal']+' ('+str(round(obj['Aw']*100.))+'%)', alpha = obj['Aw'])

			try:
				FnuTot_Aw += FnuTot * obj['Aw']
				eFnuTot_Aw += (eFnuGal*obj['Aw'])**2.
				FnuGal_Aw += FnuGal * obj['Aw']
				if obj['AGNon'] == 1:
					FnuAGN_Aw += FnuAGN * obj['Aw']
			except:
				FnuTot_Aw = FnuTot * obj['Aw']
				eFnuTot_Aw = (eFnuGal * obj['Aw'])**2.
				FnuGal_Aw = FnuGal * obj['Aw']
				if obj['AGNon'] == 1:
					FnuAGN_Aw = FnuAGN * obj['Aw']
				else:
					FnuAGN_Aw = FnuGal_Aw * 0.
			

	axs2.errorbar(wavTempl, FnuTot_Aw, yerr = np.sqrt(eFnuTot_Aw), fmt = '-', color = 'k', elinewidth = 0.5, linewidth = 2, label = 'Best weighted fit [Total]',\
				 ecolor = 'k', alpha = 1., errorevery = 3)
	axs2.plot(wavTempl, FnuGal_Aw, '--', color = '#E94B3C', \
						 linewidth = 2, label = 'Best weighted fit [Galaxy]', alpha = 0.8)
	axs2.plot(wavTempl, FnuAGN_Aw, '-.', color = '#6395F2', \
						 linewidth = 2, label = 'Best weighted fit [AGN]', alpha = 0.8)

	o = np.where(UL < 1.)[0]
	axs2.errorbar(wav[o]/(1.+z), flux[o], yerr = eflux[o], fmt = 'o', color = '#292F33', alpha = 0.9, label = 'Observed SED', mfc = 'none', mew = 1, ms = 10)

	o = np.where(UL > 0.)[0]
	axs2.errorbar(wav[o]/(1.+z), flux[o], yerr = flux[o]/5., fmt = 'o', color = '#292F33', alpha = 0.9, label = '_no_label_', uplims = True, mfc = 'none', mew = 1, ms = 10)

	axs2.set_xscale('log')
	axs2.set_yscale('log')
	axs2.set_xlim([3./(1.+z), 800./(1.+z)])
	if NOAGN != True:
		axs2.set_ylim([min(flux)/5., max(flux)*10.])
	else:
		axs2.set_ylim([min(flux)/5., max(FnuGal_Aw)*10.])
	axs2.set_xlabel(r'$\lambda_{\rm rest}\ (\mu {\rm m})$', fontproperties = font0)
	axs2.set_ylabel(r'Flux (Jy)', fontproperties = font0)
	axs2.legend(frameon = False, fontsize = 16, ncol = 2)

	# if saveRes is True, save the figures at the locations pathFig.
	if saveRes == True:
		fig1.savefig(pathFig+sourceName+'_fitResAll_photo.pdf')
		fig2.savefig(pathFig+sourceName+'_fitResBM_photo.pdf')
	else:
		plt.show()
	plt.close('all')