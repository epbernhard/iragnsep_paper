import os
import iragnsep
import pandas as pd

from .func import getFluxInFilt

class modelToSED:
	"""
    This class calculates the synthetic fluxes within a filter given a model nuLnu and a redshift.
    """
	z = 0.1

	def __init__(self, lambda_mic, nuLnu, z = z):
		"""Initialise to z = 0.1, if not provided."""
	
		self.lambda_mic = lambda_mic
		self.z = z
		self.nuLnu = nuLnu

	def ISOPHTC100(self):
		filename = 'ISOPHTC100Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency

		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAC1(self):
		filename = 'IRAC1Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency

		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAC2(self):
		filename = 'IRAC2Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency

		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAC3(self):
		filename = 'IRAC3Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency

		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAC4(self):
		filename = 'IRAC4Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency

		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAS12(self):
		filename = 'IRAS12Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def MIPS24(self):
		filename = 'MIPS24Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAS60(self):
		filename = 'IRAS60Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def MIPS70(self):
		filename = 'MIPS70Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def MIPS160(self):
		filename = 'MIPS160Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def PACS70(self):
		filename = 'PACS70Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def PACS100(self):
		filename = 'PACS100Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def IRAS100(self):
		filename = 'IRAS100Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def PACS160(self):
		filename = 'PACS160Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def SPIRE250ps(self):
		filename = 'SPIRE250psFilter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def SPIRE350ps(self):
		filename = 'SPIRE350psFilter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def SPIRE500ps(self):
		filename = 'SPIRE500psFilter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def WISE_W1(self):
		filename = 'WISE_W1Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def WISE_W2(self):
		filename = 'WISE_W2Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy


	def WISE_W3(self):
		filename = 'WISE_W3Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy

	def WISE_W4(self):
		filename = 'WISE_W4Filter.csv'

		path = os.path.dirname(iragnsep.__file__)+'/Filters/'
		filt = pd.read_csv(path+filename)  # (Energy Counter)
		filt_wav = filt.lambda_ang*1e-4 #Micron (Window of the filter)
		filt_QE = filt.QE # Quantum Efficiency
		
		flux = getFluxInFilt(filt_wav, filt_QE, self.lambda_mic, self.nuLnu, self.z)

		return flux #Jy