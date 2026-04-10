import numpy as np
import matplotlib.pyplot as plt

class interception(object):

	def __init__(self, env_state, data_in):
		
		# AET_dt:	Actual evapotranspiration
		
		#self.av = np.zeros(env_state.grid_size)
		#self.savi_max = np.ones(env_state.grid_size)
		#self.savi_min = np.zeros(env_state.grid_size)
		print("Run Interception component")
		
	def run_interception_one_step(self, rain, ETo, av,
		SAVI, savi_max, savi_min, LAI, lai_a, lai_b, fcw, Sc0, agents, kc):
		"""	Canopy compartment, calculates interception and
		evaporation from canopy.
		
		PARAMETERS:
		-----------
		rain:		precipitation
		av:			fraction of vegetation cover [-]
		SAVI:		Soil-Adjusted Vegetation Index
		Kc:			crop coefficient factor
		env_state:	grid:	z:		Topograhic elevation
							h:		water table
		
		fcw:		biome-dependent coeficient		
		Scz0:		initial water content canopy
		
		Returns
		-------
		Pth:	Throughfall, precipitation minus interceptionn
		Ecw:	Canopy evaporation
		Scz:	Canopy water storage
		PET:	Potential evapotranspiration after canopy evaporation
		LAI:	Leaf Area Index
		Kc:		Crop factor
		"""
				
		#if Kc is not available:
		if av is not None:
			# Estimation of crop factor

			if agents.model.current_time >= agents.model.config['general']['start_time']:
				if SAVI is not None:
					SAVI = np.where((agents.adapt_measure_5_grid.flatten() == 1), np.minimum(SAVI + 0.15, savi_max), SAVI) # Increase SAVI by 0.15 if agents have installed soil moisture techniques
					Kc = kc
			else:
				Kc = kc
			
			if LAI is None:		
				# Estimation of Leaf area index
				LAI = get_LAI_from_SAVI(SAVI, lai_a, lai_b)
			else:
				LAI = 0
			
			# Maximum amount of water store by canopy

			if agents.model.current_time >= agents.model.config['general']['start_time']:

				av = np.where(agents.adapt_measure_5_grid.flatten() == 1, np.minimum(av + 0.1, 1.0), av) # increase av if agents have installed soil moisture techniques
			
			Sca_max = av*get_Scmax_from_LAI(LAI) 
			
			# maximum canopy saturation
			#Sca = Sca_max*(1-np.exp(-rain/Sca_max))
			Ecw = Sc0 + rain
			
			Ecw = np.where(Ecw > ETo, ETo, Ecw)

			# Potential canopy evaporation
			#Eca = av*ETo*Sc0/Sca_max
			
			Sca = Sc0 - Ecw + rain
			
			# Interception
			Pth = Sca - Sca_max

			Scz = np.where(Pth > 0, Sca_max, Sca)

			Pth = np.where(Pth > 0, Pth, 0.0)

			# limit evaporation to the amount of water available
			Pth[Pth < 0] = 0
			
			# Potential evapotranspiration after canopy evaporation
			PET = (ETo - Ecw)

			PET[PET < 0] = 0.0	
					
		else:
			Eca = None
			LAI = None
			Kc = None
			Pth = rain
			PET = ETo
			Sc = None
			
		return Pth, Eca, PET, LAI, Kc, Sc
		
	
def get_vegetation_factor(savi, savi_min, savi_max):
	"""Crop factor calculation kc for ETo
	
	PARAMETERS:
	-----------
	savi:		Soil-Adjusted Vegetation Index
	savi_min: 	min Soil-Adjusted Vegetation Index
	savi_max: 	max Soil-Adjusted Vegetation Index
	
	OUTPUT:
	-------
	kc:			Crop factor
	"""
	
	nu = 1.0
	
	kc = 1 - (savi_max - savi)/(savi_max - savi_min)
	
	#kc = kc**nu
	
	return kc
		

def get_LAI_from_SAVI(SAVI, a, b):
	"""Calculate LAI from SAVI
	
	PARAMETERS:
	----------
	SAVI:	Soil adjusted vegetation index
	a:		Coeficient of exponential function
	b: 		power value for exponential function
	
	OUTPUT:
	------
	LAI:	leaf area index
	"""

	LAI = a*np.exp(b*SAVI) 

	return LAI
	

def get_Scmax_from_LAI(LAI):
	"""Calculate maximum amount of water store by canopy
	
	PARAMETERS:
	----------
	LAI:	leaf area index
	
	OUTPUT:
	------
	Sca_max:	Maximum amount of water store by canopy
	"""
	
	Sca_max = 0.935 + 0.498*LAI - 0.00575*(LAI**2)
	
	return Sca_max
	
def get_Scmax_from_LAI_and_fcw(fcw, LAI, agents):
	"""Calculate maximum amount of water store by canopy
	exponential approach
	http://refhub.elsevier.com/S0022-1694(19)30648-1/h0170
	PARAMETERS:
	----------
	LAI:	leaf area index [-]
	fcw:	biome-dependent coeficient [-]

	1. Evergreen needle-leaved forest	0.06
	2. Evergreen broad-leaved	0.02
	3. Dry deciduous forest	0.02
	4. Cold deciduous forest	0.06
	5. Mediterranean veg. shrubs 0.02
	6. Grasslands	0.01

	OUTPUT:
	------
	Sca_max:	Maximum amount of water store by canopy
	"""
	if agents.model.current_time >= agents.model.config['general']['start_time']:
		fcw = np.where((agents.land_use_flat >= 112) | (agents.adapt_measure_5_grid.flatten() ==1) , 0.02, 0.01) # where agents have installed soil moisture techniques
	else:
		fcw = np.where((agents.land_use_flat >= 112), 0.02, 0.01)
	Sca_max = fcw*np.log(1+LAI)
	
	return Sca_max