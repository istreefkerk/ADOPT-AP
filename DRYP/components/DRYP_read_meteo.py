import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
from DRYP.components.DRYP_projection import reproject_dataset

class rainfall(object):
	"""Read all input dataset
	rain:	Rainfall
	ETo:	Potential evapotranspiration
	SAVI:	Soil-Adjusted Vegetation Index
	Kc:		Crop coeficient factor	
	"""
	def __init__(self, inputfile, env_state):
		""" Setting variables and time steps for reading inputs
		"""
		if inputfile.first_read == 1:
			t_end = inputfile.ndays#Sim_period.days
			
			if inputfile.dt != 60:
				date_sim_m = pd.date_range(inputfile.ini_date,
					periods=t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly,
					freq=str(np.int(inputfile.dt))+'min')
			freq_dt = 'H'
			if inputfile.dt > 60:
				freq_dt = str(np.int(inputfile.dt/60))+'H'
			
			# generate time series of houtly dates
			date_sim_h = pd.date_range(inputfile.ini_date,
										periods=t_end*inputfile.dt_hourly,
										freq=freq_dt)
			
			# generate time series of daily dates
			date_sim_d = pd.date_range(inputfile.ini_date, periods=t_end, freq='d')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
			
		# To be completed
		# this will allow to read just one netCDF files for the simulation
		# to read multiples files, take a look at input_datasets_bigfiles
		#inputfile.netcf_savi = 0
		#inputfile.netcf_kc = 0
		
		
		if inputfile.first_read == 1:			
			if inputfile.netcf_pre == 1: 
				# Read netCDF fiels
				fpre = Dataset(inputfile.fname_TSPre, 'r')
				
				# Change number to datetime
				time_pre_aux = num2date(fpre['time'][:-1],
								units=fpre['time'].units,
								calendar=fpre['time'].calendar)
				
				# change time nstimedate to datetime
				time_pre = []
				for iidate in time_pre_aux:
					if not iidate == None:
						time_pre.append(datetime(
							iidate.year,
							iidate.month,
							iidate.day,
							iidate.hour,
							iidate.minute)
							)
					else:
						time_pre.append(None)
				time_pre = np.array(time_pre)
			
			else:
				# Read time series of precipitation	csv
				fpre = pd.read_csv(inputfile.fname_TSPre)
				
				# change to txt to datetime
				fpre["Date"] = pd.to_datetime(fpre['Date'])#,format = '%d/%m/%Y %H:%M')
				
				if inputfile.dt > 60:
					# aggregate data to the model time step
					fpre.index = pd.DatetimeIndex(fpre['Date'])
					fpre = (fpre.resample(inputfile.Agg_method).sum()).reset_index()
				time_pre = fpre["Date"]
			
			# Read time series of evapotranspiration
			if inputfile.netcf_ETo == 1:
				#inputfile.netcf_savi = 1
				#inputfile.netcf_kc = 1
				
				# Read netCDF files
				dataETo = Dataset(inputfile.fname_TSMeteo, 'r')
				
				# Change number to datetime
				time_ETo_aux = num2date(dataETo['time'][:-1],
								units=dataETo['time'].units,
								calendar=dataETo['time'].calendar
								)
				
				# change time nstimedate to datetime
				time_ETo = []
				for iidate in time_ETo_aux:
					if not iidate == None:
						time_ETo.append(datetime(
							iidate.year,
							iidate.month,
							iidate.day,
							iidate.hour,
							iidate.minute)
							)
					else:
						time_ETo.append(None)
				time_ETo = np.array(time_ETo)
				
			else:
				# Read time series of precipitation	csv
				dataETo = pd.read_csv(inputfile.fname_TSMeteo)
				
				# change to txt to datetime
				dataETo["Date"] = pd.to_datetime(dataETo['Date'])				
				
				if inputfile.dt > 60:
					# aggregate data to the model time step
					dataETo.index = pd.DatetimeIndex(dataETo['Date'])
					dataETo = (dataETo.resample(inputfile.Agg_method).sum()).reset_index()
				time_ETo = dataETo["Date"]
				
			# Time periods-----------------------------------------------------------------------
			# Find id of the precipitation array for the the simulation period 
			idate_aux = np.where((time_pre < inputfile.end_date)
								& (time_pre >= inputfile.ini_date))[0]
								
			if inputfile.dt != 60:
				idatepre = np.zeros(len(date_sim_m))
				idate_pre = date_sim_m.isin(time_pre)
			else:
				idatepre = np.zeros(len(date_sim_h))
				idate_pre = date_sim_h.isin(time_pre)
			idatepre[idate_pre == True] = idate_aux
			idatepre[idate_pre == False] = np.nan
			
			# Find id of the potential ET array for the the simulation period 
			idateETo = np.where((time_ETo < inputfile.end_date)
								& (time_ETo >= inputfile.ini_date))[0]
			
			# check if time periods match both time series
			if len(idateETo) < t_end*inputfile.dt_hourly:
				if inputfile.netcf_pre == 1:
					t_end = (time_ETo[-1] - inputfile.ini_date).days
				else:
					t_end = (time_ETo.iloc[-1] - inputfile.ini_date).days
				date_sim_h = pd.date_range(inputfile.ini_date, periods=t_end*24, freq='H')
				date_sim_d = pd.date_range(inputfile.ini_date, periods=t_end, freq='d')	
			
			# time periods for optional variables
			# match the period between dataset to avoid errors
			
		self.date_sim_d = date_sim_d
		self.date_sim_h = date_sim_h
		
		# create time series of dates at model time step frequency
		self.date_sim_dt = pd.Series(pd.date_range(
			inputfile.ini_date,
			periods =t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly,
			freq = inputfile.Agg_method))
		
		self.fpre = fpre
		self.dataETo = dataETo
		self.idateETo = idateETo
		self.idatepre = idatepre
		self.PET = np.zeros(env_state.grid_size)
		#self.PETr = np.zeros(env_state.grid_size)
		self.t_end = t_end		

	# find precipitation and PET for an specific time step
	def run_rainfall_one_step(self, j_tp, j_te, j_tsavi, j_tkc, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		"""
		self.rain_day_before = 0
		self.rain = np.zeros(env_state.grid_size)
		
		if not np.isnan(self.idatepre[j_tp]):
			if inputfile.netcf_pre == 1:
				self.rain = (self.fpre.variables['pre'][self.idatepre[j_tp]][:]).flatten()
			else: # Uniform precipitation over the whole catchement
				self.rain = np.ones(env_state.grid_size)*self.fpre['pre'][self.idatepre[j_tp]]
			self.rain_day_before = 1
			
		if not np.isnan(self.idateETo[j_te]):
			if inputfile.netcf_ETo == 1:
				self.PET = (self.dataETo.variables['pet'][self.idateETo[j_te]][:]).flatten()*inputfile.unit_sim
				#self.PETr += self.dataETo.variables['pet'][self.idateETo[j_te]][:].flatten()*inputfile.unit_sim
			else: # Uniform precipitation over the whole catchement
				self.PET[:] = self.dataETo['pet'][self.idateETo[j_te]]*inputfile.unit_sim
				#Cummulative value of ETp for daily stimation of AET in river cells
				#self.PETr += self.dataETo['ETo'][self.idateETo[j_te]]*inputfile.unit_sim
		
class read_temporal_dataset():
	"""Read all input dataset
	INPUT:
	------
	file_type:	0 for csv and 1 for netCDF
	
	dataset:
	
	"""
	def __init__(self, filename, file_type, dt, end_date, ini_date):#, str_dt):
		""" Setting variables and time steps for reading inputs
		Call this to execute a step in the model.
		INPUT:
		------
		filename: 	file name of dataset
		file_type:	file format 0 for csv files and 1 to netCDF fiels
		end_date:	end date of simulation
		ini_date:	initial date
		dt:			model time step)
		
		OUTPUT:
		-------
		
		"""
		
		freq_dt=str(np.int(dt))+'min'
		
		if os.path.exists(filename):
		
			if file_type == 1: 
				# Read netCDF fiels
				data_set = Dataset(filename, 'r')
				
				# slicing, aggregation, and interpolation.
				
				# Change number to datetime
				time_aux = num2date(data_set['time'][:-1],
								units=data_set['time'].units,
								calendar=data_set['time'].calendar)
				
				self.data_set = data_set
				
				# change time nstimedate to datetime
				time = []
				for iidate in time_aux:
					if not iidate == None:
						time.append(datetime(
							iidate.year,
							iidate.month,
							iidate.day,
							iidate.hour,
							iidate.minute)
							)
					else:
						time.append(None)
				time = np.array(time)
			
			else:
				
				# Read time series of precipitation	csv
				data_set = pd.read_csv(filename)
				
				# change to txt to datetime
				data_set["Date"] = pd.to_datetime(data_set['Date'], format = '%d/%m/%Y %H:%M')
							
				# aggregate data to the model time step
				data_set.index = pd.DatetimeIndex(data_set['Date'])
				
				self.data_set = (data_set.resample(freq_dt).sum()).reset_index()
				
				time = data_set["Date"]
				
				# Find id of the precipitation array for the the simulation period 
				idate_aux = np.where((time < end_date)
									& (time >= ini_date))[0]
				
				self.data_set = self.data_set.iloc[idate_aux]
				#print(self.data_set)
				#import matplotlib.pyplot as plt
				#plt.plot(self.data_set['flux_0'])
				#plt.show()
				if not idate_aux.size:
					print(filename)
					raise Exception("Dataset do not match the simulation period")
		else:
			
			self.data_set = None
			print(filename, 'not provided')
			
			
		#self.time
		#self.da = #return	

		
		# To be completed
		# this will allow to read just one netCDF files for the simulation
		# to read multiples files, take a look at input_datasets_bigfiles
		#inputfile.netcf_savi = 0
		#inputfile.netcf_kc = 0
		#self.fpre = fpre
		#self.dataETo = dataETo
		#self.idateETo = idateETo
		#self.idatepre = idatepre
		#self.PET = np.zeros(env_state.grid_size)
		##self.PETr = np.zeros(env_state.grid_size)
		#self.t_end = t_end		
		
		# find precipitation and PET for an specific time step
	def get_dataset_one_step(self, t, env_state, file_type, field):#, filename):
		"""
		Call this to execute a step in the model.
		INPUT:
		------
		t: 			time step index
		env_state:	grid
		file_type:	file format 0 for csv files and 1 to netCDF fiels
		field:		variable name
		
		OUTPUT:
		-------
		dataset_at_t:	list of values
		
		"""		
		dataset_at_t = np.zeros(env_state.grid_size)
		
		if not np.isnan(self.time[t]):
			
			if file_type == 1:
				dataset_at_t = (self.data_set.variables[field][self.time[t]][:]).flatten()
			
			else: # Uniform precipitation over the whole catchement
				# add interpo;ation here for future versions**
				dataset_at_t = np.ones(env_state.grid_size)*self.data_set[field][self.time[t]]
		
		return dataset_at_t
		
	def get_point_dataset_one_step(self, t):
		"""
		Call this to read forcing data at time step t
		INPUT:
		------
		t: 			time step index
		env_state:	grid
		OUTPUT:
		-------
		dataset_at_t:	list of values
		
		"""
		head = list(self.data_set)
		
		head.remove('Date')
		#print(type(head), t)
		dataset_at_t = np.array(self.data_set[head].iloc[t])
	
		return dataset_at_t

class input_datasets_bigfiles_new(object):
	"""Read input datasets from different sequential files
	Parameters:
	Outputs:
	
	"""
	def __init__(self, inputfile, env_state):
		"""set model grid time series and model files
		INPUT
		-----
		inputfile:	
		OUTPUT
		------
		
		"""
				
		# Define the time step for temporal aggregation
		self.freq_dt=str(np.int(inputfile.dt))+'min'
		
		# Define dataset x and y intervals for spatial interpolation
		#self.lon = env_state.lon
		#self.lat = env_state.lat
		
		# Check if the simulation period is rigth
		if inputfile.first_read == 1:
			t_end = inputfile.Sim_period.days
			self.date_sim_dt = pd.date_range(inputfile.ini_date,
				periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly,
				freq = str(np.int(inputfile.dt))+'min')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
		
		# Save number of time steps
		self.t_end = t_end
		self.read_before_pre = 1
		self.read_before_pet = 1
		
		# to be deleted in new versions
		self.year_pre = int(inputfile.ini_date.year)
		self.year_pet = int(inputfile.ini_date.year)
		
		# rainfall component time step
		self.dt = inputfile.dt
		
		# to be deleted in subsequent versions
		# number of time steps per model time step
		self.nsteps_pre = int(inputfile.dt/inputfile.dt_pre)
		self.nsteps_pet = int(inputfile.dt/inputfile.dt_pet)
		
		# check if temporal interpolation is required for precipitation
		if inputfile.dt_pre != self.dt:
			self.nsteps_day_pre = int(1440/inputfile.dt)
		else:
			self.nsteps_day_pre = int(1440/inputfile.dt_pre)
		
		# check if temporal interpolation is required for evapotrasnpiration
		if inputfile.dt_pet != self.dt:
			self.nsteps_day_pet = int(1440/inputfile.dt)
		else:
			self.nsteps_day_pet = int(1440/inputfile.dt_pet)
		
		# to be deleted in subsequent versions
		# number of time steps per hour model time step
		self.nsteps_hour_pre = int(inputfile.dt_pre/60)
		self.nsteps_hour_pet = int(inputfile.dt_pet/60)
		
		self.fill_value = 1
		self.idatesavi = None
		self.idatekc = None
		
	def run_dataset_one_step(self, j, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		Parameters:
			j:	Counter for time
		Outputs:
			rain:	precipitation for the actual time step
			pet:	potential evapotranspiration for current timestep
		"""
		date = self.date_sim_dt[j]

		# PRECIPITATION
		idate_pre = self.date_sim_dt[j]# - timedelta(hours=(self.nsteps_pre-1))
		
		# create zero array for precipitation
		self.rain = np.zeros(env_state.grid_size)
		
		# find the location of the date in the dataset
		# lacation depending on the hour
		hour_pre = int(int(idate_pre.strftime('%H'))/self.nsteps_hour_pre)
		
		# location depending on the day
		j_tp = int(int(idate_pre.strftime('%j'))-1)*self.nsteps_day_pre + hour_pre
		
		keys = ['lon', 'lat']
		
		# Read data at the begining of the simulation or if a new dataset starts
		if (self.read_before_pre == 1) or (j_tp == 0):
			# Filename of the current year
			fname_pre = inputfile.fname_TSPre + '_' + str(idate_pre.year)# + '.nc'			
			
			# read dataset
			self.fpre = xr.open_mfdataset(fname_pre)
			
			# reproject dataset
			if inputfile.reproject_pre == 1:
				self.fpre = reproject_dataset(self.fpre, keys)
			
			if inputfile.dt_pre != self.dt:
				# temporal resampling
				self.fpre = self.fpre.resample(time=self.freq_dt).sum()
							
			# flag to no read every time the whole dataset
			self.read_before_pre = 0
					
		if inputfile.interpolate_pre == 1:
			# Spatial interpolation
			fpre = self.fpre.isel(time=[j_tp]).interp(lat=(env_state.lat), lon=(env_state.lon))
		else:
			fpre = self.fpre.isel(time=[j_tp])
		#print(np.mean(fpre.mean('time')))		
		self.rain = np.array(fpre.variables['pre'][0][:]).flatten()
		
		# EVAPOTRANSPIRATION
		idate_pet = self.date_sim_dt[j]# - timedelta(hours=(self.nsteps_pet-1))
		self.PET = np.zeros(env_state.grid_size)
		
		# find the location of the date in the dataset
		# lacation depending on the hour
		hour_pet = int(int(idate_pet.strftime('%H'))/self.nsteps_hour_pet)
		j_te = (int(idate_pet.strftime('%j'))-1)*self.nsteps_day_pet + hour_pet				
		
		keys = ['longitude', 'latitude']
		
		# Read data at the begining of the simulation
		if (self.read_before_pet == 1) or (j_te == 0):
			fname_pet = inputfile.fname_TSMeteo + '_' + str(idate_pet.year) + '.nc'
			
			# read dataset
			self.fpet = xr.open_mfdataset(fname_pet)
			
			# fill empty values
			if self.fill_value == 1:
				self.fpet = self.fpet.where(self.fpet < 1e10)
				self.fpet = self.fpet.ffill(keys[0])
				self.fpet = self.fpet.bfill(keys[1])
				self.fpet = self.fpet.ffill(keys[0])
				self.fpet = self.fpet.bfill(keys[1])
						
			if inputfile.dt_pet != self.dt:
				# temporal resampling
				self.fpet = self.fpet.resample(time=self.freq_dt).sum()
			
			# reproject dataset
			if inputfile.reproject_pet == 1:
				fpet = fpet.rio.write_crs(oldPP) # write crs
				self.fpet = fpet.rio.reproject(newPP) #reproject the file
						
			# flag to no read every time the whole dataset
			self.read_before_pet = 0
		
		if inputfile.interpolate_pet == 1:
			# Spatial interpolation
			if inputfile.reproject_pet == 1:
				fpet = self.fpet.isel(time=[j_te]).interp(
					y=env_state.lat,
					x=env_state.lon,
					method="nearest"
					)
			else:
				fpet = self.fpet.isel(time=[j_te]).interp(
					latitude=(env_state.lat),
					longitude=(env_state.lon),
					method="nearest"
					)
		else:
			fpet = self.fpet.isel(time=[j_te])
		
		self.PET = np.array(fpet.variables['pet'][0][:]).flatten()
		self.PET[self.PET < 0] = 0
		
		# SOIL-ADJUSTED VEGETATION INDEX
		# optional dataset
		self.LAI = None
		# Soil-Adjusted Vegetation Index
		if self.idatesavi is not None:
			idate_savi = self.date_sim_dt[j]# - timedelta(hours=(self.nsteps_savi-1))
			self.SAVI = np.zeros(env_state.grid_size)
		
			hour_savi = int(int(idate_savi.strftime('%H'))/self.nsteps_hour_savi)
			j_te = (int(idate_savi.strftime('%j')-1))*self.nsteps_day_savi + hour_savi
			
			# Read data at the begining of the simulation
			if (self.read_before_savi == 1) or (j_te == 0):
				fname_savi = inputfile.fname_TSMeteo + '_' + str(idate_savi.year) + '.nc'
				
				# read dataset
				fsavi = xr.open_mfdataset(fname_savi)
				# temporal resampling
				fsavi = fsavi.resample(time=self.freq_dt).sum()
				# Spatial interpolation
				self.fsavi = fsavi.interp(lat=(self.lat), lon=(self.lon))
				
				# flag to no read every time the whole dataset
				self.read_before_savi = 0
					
			self.SAVI = np.array(self.fsavi.variables['savi'][j_te][:]).flatten()
		
		else:
			self.SAVI = 1
		
		# CROP COEFICIENT FACTOR
		if self.idatekc is not None:
			idate_kc = self.date_sim_dt[j]# - timedelta(hours=(self.nsteps_kc-1))
			self.Kc = np.zeros(env_state.grid_size)
			
			hour_kc = int(int(idate_kc.strftime('%H'))/self.nsteps_hour_kc)
			j_te = (int(idate_kc.strftime('%j'))-1)*self.nsteps_day_kc + hour_kc
			
			# Read data at the begining of the simulation
			if (self.read_before_kc == 1) or (j_te == 0):				
				fname_kc = inputfile.fname_TSMeteo + '_' + str(idate_kc.year) + '.nc'
				
				# read dataset
				fkc = xr.open_mfdataset(fname_kc)
				# temporal resampling
				fkc = fkc.resample(time=self.freq_dt).sum()
				# Spatial interpolation
				self.fkc = fkc.interp(lat=(self.lat), lon=(self.lon))
				
				# flag to no read every time the whole dataset
				self.read_before_kc = 0
						
			self.Kc = np.array(self.fkc.variables['kc'][j_te][:]).flatten()				

		else:	
			self.Kc = 1

class input_datasets_bigfiles(object):
	"""Read input datasets from different sequential files
	Parameters:
	Outputs:
	
	"""
	def __init__(self, inputfile, env_state):
		#dt = np.min([inputfile.dtOF, inputfile.dtUZ, inputfile.dtSZ])
		if inputfile.first_read == 1:
			t_end = inputfile.ndays#Sim_period.days
			self.date_sim_dt = pd.date_range(inputfile.ini_date,
				periods = t_end*inputfile.dt_hourly*inputfile.dt_sub_hourly,
				freq = str(np.int(inputfile.dt))+'min')
			if t_end <= 0:
				sys.exit("End of the simulation period should be later than initial date")
		self.t_end = t_end
		self.read_before_pre = 1
		self.read_before_pet = 1
		self.year_pre = int(inputfile.ini_date.year)
		self.year_pet = int(inputfile.ini_date.year)
		self.dt = inputfile.dt
		self.nsteps_pre = int(inputfile.dt/inputfile.dt_pre)
		self.nsteps_pet = int(inputfile.dt/inputfile.dt_ETo)
		self.nsteps_day_pre = int(1440/inputfile.dt_pre)
		self.nsteps_day_pet = int(1440/inputfile.dt_ETo)
		self.nsteps_hour_pre = int(inputfile.dt_pre/60)
		self.nsteps_hour_pet = int(inputfile.dt_ETo/60)
		self.idatesavi = None
		self.idatekc = None
	
	# find precipitation and PET for an specific time step
	def run_dataset_one_step(self, j, env_state, inputfile):
		"""
		Call this to execute a step in the model.
		Parameters:
			j:	Counter for time
		Outputs:
			rain:	precipitation for the actual time step
			pet:	potential evapotranspiration for current timestep
		"""
		#date = self.date_sim_dt[j]
		# Precipitation
		idate_pre = self.date_sim_dt[j] - timedelta(hours=(self.nsteps_pre-1))
		#print(type(idate_pre), idate_pre)
		self.rain = np.zeros(env_state.grid_size)
		for i in range(self.nsteps_pre):			
			if self.year_pre == idate_pre.year:
				hour_pre = int(int(idate_pre.strftime('%H'))/self.nsteps_hour_pre)
				j_tp = (int(idate_pre.strftime('%j'))-1)*self.nsteps_day_pre + hour_pre
				
				if self.read_before_pre == 1:
					fname_pre = inputfile.fname_TSPre + '_' + str(idate_pre.year) + '.nc'
					self.fpre = Dataset(fname_pre, 'r')
					rain = (self.fpre.variables['pre'][j_tp][:]).flatten()
					self.read_before_pre = 0
				else:
					rain = (self.fpre.variables['pre'][j_tp][:]).flatten()
					self.read_before_pre = 0
			else:
				hour_pre = int(int(idate_pre.strftime('%H'))/self.nsteps_hour_pre) 
				j_tp = (int(idate_pre.strftime('%j'))-1)*self.nsteps_day_pre + hour_pre
				
				fname_pre = inputfile.fname_TSPre + '_' + str(idate_pre.year) + '.nc'
				self.fpre = Dataset(fname_pre, 'r')
				rain = (self.fpre.variables['pre'][j_tp][:]).flatten()
				self.read_before_pre = 0
			self.rain += rain
			self.year_pre = int(idate_pre.year)
			idate_pre += timedelta(hours=1)
		
		# Evapotranspiration
		idate_pet = self.date_sim_dt[j] - timedelta(hours=(self.nsteps_pet-1))
		self.PET = np.zeros(env_state.grid_size)
		for i in range(self.nsteps_pet):			
			if self.year_pet == idate_pet.year:				
				hour_pet = int(int(idate_pet.strftime('%H'))/self.nsteps_hour_pet)
				j_te = (int(idate_pet.strftime('%j'))-1)*self.nsteps_day_pet + hour_pet
				if self.read_before_pet == 1:
					fname_pet = inputfile.fname_TSMeteo + '_' + str(idate_pet.year) + '.nc'
					self.fpet = Dataset(fname_pet, 'r')
					PET = (self.fpet.variables['pet'][j_te][:]).flatten()
					self.read_before_pet = 0
				else:
					PET = (self.fpet.variables['pet'][j_te][:]).flatten()
					self.read_before_pet = 0
			else:
				hour_pet = int(int(idate_pet.strftime('%H'))/self.nsteps_hour_pet)
				j_te = (int(idate_pet.strftime('%j'))-1)*self.nsteps_day_pet + hour_pet 
				fname_pet = inputfile.fname_TSMeteo + '_' + str(idate_pet.year) + '.nc'
				self.fpet = Dataset(fname_pet, 'r')
				PET = (self.fpet.variables['pet'][j_te][:]).flatten()				
				self.read_before_pet = 0
			PET[PET < 0] = 0
			self.PET += PET
			self.year_pet = int(idate_pet.year)
			idate_pet += timedelta(hours=1)
