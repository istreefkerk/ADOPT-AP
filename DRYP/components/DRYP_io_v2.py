import os
import time
import datetime
import numpy as np
import pandas as pd
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii, write_esri_ascii
from datetime import timedelta, datetime
from netCDF4 import Dataset, num2date, date2num
from landlab.components import FlowDirectorSteepest, FlowAccumulator
import rasterio
# Global parameters
ABC_RIVER = 0.99 # River abstraction parameter

class inputfile(object):
	"""
	"""
	def __init__(self, filename_inputs, config, first_read=1):
		"""Model paramter settings and input file namens and location
		"""
		self.first_read = first_read
		# =================================================================
		f = pd.read_csv(filename_inputs)
		self.Mname = f.drylandmodel[1]
				
		#==================================================================		
		# PARAMETER FILE		
		filename_simpar = f.drylandmodel[87]
		fsimpar = pd.read_csv(filename_simpar)
		
		self.ini_date = datetime.strptime(fsimpar.DWAPM_SET[2], '%Y %m %d')
		self.end_date = datetime.strptime(fsimpar.DWAPM_SET[4], '%Y %m %d')
		self.dtOF = int(fsimpar.DWAPM_SET[6])
		self.dtUZ = int(fsimpar.DWAPM_SET[8])
		self.dtSZ = int(fsimpar.DWAPM_SET[10])
		
		netcdf_opt = fsimpar.DWAPM_SET[13].split()
		time_step = fsimpar.DWAPM_SET[15].split()
		reproj_opt = fsimpar.DWAPM_SET[17].split()
		interp_opt = fsimpar.DWAPM_SET[19].split()
				
		# Datasets format
		self.netcf_pre = int(netcdf_opt[0])
		self.netcf_ETo = int(netcdf_opt[1])
		self.netcf_ABC = int(netcdf_opt[2])
		self.netcf_kc =  int(netcdf_opt[3])
		self.netcf_Flux =int(netcdf_opt[4])
		self.netcf_savi =int(netcdf_opt[5])
		self.netcf_savi_min =int(netcdf_opt[6])
		self.netcf_savi_max =int(netcdf_opt[7])
		
		# Dataset time step
		self.dt_pre = int(time_step[0])
		self.dt_ETo = int(time_step[1])
		self.dt_ABC = int(time_step[2])
		self.dt_kc =  int(time_step[3])
		self.dt_Flux =int(time_step[4])
		self.dt_savi =int(time_step[5])
		self.dt_savi_min = int(time_step[6])
		self.dt_savi_max = int(time_step[7])
		
		# Datasets reprojection
		self.reproject_pre = int(reproj_opt[0])
		self.reproject_ETo = int(reproj_opt[1])
		self.reproject_ABC = int(reproj_opt[2])
		self.reproject_kc =  int(reproj_opt[3])
		self.reproject_Flux =int(reproj_opt[4])
		self.reproject_savi =int(reproj_opt[5])
		self.reproject_savi_min = int(reproj_opt[6])
		self.reproject_savi_max = int(reproj_opt[7])
		
		# Datasets interpolate
		self.interpolate_pre = int(interp_opt[0])
		self.interpolate_ETo = int(interp_opt[1])
		self.interpolate_ABC = int(interp_opt[2])
		self.interpolate_kc =  int(interp_opt[3])
		self.interpolate_Flux =int(interp_opt[4])	
		self.interpolate_savi =int(interp_opt[5])
		self.interpolate_savi_min = int(interp_opt[6])
		self.interpolate_savi_max = int(interp_opt[7])
					
		# read separeted files for all inputs
		#self.nfiles = int(fsimpar.DWAPM_SET[31])
						
		self.inf_method = int(fsimpar.DWAPM_SET[22])
		
		if self.inf_method > 3:
			self.inf_method = 0
		
		# read groundwater model activation
		aux_run_GW = fsimpar.DWAPM_SET[24].split()		
		self.run_GW = int(aux_run_GW[0])
		if len(aux_run_GW) > 1:
			self.gw_func = int(aux_run_GW[1])
		else:
			self.gw_func = 0
		
		# save netcdf files of model results
		self.save_results = int(fsimpar.DWAPM_SET[33])
		
		# activate lakes
		self.lakes = int(fsimpar.DWAPM_SET[43])
		
		# temporal agregation of model outputs
		self.dt_results = fsimpar.DWAPM_SET[35]
		# save discharge units
		# 0: volumetric
		# 1: depth
		self.save_dis_depth = int(fsimpar.DWAPM_SET[39])
				
		# Unsaturated zone factors =========================================
		self.kdt_r = float(fsimpar.DWAPM_SET[46])
		self.kDroot = float(config['parameters']['rooting_depth'])	# k for soil depth fsimpar.DWAPM_SET[48]
		self.kAWC = 1.#float(fsimpar.DWAPM_SET[50])	# k for AWC
		self.kKsat = float(config['parameters']['sat_hyd_con'])		# k for soil infiltration fsimpar.DWAPM_SET[52]
		self.k_sigma_ks = float(fsimpar.DWAPM_SET[54]) 
		
		# River routing factors ============================================
		self.kKch = float(config['parameters']['ch_sat_hyd_con'])	# infiltration on channel fsimpar.DWAPM_SET[56]
		self.kTch = float(config['parameters']['recession_time'])	# Runoff decay flow factor fsimpar.DWAPM_SET[58]
		self.kpe = float(fsimpar.DWAPM_SET[60])
		
		# Saturated zone factors ===========================================
		self.kKsat_gw = float(config['parameters']['aq_sat_hyd_con'])	# Ksat factor fsimpar.DWAPM_SET[62]
		self.kSy_gw = float(config['parameters']['specific_yield'])		# Sy factor fsimpar.DWAPM_SET[64]
		if len(fsimpar.DWAPM_SET) == 66:
			self.kFlux = float(fsimpar.DWAPM_SET[66])
		else:
			self.kFlux = 1
			
		if len(fsimpar.DWAPM_SET) == 68:
			self.GW_Cond_factor = float(fsimpar.DWAPM_SET[68])
		else:
			self.GW_Cond_factor = 50
		#self.kTr_ini_par = float(fsimpar.DWAPM_SET[51])
		#self.kpkKch = float(fsimpar.DWAPM_SET[51])
		#self.kpLoss = float(fsimpar.DWAPM_SET[51])
		#self.Ktr = float(fsimpar.DWAPM_SET[51])
		self.dt = np.min([self.dtOF, self.dtUZ, self.dtSZ])
		if self.dt > 60:
			self.dt_sub_hourly = 1
			self.dt_hourly = np.int(1440/self.dt)
			self.unit_sim = self.dt/1440		#change mm/d -> mm/dt
			self.unit_sim_k = self.dt*24/1440	#change mm/h -> mm/dt
			self.kT_units = self.dt/60
		else:
			self.dt_sub_hourly = np.int(60/self.dt)
			self.dt_hourly = 24
			self.unit_sim = self.dt/60			#change mm/d -> mm/dt
			self.unit_sim_k = self.dt/60		#change mm/h -> mm/dt
			self.kT_units = self.dt/60
		self.unit_change_manning = (1/(self.dt*60))**(3/5)
		self.Agg_method = str(self.dt)+'T'
		#self.kpkKch = 1.0							# initial kKch increase for TL
		self.T_str_channel = 0.0					# duration of initial kKch increase for TL
		self.kKch = self.kKch*self.unit_sim_k
		self.river_banks = 100.0 					# Riparian zone with [m]
		self.run_FAc = 1
		self.dt_OF = 1
		self.ndays = (self.end_date - self.ini_date).days
		#print(self.ndays)
		
		#==================================================================
		# READING MODEL PARAMETER FILES ===================================
		# INTERCEPTION COMPONENT
		self.fname_interception = f.drylandmodel[91]
		if os.path.exists(self.fname_interception):
			fcp = pd.read_csv(self.fname_interception)
			#print(fcp.INTERCEPTION)
			# Soil component
			self.fname_av = fcp.INTERCEPTION[1]
			self.fname_savi = fcp.INTERCEPTION[3]
			#self.fname_kc = fcp.INTERCEPTION[3]
			#self.fname_avc = fcp.INTERCEPTION[5]
			self.fname_laia = fcp.INTERCEPTION[5]
			self.fname_laib = fcp.INTERCEPTION[7]
			self.fname_savi_min = fcp.INTERCEPTION[9]
			self.fname_savi_max = fcp.INTERCEPTION[11]
			self.fname_lai = fcp.INTERCEPTION[13]
			
			#Riparian component
			self.fname_avrip = fcp.INTERCEPTION[15]
			self.fname_savi_rip = fcp.INTERCEPTION[15]
			#self.fname_kc = fcp.INTERCEPTION[3]
			#self.fname_avc = fcp.INTERCEPTION[5]
			self.fname_laiarip = fcp.INTERCEPTION[17]
			self.fname_laibrip = fcp.INTERCEPTION[19]
			self.fname_savi_minrip = fcp.INTERCEPTION[21]
			self.fname_savi_maxrip = fcp.INTERCEPTION[23]
			self.fname_lairip = fcp.INTERCEPTION[25]
			
			self.fname_tap_depth = fcp.INTERCEPTION[27]
			self.fname_final_depth = fcp.INTERCEPTION[29]
			
			#new parameter
			self.fname_fcw_canopy = fcp.INTERCEPTION[31]
			self.fname_Sc0_canopy = fcp.INTERCEPTION[33]
			
		else:
			#Soil component
			self.fname_savi = 'None'
			self.fname_av = 'None'
			#self.fname_kc = 'None'
			#self.fname_avc = 'None'
			self.fname_laia = 'None'
			self.fname_laib = 'None'
			self.fname_savi_max = 'None'
			self.fname_savi_min = 'None'
			self.fname_lai = 'None'
			
			# riparian component
			self.fname_savi_rip = 'None'
			self.fname_avrip = 'None'
			#self.fname_kc = 'None'
			#self.fname_avc = 'None'
			self.fname_laiarip = 'None'
			self.fname_laibrip = 'None'
			self.fname_savi_maxrip = 'None'
			self.fname_savi_minrip = 'None'
			self.fname_lairip = 'None'
			
			self.fname_tap_depth = 'None'
			self.fname_final_depth = 'None'
			
			self.fname_fcw_canopy = 'None'
			self.fname_Sc0_canopy = 'None'
				
		# SURFACE COMPONENT ======================================== SZ = 
		self.fname_DEM = f.drylandmodel[4]
		self.fname_Area = f.drylandmodel[6]
		self.fname_FlowDir = f.drylandmodel[8]
		self.fname_Mask = f.drylandmodel[12]
		self.fname_River = f.drylandmodel[14]
		self.fname_RiverWidth = f.drylandmodel[16]
		self.fname_RiverElev = f.drylandmodel[18]		
		
		# UNSATURATED COMPONENT ===================================== UZ = 
		self.fname_n = f.drylandmodel[28]		# porosity (n)
		self.fname_theta_r = f.drylandmodel[30]	# Saturated infiltration rate (a-Ks)
		self.fname_AWC = f.drylandmodel[32]		# Available water content (AWC)
		self.fname_wp = f.drylandmodel[34]		# wilting point (wp)
		self.fname_SoilDepth = f.drylandmodel[36] # root zone (D)
		self.fname_b_SOIL = f.drylandmodel[38]	# Soil parameter alpha (b)
		self.fname_PSI = f.drylandmodel[40]		# Soil parameter alpha (alpha)
		self.fname_Ksat_soil = f.drylandmodel[42] # Saturated infiltration rate (a-Ks)
		self.fname_sigma_ks = f.drylandmodel[44]
		self.fname_theta = f.drylandmodel[46]	# Initial water content [-]
		self.fname_Ksat_ch = f.drylandmodel[48]
		
		# Boundary conditions ======================================== rz =
		if len(f) == 96:
			self.fname_bc = f.drylandmodel[95]
		else:
			self.fname_bc = 'None'
		
		self.fname_TSOF = 'None'
		self.filename_OF_points = 'None'
		
		if os.path.exists(self.fname_bc):
			fbc = pd.read_csv(self.fname_bc)
			self.fname_TSOF = fbc.OFBC[1]
			self.filename_OF_points = fbc.OFBC[3]
				
		# RIPARIAN COMPONENT ======================================== rz =
		if len(f) == 94:
			self.fname_riparian_zone = f.drylandmodel[93]
		else:
			self.fname_riparian_zone = 'None'
			
		if os.path.exists(self.fname_riparian_zone):
			frz = pd.read_csv(self.fname_riparian_zone)
		
			self.fname_rz_n = f.RIPARIAN[1]		# porosity (n)
			self.fname_rz_theta_r = f.RIPARIAN[3]	# residual water content
			self.fname_rz_AWC = f.RIPARIAN[5]		# Available water content (AWC)
			self.fname_rz_wp = f.RIPARIAN[7]		# wilting point (wp)
			self.fname_rz_SoilDepth = f.RIPARIAN[9] # riparian root zone depth (D)
			self.fname_rz_b_SOIL = f.RIPARIAN[11]	# Soil particle distribution (lambda)
			self.fname_rz_PSI = f.RIPARIAN[13]		# Air-entry pressure/suction head (psi)
			self.fname_rz_Ksat_ch = f.RIPARIAN[15]  # Channel Sat. hydraulic conductivity (Ksat)
			self.fname_rz_sigma_ks = f.RIPARIAN[17] # riparian sigma Ksat
			self.fname_rz_theta = f.RIPARIAN[19]	# Initial water content [-]
			
			
		else:	
			self.fname_rz_n = f.drylandmodel[28]		# riparian porosity (n)
			self.fname_rz_theta_r = f.drylandmodel[30]	# riparian Saturated infiltration rate (a-Ks)
			self.fname_rz_AWC = f.drylandmodel[32]		# riparian Available water content (AWC)
			self.fname_rz_wp = f.drylandmodel[34]		# riparian wilting point (wp)
			self.fname_rz_SoilDepth = f.drylandmodel[36]# riparian root zone (D)
			self.fname_rz_b_SOIL = f.drylandmodel[38]	# riparian Soil parameter alpha (b)
			self.fname_rz_PSI = f.drylandmodel[40]		# riparian Soil parameter alpha (alpha)
			self.fname_rz_sigma_ks = f.drylandmodel[44] # riparian sigma Ksat
			self.fname_rz_theta = f.drylandmodel[46]	# riparian Initial water content [-]
			self.fname_rz_Ksat_ch = f.drylandmodel[42]  # riparian Channel Saturated hydraulic conductivity (Ks)
		
		# Groundwater components ==================================== GW = 
		self.fname_GWdomain = f.drylandmodel[51]# GW Boundary conditions
		self.fname_SZ_Ksat = f.drylandmodel[53] # Saturated hydraulic conductivity (Ks)
		self.fname_SZ_Sy = f.drylandmodel[55] 	# Specific yield
		self.fname_GWini = f.drylandmodel[57] 	# Initial water table
		self.fname_FHB = f.drylandmodel[59]		# flux head boundary
		self.fname_CHB = f.drylandmodel[61]		# Constant flux boundary
		self.fname_SZ_bot = f.drylandmodel[63]	# Aquifer bottom elevation
		# additional parameters
		self.fname_a_aq = 'None'
		self.fname_b_aq = 'None'
		self.fname_aquifertype = 'None'
		self.fname_lakes_elevation = 'None'
		
		if self.gw_func	== 1 or self.gw_func == 3:
			if os.path.exists(f.drylandmodel[89]):
				fgw = pd.read_csv(f.drylandmodel[89])
				self.fname_a_aq = fgw.GROUNDWATER[1]
				self.fname_b_aq = fgw.GROUNDWATER[3]
				self.fname_aquifertype = fgw.GROUNDWATER[16] # Constant flux boundary
				self.fname_lakes_elevation = fgw.GROUNDWATER[18] # Constant flux boundary
		
		# groundwater second layer ========================================
		self.fname_SZ_botb = 'None'
		self.fname_SZ_Ksatb = 'None'
		self.fname_SZ_Syb = 'None'
		self.fname_SZ_Ssb = 'None'
		self.fname_GWinib = 'None'
		#self.fname_FHBb = 'None'
		#self.fname_CHBb = 'None'
		self.fname_mask_of = 'None'
		
		
		if self.run_GW > 0:
			if os.path.exists(f.drylandmodel[89]):
				fgw = pd.read_csv(f.drylandmodel[89])
				#print(fgw.GROUNDWATER)
				self.fname_SZ_botb = fgw.GROUNDWATER[6]	# Aquifer bottom elevation
				self.fname_SZ_Ksatb = fgw.GROUNDWATER[8] # Saturated hydraulic conductivity (Ks)
				self.fname_SZ_Syb = fgw.GROUNDWATER[10] 	# Specific yield
				self.fname_SZ_Ssb = fgw.GROUNDWATER[12] 	# Specific yield
				self.fname_GWinib = fgw.GROUNDWATER[14] 	# Initial water table
				#self.fname_FHBb = fgw.GROUNDWATER[59]		# flux head boundary
				#self.fname_CHBb = fgw.GROUNDWATER[61]		# Constant flux boundary
				# only for Manny's model
				self.fname_mask_of = fgw.GROUNDWATER[16]	# Constant flux boundary
				self.fname_lakes_elevation = fgw.GROUNDWATER[18]
		#print(self.fname_lakes_elevation)
		
		#==================================================================
		# Meterological data ==============================================
		self.fname_TSPre = f.drylandmodel[66]	# Precipitation file
		self.fname_TSMeteo = f.drylandmodel[68]	# Evapotranspiration file
		self.fname_TSABC = f.drylandmodel[70]	# Abstraction file: AOF, AUZ, ASZ
		#self.fname_savi = f.drylandmodel[70]
		#self.fname_kc = f.drylandmodel[72]
		# Vegetation parameters ==========================================
		self.fname_TSKc = f.drylandmodel[21]	# Vegetation parameter Kc
		self.fname_Rip_width = f.drylandmodel[23]#Available
		self.fname_Rip_init = f.drylandmodel[25] #Available
		# Output files maps ===================================== Print = 
		self.DirOutput = config['general']['report_folder']#f.drylandmodel[81]		# Output directory
		#reading output points
		self.fname_DISpoints = f.drylandmodel[75]	# Discharge points
		self.fname_SMDpoints = f.drylandmodel[77]	# Soil moisture points
		self.fname_GWpoints = f.drylandmodel[79]	# Groundwater observation points
		
		print("Model Name: ",self.Mname)
		
class model_environment_status(object):
	"""Setting model input varables and environmental states
	"""
	def __init__(self, inputfile):
		"""Create variables to store model states and input data sets.
		Read all input datasst and variables for all components
		"""
		# build the data classes
		# ================ Reading surface water model inputs ==============
		
		print('******************* Reading Input Files ********************')
		
		# Reading digital elevation model
		if os.path.exists(inputfile.fname_DEM):
			(rg, z) = read_esri_ascii(inputfile.fname_DEM,
				name='topographic__elevation')
		else:
			raise Exception("A digital elevation model map must be supplied")
			
		# Reading the raster file of river network
		if os.path.exists(inputfile.fname_River):     
			read_esri_ascii(inputfile.fname_River,
				name='river_length', grid=rg)[1]
			
		else:
			rg.add_ones('node', 'river_length', dtype=float)
			rg.at_node['river_length'][:] *= rg.dx
			print('River network................. not provided as raster')
			print('All cells are considered rivers with length of grid size')
		
		riv = rg.add_ones('node', 'river', dtype=int)
		riv[np.where(rg.at_node['river_length'][:] <= 0)[0]] = 0
		self.riv_nodes = np.where(riv > 0)[0] # River nodes for domain arrays
		
		# Reading the raster file of river width		
		if not os.path.exists(inputfile.fname_RiverWidth):
			riv_width = rg.add_zeros('node', 'river_width', dtype=float)
			rg.at_node['river_width'][:] = 10.0
			rg.at_node['river_width'][rg.at_node['river_width'] > rg.dx] = np.array(rg.dx)
			print('River width................... not provided as raster. Global default applied of W = 10 m')
		else:
			riv_width = read_esri_ascii(inputfile.fname_RiverWidth,
				name = 'river_width', grid = rg)[1]
		
		# Reading the raster file of river elevation		
		if not os.path.exists(inputfile.fname_RiverElev):
			rg.add_zeros('node', 'river_topo_elevation', dtype=float)
			rg.at_node['river_topo_elevation'][:] = np.array(z) #[m]
			print('River bottom.................. not provided as raster')
			print('River bottom elevation: surface elevation')
		else:
			read_esri_ascii(inputfile.fname_RiverElev,
				name='river_topo_elevation',
				grid=rg)[1]
		
		# Reading a raster file of flow direction in LandLab format (receiving node ID)
		if os.path.exists(inputfile.fname_FlowDir):
			fd = read_esri_ascii(inputfile.fname_FlowDir,
				name='flow__receiver_node', grid=rg)[1]
			self.act_update_flow_director = False
		else:
			print('Flow direction................ not provided as raster')
			self.act_update_flow_director = True

		# Reading catchment cell areas
		if os.path.exists(inputfile.fname_Area):
			cth_area = read_esri_ascii(inputfile.fname_Area,
				name='cth_area_k', grid=rg)[1]
			
			area_aux = FlowAccumulator(rg, 'topographic__elevation',
										flow_director='D8',
										runoff_rate='cth_area_k')
					
			area_aux.accumulate_flow(
				update_flow_director=self.act_update_flow_director)	
			self.area_discharge = np.array(
				rg.at_node["surface_water__discharge"])
			rg.at_node["surface_water__discharge"][:] = 0
			self.run_flow_accum_areas = 1
		else:
			self.run_flow_accum_areas = 0
			cth_area = rg.add_ones('node', 'cth_area_k', dtype=float)
			rg.add_zeros('node', "surface_water__discharge", dtype=float)
			#self.cth_area = np.array(cth_area)
			print('Cells factor area............. not provided as raster')
		
		self.cth_area = np.array(cth_area)
		
		# Catchment area: raster file of ceros and ones: ones represent the main cathment
		# The area can be the model domain or any area inside the model domain
		if not os.path.exists(inputfile.fname_Mask):
			mask = read_esri_ascii(inputfile.fname_DEM,
				name='basin', grid=rg)[1]
			print('Basin boundary................ not provided')
		else:
			mask = read_esri_ascii(inputfile.fname_Mask,
				name='basin', grid=rg)[1]
		
		# Reading Soil saturated hydraulic conductivity
		if not os.path.exists(inputfile.fname_Ksat_soil):
			rg.add_ones('node', 'Ksat_uz', dtype=float)
			print('Hydraulic conductivity........ not provided as raster. Global default applied of 1.0 mm/h')
		else:
			read_esri_ascii(inputfile.fname_Ksat_soil,
				name = 'Ksat_uz', grid = rg)[1]
		
		# Change units and applying scale factor kKs
		rg.at_node['Ksat_uz'] = (rg.at_node['Ksat_uz']*
				inputfile.unit_sim_k*inputfile.kKsat)
				
		# Reading soil depth map: raster file [mm]
		if not os.path.exists(inputfile.fname_SoilDepth):
			rg.add_ones('node', 'Soil_depth', dtype=float)
			rg.at_node['Soil_depth'] *= 1000.0	# default value 1000 mm
			print('Rooting depth................. not provided as raster. Global default applied of 1000mm')
		else:
			read_esri_ascii(inputfile.fname_SoilDepth,
				name='Soil_depth', grid=rg)[1]
		# Applying scale factor kDroot
		rg.at_node['Soil_depth'] *= inputfile.kDroot
		self.Droot = np.array(rg.at_node['Soil_depth'])
		
		# Reading residual water content
		if not os.path.exists(inputfile.fname_theta_r): 			
			rg.add_zeros('node', 'theta_r', dtype=float)
			rg.at_node['theta_r'][:] += 0.025
			print('Residual moisture content..... not provided as raster. Global default applied of 0.025')
		else:
			read_esri_ascii(inputfile.fname_theta_r,
				name='theta_r', grid=rg)[1]
		
		# Reading Wilting point field
		if not os.path.exists(inputfile.fname_wp):
			rg.add_ones('node', 'wilting_point', dtype=float)
			rg.at_node['wilting_point'][:] = 0.05
			print('Wilting point................. not provided as raster. Global default applied of 0.05')
		else:
			read_esri_ascii(inputfile.fname_wp,
				name='wilting_point', grid=rg)[1]
				
		# Read Saturated water content (porosity)
		if not os.path.exists(inputfile.fname_n): 
			rg.add_ones('node', 'saturated_water_content', dtype=float) # Under unsaturated conditions [mm]
			rg.at_node['saturated_water_content'][:] = 0.40
			print('Porosity...................... not provided as raster. Global default applied of 0.4')
		else:
			read_esri_ascii(inputfile.fname_n, name='saturated_water_content', grid=rg)[1]
		
		# Reading available water content: raster file		
		if not os.path.exists(inputfile.fname_AWC):
			rg.add_zeros('node', 'AWC', dtype=float)
			rg.at_node['AWC'][:] = 0.10
			print('Available Water Content....... not provided as raster. Global default applied of 0.10')
		else:
			read_esri_ascii(inputfile.fname_AWC, name='AWC', grid=rg)[1]
		# Applying scale factor kAWC
		rg.at_node['AWC'] = rg.at_node['AWC']*inputfile.kAWC
					
		# Exponent for soil moisture - matrix potential relation
		# Rawls (1982), and Clapp and Hornberger (1978)
		if not os.path.exists(inputfile.fname_b_SOIL):			
			rg.add_zeros('node', 'b_SOIL', dtype=float)
			rg.at_node['b_SOIL'][:] = 10.05
			print('Soil particle distribution par. not provided as raster. Global default applied of 10.5')
		else:
			read_esri_ascii(inputfile.fname_b_SOIL,
				name='b_SOIL', grid=rg)[1]
	
		# air-entry/saturated capillary potential, [mm]
		if not os.path.exists(inputfile.fname_PSI):
			rg.add_ones('node', 'PSI', dtype=float)
			rg.at_node['PSI'][:] = 153.0
			print('Suction head.................. not provided as raster. Global default applied of 153 mm')
		else:
			read_esri_ascii(inputfile.fname_PSI, name='PSI', grid=rg)[1]
		
		# Saturated suction for the Campbell model
		rg.at_node['PSI'] = (rg.at_node['PSI']*(rg.at_node['b_SOIL']*2+2.5)
							/ (rg.at_node['b_SOIL']+2.5))
		
		# Read Kc for transforming RET to PET
		if not os.path.exists(inputfile.fname_TSKc): 
			Kc = rg.add_ones('node', 'Kc', dtype=float)
			self.Kc = np.array(Kc)
			print('Kc vegetation................. not provided as raster. Global default applied of 1')
		else:
			Kc = read_esri_ascii(inputfile.fname_TSKc, name='Kc', grid=rg)[1]
			self.Kc = np.array(Kc)
		
		# ======================== Water bastraction component ===============
		rg.add_zeros('node', 'AOF', dtype=float)
		rg.add_ones('node', 'AOFT', dtype=float)
		rg.at_node['AOFT'][:] = ABC_RIVER # additional threshold variable
		
		# ======================== GW - groundwater parameters ===================
		#read model domain of the GW model and merge it with the surface model
		if inputfile.run_GW > 0:
			if not os.path.exists(inputfile.fname_GWdomain):
				if os.path.exists(inputfile.fname_Mask):
					(gw, gwz) = read_esri_ascii(inputfile.fname_Mask,
								name='model_domain')
					print('GW domain ................ not provided, default surface basin')
					
				else:
					(gw, gwz) = read_esri_ascii(inputfile.fname_DEM,
								name='model_domain')
					print('GW domain ................ not provided, default surface basin')
				
			else:
				(gw, gwz) = read_esri_ascii(inputfile.fname_GWdomain, name='model_domain')
		else:
			if os.path.exists(inputfile.fname_Mask):
				(gw, gwz) = read_esri_ascii(inputfile.fname_Mask,
							name='model_domain')
				#print('Basin boundary................ not provided')
					
			else:
				(gw, gwz) = read_esri_ascii(inputfile.fname_DEM,
								name='model_domain')
			#(gw, gwz) = read_esri_ascii(inputfile.fname_Mask, name='model_domain')
		
		# Setting boundary conditions
		rg.status_at_node[rg.status_at_node == rg.BC_NODE_IS_FIXED_VALUE] = rg.BC_NODE_IS_CLOSED
		gw.status_at_node[gw.status_at_node == gw.BC_NODE_IS_FIXED_VALUE] = gw.BC_NODE_IS_CLOSED
		gw.status_at_node[np.where(gwz <= 0)[0]] = gw.BC_NODE_IS_CLOSED # Model domain of the GW
		rg.status_at_node[np.where(gwz <= 0)[0]] = rg.BC_NODE_IS_CLOSED # Model domain of the OF
		
		# ========================== Creating landlab fields =====================
		# Soil parameters fields
		rg.add_zeros('node', 'runoff', dtype=float)
		#rg.add_zeros('node', 'percolation', dtype=float)
		rg.add_zeros('node', 'recharge', dtype=float)
		rg.add_zeros('node', 'PET', dtype=float)
		#rg.add_zeros('node', 'Sorptivity', dtype=float)
	
		# Transmission losses parameters
		rg.add_zeros('node', 'Transmission_losses', dtype=float) # river transmission losses
		#rg.add_zeros('node', 'Base_flow', dtype=float) # river transmission losses
		rg.add_zeros('node', 'decay_flow', dtype=float) # river decay flow parameter
		#rg.add_zeros('node', 'AETp_riv', dtype=float) # Actual ET river
		#rg.add_zeros('node', 'ETp_riv', dtype=float) # PET river
		rg.add_zeros('node', 'Q_ini', dtype=float)
		#rg.add_ones('node', 'kTr_ini', dtype=float)		

		#-------------------------------------------------------------------
		## Read overland flow mask for Manny's model
		#if not os.path.exists(inputfile.fname_mask_of): 
		#	rg.add_ones('node', 'mask_of', dtype=float)
		#	print('mask overland flow...... not provided as raster. Assumed all cells flux')
		#else:
		#	read_esri_ascii(inputfile.fname_mask_of, name = 'mask_of', grid=rg)[1]
		
		# =============================================================================
		# =================== GROUNDWATER COMPONENT: Setting ==========================
		if inputfile.run_GW > 0:
			print("Running Groundwater component")
			# Reading specific yield
			if not os.path.exists(inputfile.fname_SZ_Sy):				
				gw.add_zeros('node', 'SZ_Sy', dtype=float) # Water table elevation
				gw.at_node['SZ_Sy'] += 0.01
				print('Specific yield................ not provided as raster. Global default applied of 0.01')
			else:
				read_esri_ascii(inputfile.fname_SZ_Sy,
					name='SZ_Sy', grid=gw)[1]
			# Applying scale factor kSy
			gw.at_node['SZ_Sy'] *= inputfile.kSy_gw
			
			# Aquifer bottom
			if not os.path.exists(inputfile.fname_SZ_bot): 
				gw.add_zeros('node', 'BOT', dtype = float)
				#gw.at_node['BOT'] = z*0.0#+1450.0 # Defailt values of aquifer bottom 
				print('Aquifer bottom elevation...... not provided as raster. Global default applied of 0.0 m')
			else:
				SZ_bot = read_esri_ascii(inputfile.fname_SZ_bot,
					name='BOT', grid=gw)[1]
			
			# Aquifer Saturated hydraulic conductivity
			if not os.path.exists(inputfile.fname_SZ_Ksat): 
				gw.add_ones('node', 'Hydraulic_Conductivity', dtype=float)
				#gw.at_node['Hydraulic_Conductivity'] += 1.0 # Defailt values of aquifer bottom 
				print('Aquifer Ksat.................. not provided as raster. Global default applied of 1.0 m/h')
			else:
				read_esri_ascii(inputfile.fname_SZ_Ksat, name = 'Hydraulic_Conductivity', grid=gw)[1]
			# Applying scale factor kKsat_gw
			gw.at_node['Hydraulic_Conductivity'] *= inputfile.kKsat_gw
							
			# Check if flux boundary is provided m/h
			if not os.path.exists(inputfile.fname_FHB):
				gw.add_zeros('node', 'SZ_FHB', dtype=float)
				print('Flux boundary conditions. not provided as raster')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_FHB,
					name='SZ_FHB', grid=gw)[1]
				gw.at_node['SZ_FHB'][gw.at_node['SZ_FHB'] == -9999] = 0				
				#gw.at_node['SZ_FHB'][:] = gw.at_node['SZ_FHB'][:]*2/(np.power(rg.dx, 2))
			gw.at_node['SZ_FHB'][gw.status_at_node[gw.status_at_node == gw.BC_NODE_IS_CLOSED]] = 0
			gw.at_node['SZ_FHB'][:] = gw.at_node['SZ_FHB']*inputfile.kFlux
			# Read parameters for calulating  effective thickness
			# a: numerator, and b: denominator
			# Read aquifer paramter a for calculating effective thickness
			if not os.path.exists(inputfile.fname_a_aq):
				gw.add_zeros('node', 'SZ_a_aq', dtype=float)
				gw.at_node['SZ_a_aq'][:] = 50.0
				print('Not available aquifer effective depth... a=50m')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_a_aq,
					name='SZ_a_aq', grid=gw)[1]
			
			gw.at_node['SZ_a_aq'][:] += np.array(rg.at_node['Soil_depth']*0.001)
			
			# Read aquifer parameter b for calculating effective thickness
			if not os.path.exists(inputfile.fname_b_aq):
				gw.add_zeros('node', 'SZ_b_aq', dtype=float)
				print('Not available b parameter aquifer, b=0')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_b_aq,
					name='SZ_b_aq', grid=gw)[1]
						
			# Initial water table depth
			print("reading initial water table elevation...")
			if not os.path.exists(inputfile.fname_GWini): 
				h = gw.add_zeros('node', 'water_table__elevation', dtype=float)
				gw.at_node['water_table__elevation'] = z - rg.at_node['Soil_depth']*0.001
				print('Initial water table elevation. not provided as raster')
				print('Initial water table elevation assumed equal to root depth elevation')
			else:
				h = read_esri_ascii(inputfile.fname_GWini,
					name='water_table__elevation', grid=gw)[1]

			# Check if constant head boundary is provided
			if not os.path.exists(inputfile.fname_CHB):				
				print('Constant head boundary conditions not provided as raster')
			else:
				SZ_CHBa = read_esri_ascii(inputfile.fname_CHB,
					name='SZ_CHB', grid=gw)[1]
				id_CHB = np.where(gw.at_node['SZ_CHB'] != -9999)[0]
				gw.at_node['water_table__elevation'][id_CHB] = gw.at_node['SZ_CHB'][id_CHB]
				gw.status_at_node[id_CHB] = gw.BC_NODE_IS_FIXED_VALUE
			
			# read aquifer type
			# 1: exponential model
			# 2: constant model
			# 3: linear model
			if os.path.exists(inputfile.fname_aquifertype):
				self.gwtype = rasterio.open(inputfile.fname_aquifertype).read(1).flatten()
			else:
				print('No transmissivity model type provided')
				self.gwtype = None
						
			# Calculating intial groundwater river deficit
			rg.add_ones('node', 'riv_sat_deficit', dtype=float)
			root_aux = np.array(z - h)*1000
			root_aux = np.where(root_aux > np.array(rg.at_node['Soil_depth']),
					np.array(rg.at_node['Soil_depth']), root_aux)
			rg.at_node['riv_sat_deficit'] = (z - h)*np.power(rg.dx, 2)
			self.Duz = np.array(root_aux)
			
			#===============================================================================
			# SETTING TWO-LAYER GROUNDWATER MODEL ==========================================
			if inputfile.run_GW > 1:
				print("Running Groundwater with two layers")
				# Reading specific yield, second layer
				#print(inputfile.fname_SZ_Syb)
				if not os.path.exists(inputfile.fname_SZ_Syb):				
					gw.add_zeros('node', 'Sy_2', dtype=float) # Water table elevation
					gw.at_node['Sy_2'] += 0.01
					print('Second Layer, Specific yield...........not provided as raster. Global default applied of 0.01')
				else:
					read_esri_ascii(inputfile.fname_SZ_Syb,
						name='Sy_2', grid=gw)[1]
				
				# Reading specific storativity, second layer
				if not os.path.exists(inputfile.fname_SZ_Ssb):				
					gw.add_zeros('node', 'Ss_2', dtype=float) # Water table elevation
					gw.at_node['Sy_2'] += 0.001
					print('Second Layer, Specific storage........not provided as raster. Global default applied of 0.01')
				else:
					read_esri_ascii(inputfile.fname_SZ_Ssb,
						name='Ss_2', grid=gw)[1]
				# Applying scale factor kSy
				#gw.at_node['SZ_Sy'] *= inputfile.kSy_gw
				
				# Aquifer bottom second layer
				if not os.path.exists(inputfile.fname_SZ_botb): 
					gw.add_zeros('node', 'BOTb', dtype = float)
					gw.at_node['BOTb'][:] = np.array(gw.at_node['BOT'][:]-50.) # Defailt values of aquifer bottom 
					print('Second Layer, Aquifer bottom elevation..not provided as raster. Global default applied of 0.0 m')
				else:
					SZ_bot = read_esri_ascii(inputfile.fname_SZ_botb,
						name='BOTb', grid=gw)[1]
				
				# Aquifer Saturated hydraulic conductivity, second layer
				if not os.path.exists(inputfile.fname_SZ_Ksatb): 
					gw.add_ones('node', 'Ksat_2', dtype=float)
					#gw.at_node['Hydraulic_Conductivity'] += 1.0 # Defailt values of aquifer bottom 
					print('Second Layer, Aquifer Ksat...... not provided as raster. Global default applied of 1.0 m/h')
				else:
					read_esri_ascii(inputfile.fname_SZ_Ksatb, name = 'Ksat_2', grid=gw)[1]
				
				# Read initial hydraulic head, second layer
				if not os.path.exists(inputfile.fname_GWinib): 
					gw.add_zeros('node', 'HEAD_2', dtype=float)
					gw.at_node['HEAD_2'][:] += np.array(gw.at_node['water_table__elevation'][:])
					print('Second Layer, Initial conditions...... not provided as raster. Assumed equal to water table upper layer')
				else:
					read_esri_ascii(inputfile.fname_GWinib, name = 'HEAD_2', grid=gw)[1]
								
		else:
			print("Groundwater component not used")
			
			# set water table below the soil depth [m]
			h = z - rg.at_node['Soil_depth']*0.001
			gw.add_field('water_table__elevation', np.array(h), at="node")
			
			# setting unsaturated zone depth equal to rooting depth [mm]
			self.Duz = np.array(z-h)*1000.0#rg.at_node['Soil_depth'])
			
			# adding the saturated deficit [m3]
			# setting a high saturated deficit to allow free drainage
			rg.add_ones('node', 'riv_sat_deficit', dtype=float)
			rg.at_node['riv_sat_deficit'][:] = 1000*np.power(rg.dx, 2)
			
		self.SZgrid = gw
		
	# =====================================================================				
	# ======================== Defining core cells ========================
		# defining nodes to reduce the number of operation
		act_nodes = rg.core_nodes
		aux_mask = np.zeros_like(z)
		aux_mask[act_nodes] = 1
		mask = np.where(mask > 0, 1, 0)
		self.mask = np.where(mask > 0, aux_mask, mask)
		
		# nodes inside the catchement for domain arrays
		self.basin_nodes = np.where(self.mask > 0)[0]
		
		# Nodes inside the catchment for active node arrays
		self.basin_ids_core = np.where(self.mask[act_nodes] > 0)[0]
		
		# river nodes inside the basin for active node arrays
		self.river_ids_core = np.where(riv[act_nodes] == 1)[0]
		
		# river nodes inside the basin for domain arrays
		self.river_ids_nodes = act_nodes[self.river_ids_core]		
		# Nodes inside the catchment for active node arrays
		#self.river_ids_cath = np.where((riv[act_nodes])[self.basin_ids_core] == 1)[0]
		# river nodes inside the catchment for domain arrays
		#self.river_ids_cath_node = self.basin_nodes[np.where(riv[self.basin_nodes] > 0)[0]]
	
	#======================================================================	
	# === Soil moisture and water content thresholds for unsaturated zone =
		# Soil moisture at Field capacity (m3/m3))
		fc = np.array(rg.at_node['wilting_point']+rg.at_node['AWC']) 
		
		# Saturated water content [mm]
		self.Lsat = np.array(rg.at_node['Soil_depth']
			*rg.at_node['saturated_water_content'])
		
		# Water content at wilting point (mm)
		Lwp = np.array(rg.at_node['wilting_point']
			*rg.at_node['Soil_depth'])
		
		# Water content at field capacity (mm)
		Lfc = np.array(fc*rg.at_node['Soil_depth'])
		
		# Exponent c for Rawls (1982), and Clapp and Hornberger (1978)
		# c_SOIL = np.array(rg.at_node['b_SOIL']2+2.5
		# Campbell (1974)
		c_SOIL = 2/np.array(rg.at_node['b_SOIL']) + 3
		
		# Channel hydraulic parameters
		# Assuming a flow velocity of 1 m/s => 3600 m/h
		rg.at_node['decay_flow'][:] = (inputfile.kT_units*3600.0
									*inputfile.kTch/rg.dx)
		#river_banks = 30.0 # It is hard coded for now and will be pass as a raster grid
		#print(rg.at_node['decay_flow'][:],inputfile.kT_units,
		#							inputfile.kTch,rg.dx)
		# Read saturated hydraulic conductivity channel
		if not os.path.exists(inputfile.fname_Ksat_ch):			
			rg.add_ones('node', 'Ksat_ch', dtype=float)
			print('Channel Ksat.................. not provided')
			print('Assumed equal to soil Ksat')
		else:		
			read_esri_ascii(inputfile.fname_Ksat_ch, name='Ksat_ch', grid=rg)[1]
		
		# Changing channel Ksat_ch units from mm/h to m/dt
		rg.at_node['Ksat_ch'] = (rg.at_node['Ksat_ch']*0.001
								*inputfile.kKch
								*self.mask)		#-> m/h
		
		rg.at_node['SS_loss'] = (rg.at_node['river_width']
								*rg.at_node['Ksat_ch']
								*rg.at_node['river_length']) # m3/dt
		
		# INTERCEPTION COMPONENT ==================================================
		#if 
		#if os.path.exist(inputfile.fname_savi):
		#	self.av = rasterio.open(fname_in).read(1)
		#else:
		#	print('Fraction of vegetation cover..not provided as raster. Global default 1')
		#	self.av = 1.0
		
		#if os.path.exist(inputfile.fname_kc):
		#	self.av = rasterio.open(fname_in).read(1)
		#else:
		#	print('Fraction of vegetation cover..not provided as raster. Global default 1')
		#	self.av = 1.0

		# read crop vegetation factor: default 1
		if os.path.exists(inputfile.fname_interception):
			if os.path.exists(inputfile.fname_av):
				self.av = np.flip(rasterio.open(inputfile.fname_av).read(1), 0).flatten()
			else:
				print('Fraction of vegetation cover..not provided as raster. Global default 1')
				self.av = 0.4
		else:
			self.av = None
		# read Coeficient of exponential function: default 0
		if os.path.exists(inputfile.fname_laia):
			self.lai_a = rasterio.open(inputfile.fname_laia).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.lai_a = 0

		# read Power value for exponential function: default 0
		if os.path.exists(inputfile.fname_laib):
			self.lai_b = rasterio.open(inputfile.fname_laib).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.lai_b = 0

		# read Min Soil-Adjusted Vegetation Index: default 0
		if os.path.exists(inputfile.fname_savi_min):
			self.savi_min = rasterio.open(inputfile.fname_savi_min).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.savi_min = 0

		# Max Soil-Adjusted Vegetation Index: defgault 1
		if os.path.exists(inputfile.fname_savi_max):
			self.savi_max = rasterio.open(inputfile.fname_savi_max).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.savi_max = 1.0
		
		#------Modification for Dyna-Veg---------------------------------------------------
		# bioma-dependent coefficient
		if os.path.exists(inputfile.fname_fcw_canopy):
			self.fcw_cn = rasterio.open(inputfile.fname_fcw_canopy).read(1).flatten()
		else:
			print('Biome-dependent coefficient, not provided. Global 1 []')
			self.fcw_cn = np.ones(len(rg.at_node['Soil_depth']))
		
		# inital water content of the canopy storage
		if os.path.exists(inputfile.fname_Sc0_canopy):
			self.Sc0_cn = rasterio.open(inputfile.fname_Sc0_canopy).read(1).flatten()
		else:
			print('Initial canopy storage, not provided. Global 0 []')
			self.Sc0_cn = np.zeros(len(rg.at_node['Soil_depth']))
		
		# inital water content of the canopy storage, riparian zone
		if os.path.exists(inputfile.fname_Sc0_canopy):
			self.Sc0_cnrp = rasterio.open(inputfile.fname_Sc0_canopy).read(1).flatten()
		else:
			print('Initial riparian canopy storage, not provided. Global 0 []')
			self.Sc0_cnrp = np.zeros(len(rg.at_node['Soil_depth']))
		
		# Tap water threshold for evaporation uptake
		# tap needs to be equal or higher than the soil depth
		if os.path.exists(inputfile.fname_tap_depth):
			self.tap_depth = rasterio.open(inputfile.fname_tap_depth).read(1).flatten()
		else:
			print('Tap water level, not provided. Global 0 [mm]')
			self.tap_depth = np.zeros(len(rg.at_node['Soil_depth']))
		
		self.ztap = z - self.tap_depth*0.001
		
		# Final plant water uptake threshold for evaporation uptake
		if os.path.exists(inputfile.fname_final_depth):
			self.final_depth = rasterio.open(inputfile.fname_final_depth).read(1).flatten()
		else:
			print('Final root water uptake level, not provided. Default is rooting depth [mm]')
			self.final_depth = np.array(rg.at_node['Soil_depth'])
		
		# change units: mm -> m
		self.zfinal = z - self.final_depth*0.001
		#print(self.final_depth)
		# 
		self.final_depth = -(self.tap_depth - self.final_depth)*0.001
		#print(self.ztap)
		self.final_depth[self.final_depth <= 0] = self.Droot[self.final_depth <= 0]*0.001
		#print(self.tap_depth)
		#print(self.final_depth)
		# -------- modification for lakes ------------
		# read maximum surface water elevation of lakes
		if os.path.exists(inputfile.fname_lakes_elevation):
			self.z_lakes = rasterio.open(inputfile.fname_lakes_elevation).read(1).flatten()
		else:
			print('Maximum water elevation lakes..not provided as raster. Global default z')
			self.z_lakes = np.array(rg.at_node['topographic__elevation'])
		
		#---------------------------------------------------------------------------------
		# Interception component for the riparian area
		# read crop vegetation factor: default 1
		if os.path.exists(inputfile.fname_interception):
			if os.path.exists(inputfile.fname_avrip):
				self.avrip = np.flip(rasterio.open(inputfile.fname_avrip).read(1), 0).flatten()
			else:
				print('Fraction of vegetation cover..not provided as raster. Global default 1')
				self.avrip = 0.4
		else:
			self.avrip = None
		# read Coeficient of exponential function: default 0
		if os.path.exists(inputfile.fname_laiarip):
			self.lai_arip = rasterio.open(inputfile.fname_laiarip).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.lai_arip = 0

		# read Power value for exponential function: default 0
		if os.path.exists(inputfile.fname_laibrip):
			self.lai_brip = rasterio.open(inputfile.fname_laibrip).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.lai_brip = 0

		# read Min Soil-Adjusted Vegetation Index: default 0
		if os.path.exists(inputfile.fname_savi_minrip):
			self.savi_minrip = rasterio.open(inputfile.fname_savi_minrip).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.savi_minrip = 0

		# Max Soil-Adjusted Vegetation Index: defgault 1
		if os.path.exists(inputfile.fname_savi_maxrip):
			self.savi_maxrip = rasterio.open(inputfile.fname_savi_maxrip).read(1).flatten()
		else:
			print('Fraction of vegetation cover..not provided as raster. Global default 1')
			self.savi_maxrip = 1.0
			
		
		# INITIAL CONDITIONS ------------------------------------------------------------
		# Initial water content as volumetric fraction
		print("Reading initial conditons")
		#print("Soil moisture...")
		if not os.path.exists(inputfile.fname_theta):			
			rg.add_zeros('node','Soil_Moisture', dtype=float)			
			rg.at_node['Soil_Moisture'][:] = np.array(rg.at_node['wilting_point'])*1.01			
			print('Initial soil moisture...... not provided as raster. Global default applied of 1.01*wp')			
		else:		
			read_esri_ascii(inputfile.fname_theta, name='Soil_Moisture', grid=rg)[1]
		
		#print("Riparian soil moisture...")
		if not os.path.exists(inputfile.fname_theta):			
			rg.add_zeros('node','pSoil_Moisture', dtype=float)			
			rg.at_node['pSoil_Moisture'][:] = np.array(rg.at_node['wilting_point'])			
			print('Initial riparian moisture... not provided as raster. Global default applied of 1.01*wp')			
		else:		
			read_esri_ascii(inputfile.fname_theta, name='pSoil_Moisture', grid=rg)[1]
		
		
		# Water content soil[mm]
		self.tht = np.array(rg.at_node['Soil_Moisture'][:])	
		self.L_0 = np.array(rg.at_node['Soil_Moisture'][:])*self.Duz		
		self.t_0 = np.zeros_like(self.L_0)		
		self.Ft_0 = np.zeros_like(self.L_0)		
		self.SORP0 = np.zeros_like(self.L_0)
		
		# Water content riparian area []
		self.ptht = np.array(rg.at_node['pSoil_Moisture'][:])	
		self.ptht = np.where(h >= rg.at_node['river_topo_elevation' ],
				rg.at_node['wilting_point']+rg.at_node['AWC'],
				self.ptht)
		
		ds = np.array(rg.at_node['Soil_depth'][act_nodes])
		
		# threshold capillary rise
		self.gwet_lim = ((rg.at_node['wilting_point']+0.5*rg.at_node['AWC'])/
				rg.at_node['saturated_water_content'])
		
		# Initial soil moisture deficit
		SMD_0 = np.array(Lfc-self.L_0)		
		self.SMD_0 = np.where(SMD_0 < 0.0, 0.0, SMD_0)		
		self.SMDh = SMD_0		
		#SMD_riv_0 = np.array(Lfc[self.river_ids_nodes]-self.L_0[self.river_ids_nodes])		
		#self.SMD_riv_0 = np.where(SMD_riv_0 < 0.0, 0.0, SMD_riv_0)
						
		self.L0_riv = np.array(self.L_0[self.river_ids_nodes])
		self.grid = rg
		self.act_nodes = act_nodes
		self.grid_size = len(z)
		self.c_SOIL = c_SOIL
		self.fc = fc
		
		# calculating cells area
		self.area_cells = rg.dx*rg.dy*rg.at_node['cth_area_k']
		
		#self.area_cells_hills = rg.dx*rg.dy*rg.at_node['cth_area_k']
		
		self.area_cells_banks = np.zeros_like(z)
		
		self.area_catch_factor = (rg.at_node['cth_area_k']
			/ np.sum(rg.at_node['cth_area_k'][self.basin_nodes]))
		
		self.area_river_factor = np.zeros_like(z)
		self.area_river_factor[self.river_ids_nodes] = 1 / np.sum(rg.at_node['cth_area_k'][self.basin_nodes])
		self.area_cth = 1/np.sum(rg.at_node['cth_area_k'][self.basin_nodes])
		
		# Calculate area of river banks, riparian zone
		if rg.dx > inputfile.river_banks:		
			# if river banks are smaller than grid size
			self.area_cells_banks[self.riv_nodes] = np.array(
				rg.at_node['river_length'][self.riv_nodes]
				* (riv_width[self.riv_nodes]+2*inputfile.river_banks))
				
			self.area_cells_banks = np.where(self.area_cells_banks > np.power(rg.dx, 2),
					np.power(rg.dx, 2), self.area_cells_banks)
			
		else:
			# if river banks are bigger than grid size
			# the riparian area is equal to the grid size
			self.area_cells_banks[self.riv_nodes] = (rg.dx
				*rg.dy*rg.at_node['cth_area_k'][self.riv_nodes]
				)
		
		# inactive cells, cells outside the catchment
		self.mask_grid = np.ones(len(aux_mask), dtype=int) - aux_mask
		
		# riparian area factor [-]: area_rip/area_cell
		# to pass from rip_cell to model_cell
		self.riv_factor = aux_mask*self.area_cells_banks/self.area_cells
		self.inv_riv_factor = np.zeros_like(self.riv_factor)
		self.inv_riv_factor[self.area_cells_banks > 0] =(
			self.area_cells[self.area_cells_banks > 0]/
			self.area_cells_banks[self.area_cells_banks > 0])
		#print(self.riv_factor)
		# hillslope area cell factor [-]
		#self.hill_factor = aux_mask*self.area_cells_hills/self.area_cells
		
		# proportion of riparian area in each cell [-]
		# to pass from [m3] -> [mm]
		self.rip_factor = np.zeros_like(z)
		self.rip_factor[self.area_cells_banks != 0] = (
			1000./self.area_cells_banks[self.area_cells_banks != 0])
				
		# riparian area [m2]
		self.rarea = np.array(rg.at_node['river_width']*rg.at_node['river_length'])
		
		# Groundwater selection function parameter
		self.func = int(inputfile.gw_func)
		
		# Define coordinates of grided data
		lat = rg.node_y.reshape(rg.shape)
		lon = rg.node_x.reshape(rg.shape)
		self.lon = rg.node_x[:rg.shape[1]]
		self.lat = np.linspace(np.min(rg.node_y), np.max(rg.node_y),
					num=rg.shape[0])
		
		
	# Create directories for saving results	
	def set_output_dir(self, inputfile):
		"""
		Directory *DirOutput* Created
		Directory *outputcirnetcdf* already exists
		"""
		#print('************************************************************')
		print('********************* Output Directory **********************')
		if not os.path.exists(inputfile.DirOutput):
			os.mkdir(inputfile.DirOutput)
			print("Directory ", inputfile.DirOutput, " Created ")
		else:
			print("Directory ", inputfile.DirOutput, " already exists")
		
		output_dir_nc = inputfile.DirOutput+'/TimeFrames'
		
		if not os.path.exists(output_dir_nc):
			os.mkdir(output_dir_nc)
			print("Directory ", output_dir_nc, " Created ")
		else:
			print("Directory ", output_dir_nc, " already exists")
		
		# Output filenames
		self.fnameTS_avg = inputfile.DirOutput+'/' + inputfile.Mname + '_avg'
		self.fnameTS_OF  = inputfile.DirOutput+'/' + inputfile.Mname + '_OF_'
		self.fnameTS_UZ  = inputfile.DirOutput+'/' + inputfile.Mname + '_UZ_'
		self.fnameTS_GW  = inputfile.DirOutput+'/' + inputfile.Mname + '_GW_'
		self.fnameTS_RZ  = inputfile.DirOutput+'/' + inputfile.Mname + '_RZ_'
		
	# Find coordinates of points in model components
	def points_output(self, inputfile):
		""" This function reads data points to report model results
		Values at each point will be extracted for all components depending
		on the specified points:
		OF:	surfave component
		UZ:	soil and riparian component
		GW: saturated component
		INPUT
		-----
		inputfile:	python obkject containing a list of points

		OUTPUT
		------
		list of nodes where values will be extracted
		"""				
		# Reading output points
		#fOF = pd.read_csv(inputfile.fname_DISpoints)
		#fUZ = pd.read_csv(inputfile.fname_SMDpoints)
		#fGW = pd.read_csv(inputfile.fname_GWpoints)
		
		gaugeidOF = extract_id_from_coords(self.grid,
			inputfile.fname_DISpoints
			)

		gaugeidUZ = extract_id_from_coords(self.grid,
			inputfile.fname_SMDpoints
			)
			
		gaugeidGW = extract_id_from_coords(self.grid,
			inputfile.fname_GWpoints
			)
		
		
		
		# creating variables for storing outputs
		###npointsOF = len(fOF['North'])
		###npointsUZ = len(fUZ['North'])
		###npointsGW = len(fGW['North'])
		###
		#### list of empty arrays for point list
		###gaugeidOF = []
		###gaugeidUZ = []
		###gaugeidGW = []
		gaugeidRZ = []
		###
		####OF_label = []
		####UZ_label = []
		####GW_label = []
		###
		####	Overland component points and variables
		###for ndis in range(npointsOF):
		###	gaugeidOF.append(self.grid.find_nearest_node([fOF['East'][ndis], fOF['North'][ndis]]))
		###	#OF_label.append('OF_'+str(ndis))
		###
		####	Unsaturated component points and variables
		###for nUZ in range(npointsUZ):
		###	gaugeidUZ.append(self.grid.find_nearest_node([fUZ['East'][nUZ], fUZ['North'][nUZ],]))
		###	#UZ_label.append('UZ_'+str(nUZ))
		###
		####	Saturated components points and variables
		###for nGW in range(npointsGW):
		###	gaugeidGW.append(self.grid.find_nearest_node([fGW['East'][nGW], fGW['North'][nGW]]))
		###	#GW_label.append('SZ_'+str(nGW))
		
		# find nodes in the riparian area
		riv_id = self.grid.at_node['river'][gaugeidUZ]
		aux_gaugeidRZ = list(np.where(riv_id == 1)[0])
		
		if len(aux_gaugeidRZ):
			for iRZ in aux_gaugeidRZ:
				gaugeidRZ.append(gaugeidUZ[iRZ])
		
		
		# Finding core cells, river cells and basin cells
		#act_nodes = self.grid.core_nodes # Nodes inside the model domain
		#
		#if inputfile.first_read == 1:
		#
		#	#create a range of cell for cosmos probe			
		#	cosmos_ids = []			
		#	cosmos_ids_core = []
		#	
		#	if ncell_a == 0:				
		#		cosmos_ids_core = np.where(self.grid.core_nodes == gaugeidUZ[0])[0]				
		#		cosmos_ids_core = cosmos_ids_core.astype(int)				
		#		cosmos_ids = gaugeidUZ[0].astype(int)			
		#	else:				
		#		for cos_id in range(gaugeidUZ[0]-ncell_a*self.grid.shape[1], gaugeidUZ[0]+ncell_a*self.grid.shape[1],self.grid.shape[1]):				
		#			for x in range(cos_id-ncell_a, cos_id+ncell_a):					
		#				cosmos_ids.append(x)
		#			
		#		self.cosmos_ids_core_mask = np.isin(act_nodes, cosmos_ids)							
		#		self.cosmos_ids = act_nodes[self.cosmos_ids_core_mask]
		#		cosmos_ids_core_aux = np.array(range(len(act_nodes)))				
		#		self.cosmos_ids_core_mask[self.cosmos_ids_core_mask == True] = 1				
		#		self.cosmos_ids_core = np.where(self.cosmos_ids_core_mask == 1)[0]
		#	
		#gaugeidUZ_act = []
		#
		#for dUZgi in range(len(gaugeidUZ)):		
		#	gaugeidUZ_act.append(np.where(self.grid.core_nodes == gaugeidUZ[dUZgi])[0])
		
		self.gaugeidOF = gaugeidOF
		self.gaugeidUZ = gaugeidUZ
		self.gaugeidGW = gaugeidGW
		self.gaugeidRZ = gaugeidRZ
		#print(self.gaugeidOF)
		#print(self.gaugeidUZ)
		#print(self.gaugeidGW)
		#print(self.gaugeidRZ)
		
	
	def save_data_to_file(inputfile, model_environment_status):
		"""
		Do any saving here
		"""
		pass


# COMPONENT TO BE DEVELOPED: DO NOT MODIFY THE CODE BELOW

class soil_parameters(object):
	"""Setting model input varables and environmental states
	"""
	def __init__(self, grid_size, inputfile):
		"""Read soil layer parameters
		INPUT:
		------
		grid_size:	size of the model domain
		inputfile:	list of file manes for soil parameters
		
		OUTPUT:
		-------
		"""
		print("Reading soil parameter for soil layer")		
		# Reading Soil saturated hydraulic conductivity
		if not os.path.exists(inputfile.fname_Ksat_soil):
			self.Ksat_uz = np.ones(grid_size)
			print('Hydraulic conductivity........ not provided as raster. Global default applied of 1.0 mm/h')
		else:
			self.Ksat_uz = rasterio.open(inputfile.fname_Ksat_soil).read(1).flatten()
					
		# Change units and applying scale factor kKs
		self.Ksat_uz = (self.Ksat_uz*inputfile.unit_sim_k*inputfile.kKsat)
				
		# Reading residual water content
		if not os.path.exists(inputfile.fname_theta_r):
			self.theta_res = np.zeros(grid_size)
			self.theta_res[:] += 0.025
			print('Residual moisture content..... not provided as raster. Global default applied of 0.025')
		else:
			self.theta_res = rasterio.open(inputfile.fname_theta_r).read(1).flatten()
			
		# Reading Wilting point field
		if not os.path.exists(inputfile.fname_wp):
			self.theta_wp = np.ones(grid_size)
			self.theta_wp[:] = 0.05
			print('Wilting point................. not provided as raster. Global default applied of 0.05')
		else:
			self.theta_wp = rasterio.open(inputfile.fname_wp).read(1).flatten()
				
		# Read Saturated water content (porosity)
		if not os.path.exists(inputfile.fname_n):
			self.theta_sat = np.ones(grid_size)
			self.theta_sat[:] = 0.40
			print('Porosity...................... not provided as raster. Global default applied of 0.4')
		else:
			self.theta_sat = rasterio.open(inputfile.fname_n).read(1).flatten()
			
		# Reading available water content: raster file		
		if not os.path.exists(inputfile.fname_AWC):
			self.theta_AWC = np.ones(grid_size)
			self.theta_AWC[:] = 0.10
			print('Available Water Content....... not provided as raster. Global default applied of 0.10')
		else:
			self.theta_AWC = rasterio.open(inputfile.fname_AWC).read(1).flatten()
							
		# Exponent for soil moisture - matrix potential relation
		# Rawls (1982), and Clapp and Hornberger (1978)
		if not os.path.exists(inputfile.fname_b_SOIL):
			self.lambdas = np.ones(grid_size, dtype=float)
			self.lambdas[:] = 10.05
			print('Soil particle distribution par. not provided as raster. Global default applied of 10.5')
		else:
			self.lambdas = rasterio.open(inputfile.fname_b_SOIL).read(1).flatten()
			
		# air-entry/saturated capillary potential, [mm]
		if not os.path.exists(inputfile.fname_PSI):
			self.psi_a = np.ones(grid_size, dtype=float)
			self.psi_a[:] = 153.0
			print('Suction head.................. not provided as raster. Global default applied of 153 mm')
		else:
			self.psi_a = rasterio.open(inputfile.fname_PSI).read(1).flatten()
			
		# Sorptivity for the Campbell model
		self.psi = self.psi_a*(self.lambdas*2+2.5)/(self.lambdas+2.5)
		
		# Exponent c for Rawls (1982), and Clapp and Hornberger (1978)
		# c_SOIL = np.array(self.lambdas)*2+2.5
		# Campbell (1974)
		self.c_SOIL = 2/np.array(self.lambdas) + 3
		
		# Reading soil depth map: raster file [mm]
		if not os.path.exists(inputfile.fname_SoilDepth):
			self.depth_uz = np.ones(grid_size)
			self.depth_uz *= 1000.0	# default value 1000 mm
			print('Rooting depth................. not provided as raster. Global default applied of 1000mm')
		else:
			self.depth_uz = rasterio.open(inputfile.fname_SoilDepth).read(1).flatten()
			
		# Applying scale factor kDroot
		self.depth_uz *= inputfile.kDroot
		self.Droot = np.array(self.depth_uz)

def extract_id_from_coords(grid, filename):
	""" extract nodes from a csv file
	this component uses the landlab funtion "find_nearest_node
	INPUT:
	------
	grid: landlabgrid
	filename:	csv file with coordinates
	
	OUTPUT:
	-------
	numby array with id nodes
	"""
	
	# check if file is available
	if not os.path.exists(filename):
		print(filename)
		raise Exception("File do not exis")
	
	# read coordinates from csv file
	datapoints = pd.read_csv(filename)
	
	# creating variables for storing outputs
	npoints = len(datapoints['North'])
	
	idpoint = []
	
	for ndis in range(npoints):
		idpoint.append(grid.find_nearest_node(
			[datapoints['East'][ndis],
			datapoints['North'][ndis]])
			)
			
	return idpoint
	
def extract_id_from_raster(grid, filename):
	""" find id location of boundary conditions from raster
	this component uses the landlab funtion "find_nearest_node
	INPUT:
	------
	grid: 		landlabgrid
	filename:	filename raster
	
	OUTPUT:
	-------
	numby array with id nodes
	"""
	# read Power value for exponential function: default 0
	if os.path.exists(filename):
		location_map = rasterio.open(inputfile.fname_laibrip).read(1).flatten()
	else:
		print(filename)
		raise Exception("File do not exis")
	
	id_location = np.where(location_map > 0)[0]
	
	aux = [id_location, location_map[id_location]]
	
	#sort data
	aux_sort = aux.sort(axis=0)
	
	return id_location[0]