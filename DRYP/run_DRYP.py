# -*- coding: utf-8 -*-
"""
DRYP: Dryland Water Partitioning Model
"""
import numpy as np
import pandas as pd
from DRYP.components.DRYP_io_v2 import (inputfile,
	model_environment_status, soil_parameters,
	extract_id_from_coords)
from DRYP.components.DRYP_infiltration import infiltration
from DRYP.components.DRYP_interception import interception
from DRYP.components.DRYP_read_meteo import (
	rainfall, input_datasets_bigfiles)
from DRYP.components.DRYP_read_dataset import (
	read_temporal_dataset, read_dataset)
from DRYP.components.DRYP_soil_layer import swbm
from DRYP.components.DRYP_ABM_connector import ABMconnector
#from components.DRYP_routing import runoff_routing
from DRYP.components.DRYP_flow_accum import runoff_routing
#from components.DRYP_flow_accumf90 import runoff_routing
from DRYP.components.DRYP_groundwater_EFD import (
	gwflow_EFD,	storage, storage_uz_sz,
	recharge_routing)
from DRYP.components.DRYP_Gen_Func_v2 import (
	GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar,
	save_map_to_rastergrid, check_mass_balance)
import matplotlib.pyplot as plt
import time


# Structure and model components ---------------------------------------
# data_in:	Input variables 
# env_state:Model state and fluxes
# rf:		Precipitation
# cnp:		canopy interception
# abc:		Anthropic boundary conditions
# inf:		Infiltration 
# swbm:		Soil water balance
# ro:		Routing - Flow accumulator
# gw:		Groundwater flow

#@profile

class DRYP_Model(object):
		
	def __init__(self, filename_input): #run_dryp
		
		# read model paramters and model setting file
		self.data_in = inputfile(filename_input, self.config)
		
		# setting model fluxes and state variables
		self.env_state = model_environment_status(self.data_in)
		
		# read soil paramters
		self.soil = soil_parameters(self.env_state.grid_size, self.data_in)
		
		# add variable saturated component
		self.Qusz = recharge_routing(self.env_state.grid_size)
		
		# read groundwater paramters
		
		# setting location and model results
		self.env_state.set_output_dir(self.data_in)
		self.env_state.points_output(self.data_in)
		
		# setting model components
		if self.data_in.netcf_pre == 2:
			self.rf = input_datasets_bigfiles(self.data_in, self.env_state)
		else:
			self.rf = rainfall(self.data_in, self.env_state)
		
		## Read precipitation
		#PRE = read_dataset(data_in.dt, data_in.dt_pre,
		#	data_in.ini_date, data_in.end_date,
		#	data_in.netcf_pre,
		#	data_in.reproject_pre,
		#	data_in.interpolate_pre,
		#	env_state.grid_size)
		#
		## Read reference potential evpotranpiration
		#ET0 = read_dataset(data_in.dt, data_in.dt_ETo,
		#	data_in.ini_date, data_in.end_date,
		#	data_in.netcf_ETo,
		#	data_in.reproject_ETo,
		#	data_in.interpolate_ETo,
		#	env_state.grid_size)
		
		# Read SAVI
		self.SAVI = read_dataset(self.data_in.dt, self.data_in.dt_savi,
			self.data_in.ini_date, self.data_in.end_date,
			self.data_in.netcf_savi,
			self.data_in.reproject_savi,
			self.data_in.interpolate_savi,
			self.env_state.grid_size)
			
		# Read SAVI minimum value
		self.SAVImin = read_dataset(self.data_in.dt, self.data_in.dt_savi_min,
			self.data_in.ini_date, self.data_in.end_date,
			self.data_in.netcf_savi_min,
			self.data_in.reproject_savi_min,
			self.data_in.interpolate_savi_min,
			self.env_state.grid_size)
		
		# Read SAVI maximum value
		self.SAVImax = read_dataset(self.data_in.dt, self.data_in.dt_savi_max,
			self.data_in.ini_date, self.data_in.end_date,
			self.data_in.netcf_savi_max,
			self.data_in.reproject_savi_max,
			self.data_in.interpolate_savi_max,
			self.env_state.grid_size)
		
		
		# read overland flow boundary condition
		self.dataFlux = read_temporal_dataset(
				self.data_in.fname_TSOF,
				self.data_in.netcf_Flux,
				self.data_in.dt,
				self.data_in.end_date,
				self.data_in.ini_date,
				)
					
		self.abc = ABMconnector(self.data_in, self.env_state, self.agents.farmers)
		self.inf = infiltration(self.env_state, self.data_in)
		self.cnp = interception(self.env_state, self.data_in)
		
		self.swb = swbm(self.env_state, self.env_state.Duz, self.env_state.tht, self.data_in)
		self.swb_rip = swbm(self.env_state, self.env_state.Droot, self.env_state.ptht, self.data_in)
		self.ro = runoff_routing(self.env_state, self.data_in)
		self.gw = gwflow_EFD(self.env_state, self.data_in)
		
		# Output variables and location
		self.outavg = GlobalTimeVarAvg(self.env_state.area_catch_factor)
		self.outavg_rip = GlobalTimeVarAvg(self.env_state.area_river_factor)
		self.outpts = GlobalTimeVarPts()
		self.outptsRZ = GlobalTimeVarPts()
		self.state_var = GlobalGridVar(self.env_state, self.data_in)
		
		# read location of point boundary conditions
		if self.dataFlux.data_set is not None:
			if self.data_in.netcf_ABC == 0:
				self.idFluxOF = extract_id_from_coords(
					self.env_state.grid,
					self.data_in.filename_OF_points
					)
			
			elif self.data_in.netcf_ABC == 2:
				idFluxOF = extract_id_from_raster(
					self.env_state.grid,
					self.data_in.filename_OF_points
					)
		
		self.t = 0	
		self.t_eto = 0	
		self.t_pre = 0
		self.t_savi = 0
		self.t_kc = 0
		self.t_abs = 0
		
		self.pre_mb = []
		self.exs_mb = []
		self.tls_mb = []
		self.gws_mb = []
		self.uzs_mb = []
		self.gbf_mb = []
		self.rch_mb = []
		self.aet_mb = []
		self.egw_mb = []
		self.chs_mb = []
		self.rzs_mb = []
		
		self.bcsz_mb = []
		self.pth_mb = []
		self.eca_mb = []
		self.lai_mb = []
		self.kc_mb = []
		self.pthr_mb = []
		self.kcrip_mb = []
		self.ecar_mb = []
		
		self.rch_agg = np.zeros(len(self.swb.L_0))
		self.etg_agg = np.zeros(len(self.swb.L_0))
		self.dt_GW = np.int(self.data_in.dt)

	def step(self):
		
		#while self.t < self.data_in.ndays:
		
		for UZ_ti in range(self.data_in.dt_hourly):
			
			for dt_pre_sub in range(self.data_in.dt_sub_hourly):
				
				if self.data_in.netcf_pre == 2:
					self.rf.run_dataset_one_step(self.t_pre,
							self.env_state, self.data_in)
				else:
					self.rf.run_rainfall_one_step(self.t_pre, self.t_eto, self.t_savi, self.t_kc,
							self.env_state, self.data_in)
				
				self.rain = np.array(self.rf.rain)
				self.PET = np.array(self.rf.PET)

				
				## get rainfall
				#rain = PRE.get_one_step_dataset(t_pre, data_in.fname_TSPre, 'pre')
				##rain *= 0.
				## get potential evapotranspiration
				#PET = ET0.get_one_step_dataset(t_eto, data_in.fname_TSMeteo, 'pet')
				height = 174
				width = 201
	
				# check if interception is activated
				if self.env_state.av is None:
					self.SAVIdt = None
					self.SAVIdt_min = None
					self.SAVIdt_max = None
				else:
					self.SAVIdt = np.flip(self.SAVI.get_one_step_dataset(self.t_savi, self.data_in.fname_savi, 'savi').reshape(height, width), 0).flatten()
					self.SAVIdt_min = np.flip(self.SAVImin.get_one_step_dataset(self.t_savi, self.data_in.fname_savi_min, 'savi').reshape(height, width), 0).flatten()
					self.SAVIdt_max = np.flip(self.SAVImax.get_one_step_dataset(self.t_savi, self.data_in.fname_savi_max, 'savi').reshape(height, width), 0).flatten()
				#print(SAVIdt_min)
				# calculate AV
				#print(SAVIdt_max)
				#print(SAVIdt)
				if self.env_state.av is not None:
					self.av = (self.SAVIdt - self.SAVIdt_min)/(self.SAVIdt_max - self.SAVIdt_min)
				else:
					self.av = None
				#print(av)
				# add interception component - UZ zone
				self.Pth, self.Eca, self.PETh, self.LAI, self.abc.kc, self.Sc0_cn = self.cnp.run_interception_one_step( 
						self.rain, self.PET, self.env_state.av,
						self.SAVIdt, self.SAVIdt_max, self.SAVIdt_min,
						None, self.env_state.lai_a, self.env_state.lai_b,
						self.env_state.fcw_cn, self.env_state.Sc0_cn, self.agents.farmers, self.abc.kc)
				
				# Estimate Kc for the riparian area
				self.Pthr, self.Ecar, self.PETr, self.LAIr, self.abc.kc, Sc0_cnrp = self.cnp.run_interception_one_step(
						self.rain, self.PET, self.env_state.av,
						self.SAVIdt, self.SAVIdt_max, self.SAVIdt_min,
						None, self.env_state.lai_a, self.env_state.lai_b,#)
						self.env_state.fcw_cn, self.env_state.Sc0_cnrp, self.abc.kc)
						#rain, PET, env_state.avrip,
						#SAVIdt, env_state.savi_maxrip, env_state.savi_minrip,
						#None, env_state.lai_arip, env_state.lai_brip)

								# estimate abstractions
				
				
				# estimate precipitation over the soil
				# it combines the interception from the hillslopes
				# and the riparian zone
				self.Pth = (self.Pth*(1-self.env_state.riv_factor) 
					+ self.Pthr*(self.env_state.riv_factor))

				if self.agents.model.current_time >= self.agents.model.config['general']['start_time']:

					self.abc.run_ABM_one_step(self.agents.farmers, self.env_state,
						self.Pth, self.env_state.Droot, self.swb.tht_dt, self.env_state.fc,
						self.env_state.SZgrid.at_node['water_table__elevation'],self.swb.aet_dt, self.PETh, self.config, self.ro,
						self.inf.args)	
				
				# Modify Kc for considering interaction with groundwater
				#auxKc = (env_state.grid.at_node['topographic__elevation']
				#	- env_state.SZgrid.at_node['water_table__elevation'])
																
				# add abstraction as rain, still under development
				self.Pth = np.array(self.Pth + self.abc.auz)
				
				# estimate infiltration
				self.inf.run_infiltration_one_step(self.Pth, self.env_state, self.data_in)
				
				# subsurface storage [mm]
				if self.data_in.run_GW > 0:
					self.aux_ssz = storage_uz_sz(self.env_state,
						np.array(self.swb.tht_dt), self.data_in.run_GW)
				
				# soil storage at time t0 [mm]
				self.aux_usz = np.mean((self.swb.L_0[self.env_state.act_nodes]))
						#* env_state.hill_factor)[env_state.act_nodes])
				
				# riparian storage at time t0[mm]			
				self.aux_usp = np.mean((self.swb_rip.L_0
						* self.env_state.riv_factor)[self.env_state.act_nodes])
				
				# adding groundwater evapotranspiration
				# ratio of Etp
				self.ratio_etp = self.gw.wte_dt - self.env_state.zfinal
				self.ratio_etp[self.ratio_etp < 0] = 0
				self.ratio_etp[self.env_state.final_depth > 0] = (
						self.ratio_etp[self.env_state.final_depth > 0]
						/self.env_state.final_depth[self.env_state.final_depth > 0]
						)
				
				self.ratio_etp[self.ratio_etp > 1] = 1
				#print(ratio_etp)
				# evapotranspiration
				if self.abc.kc is not None:
					self.PETh = self.abc.kc *self.PETh
					
				self.PETsz = self.PETh*self.ratio_etp
				self.PETuz = self.PETh - self.PETsz
				
				# estimate soil water balance
				self.swb.run_swbm_one_step(self.inf.inf_dt, self.PETuz,
					np.ones_like(self.PETuz),
					self.env_state.grid.at_node['Ksat_uz'], 
					self.env_state, self.data_in)
				
				# calculate riparial pet
				self.rpet_dt = self.PETuz - self.swb.aet_dt
				
				# estimate available storage at riparian zone
				# change gw discharge from m to mm per unit rip. area
				self.smd, self.qriv, self.inf_rip_dt = self.swb_rip.water_deficit(
					(self.env_state.SZgrid.at_node['discharge']*1000*
					self.env_state.inv_riv_factor),
					self.rpet_dt)
				
				self.env_state.grid.at_node['riv_sat_deficit'][:] += (
					self.smd*self.env_state.rarea)
				#env_state.grid.at_node['riv_sat_deficit'][:] *= np.array(
				#	swb_rip.tht_dt)
				
				# update groundwater discharge
				self.env_state.SZgrid.at_node['discharge'][self.env_state.riv_nodes] = (
					self.qriv[self.env_state.riv_nodes]*
					self.env_state.riv_factor[self.env_state.riv_nodes]*0.001)
				
				
				#print(dataFlux.get_point_dataset_one_step(t_abs),t_abs)
				# Update infiltration excess
				self.exs_dt = np.array(self.inf.exs_dt+self.swb.sro_dt)#[env_state.act_nodes]
				# add data abstractions/sink/source points
				# select row from dataframe and add to the excess component
				if self.dataFlux.data_set is not None:					
					self.exs_dt[idFluxOF] += self.dataFlux.get_point_dataset_one_step(self.t_abs)
				#print(exs_dt)			  
				
				# estimate runoff
				self.ro.run_runoff_one_step(self.exs_dt, self.swb, self.abc.aof, self.env_state, self.data_in)
				
				# change transmission losses rate to riparian area
				# change units from m to mm per unit rip. area
				tls_aux = self.ro.tls_flow_dt*self.env_state.rip_factor
				
				# estimate inputs to riparian zone [mm]
				rip_inf_dt = tls_aux + self.inf_rip_dt
				
				# estimate riparian water balance,
				# use Ksas of the channel in [mm/dt]
				self.swb_rip.run_swbm_one_step(rip_inf_dt, self.rpet_dt,
						np.ones_like(self.rpet_dt),
						self.env_state.grid.at_node['Ksat_ch']*1000., self.env_state,
						self.data_in, self.env_state.river_ids_nodes)
				
				# update focused recharge
				self.swb_rip.pcl_dt += self.swb_rip.sro_dt
				
				# change riparian fluxes to cell area
				self.swb_rip.inf_dt *= self.env_state.riv_factor
				self.swb_rip.luz_dt *= self.env_state.riv_factor
				self.swb_rip.pcl_dt *= self.env_state.riv_factor
				self.swb_rip.aet_dt *= self.env_state.riv_factor
				self.swb_rip.sro_dt *= self.env_state.riv_factor
				
				# correct hill slop fluxes to grid cells
				#swb.pcl_dt *= env_state.hill_factor
				#swb.aet_dt *= env_state.hill_factor
				
				# estimate total groundwater recharge
				self.rech = self.swb.pcl_dt + self.swb_rip.pcl_dt - self.abc.asz# [mm/dt]
				
				#### calculate unsaturated thickness
				###Dusz = (env_state.grid.at_node['topographic__elevation']
				###	- env_state.SZgrid.at_node['water_table__elevation']
				###	- env_state.Droot*0.001)*1000.0
				###
				#### apply dumping to groundwater recharge
				###rech = Qusz.run_recharge_routing(soil, rech, Dusz)
				
				# estimate capillary rise
				#gwe_dt = PET - (swb.aet_dt + swb_rip.aet_dt)
				#gwe_dt[gwe_dt < 0] = 0
				#etg_dt = gw.SZ_potential_ET(env_state, gwe_dt)
				self.etg_dt = self.PETsz
				#print(etg_dt)
				self.outpts.extract_point_var(self.env_state.gaugeidUZ, self.etg_dt)
				
				# temporal aggregation of fluxes for groundwater
				self.etg_agg += np.array(self.etg_dt) # [mm/h]
				self.rch_agg += np.array(self.rech) # [mm/dt]
				
				# save total catchment fluxes for water balance
				self.pre_mb.append(np.mean(self.rain[self.env_state.act_nodes]))
				self.exs_mb.append(np.mean(self.exs_dt[self.env_state.act_nodes]
							+ self.swb.sro_dt[self.env_state.act_nodes]))
				self.tls_mb.append(np.mean(self.ro.tls_dt[self.env_state.act_nodes]))
				self.pth_mb.append(np.mean(self.Pth[self.env_state.act_nodes]))
				self.pthr_mb.append(np.mean(self.Pthr[self.env_state.act_nodes]))
				
				# Save soil interception variables
				if self.env_state.av is not None:
					self.eca_mb.append(np.mean(self.Eca[self.env_state.act_nodes]))
					self.lai_mb.append(np.mean(self.LAI[self.env_state.act_nodes]))
					self.kc_mb.append(np.mean(self.abc.kc[self.env_state.act_nodes]))
					
					self.kcrip_mb.append(np.mean(self.abc.kc[self.env_state.act_nodes]))
					self.ecar_mb.append(np.mean(self.Ecar[self.env_state.act_nodes]))
				else:
					self.eca_mb.append(0)
					self.lai_mb.append(0)
					self.kc_mb.append(0)
					self.kcrip_mb.append(0)
					
				# save soil and riparian AET for water balance
				self.aet_mb.append(np.mean((self.swb_rip.aet_dt
					+self.swb.aet_dt)[self.env_state.act_nodes]))
				
				# save capillary rise from groundwater
				self.egw_mb.append(np.mean(self.etg_dt[self.env_state.act_nodes]))
				
				# save diffuse and focussed recharge for water balance
				self.rch_mb.append(np.mean(self.rech[self.env_state.act_nodes]))
				
				# soil storage at time t1 [mm]
				self.aux_usz1 = np.mean((self.swb.L_0[self.env_state.act_nodes]))
					#*env_state.hill_factor)[env_state.act_nodes])
				
				# riparian storage at time t1 [mm]
				self.aux_usp1 = np.mean((self.swb_rip.L_0
					*self.env_state.riv_factor)[self.env_state.act_nodes])
				
				# channel storage at delta time [mm]
				self.chs_mb.append(np.mean(self.ro.qfl_dt[self.env_state.act_nodes]))
				#			+ swb_rip.sro_dt[env_state.act_nodes]))
				
				# change in soil storage at delta time
				self.uzs_mb.append(self.aux_usz1-self.aux_usz) #[mm]
				
				# change in riparian storage at delta time
				self.rzs_mb.append(self.aux_usp1-self.aux_usp) #[mm]
				
				# activate groundwater component (gw)
				if self.data_in.run_GW > 0:
					if self.dt_GW == self.data_in.dtSZ:
						# empty discharge array
						self.env_state.SZgrid.at_node['discharge'][:] = 0.0
						
						# estimate and change recharge units [mm/h --> m/h]
						self.env_state.SZgrid.at_node['recharge'][:] = (
								self.rch_agg - self.etg_agg)*0.001 #[mm/dt]
						
						# run groundwater component
						if self.data_in.run_GW > 1:
							self.gw.run_one_step_gw_2Layer(self.env_state, self.data_in.dtSZ/60,
								self.swb.tht_dt,	self.env_state.Droot*0.001)
						else:
							self.gw.run_one_step_gw(self.env_state, self.data_in.dtSZ/60,
								self.swb.tht_dt,	self.env_state.Droot*0.001)
						
						# empty array
						self.rch_agg = np.zeros(len(self.swb.L_0))
						self.etg_agg = np.zeros(len(self.swb.L_0))
						self.dt_GW = 0
					
					# time accumulator for gw	
					self.dt_GW += np.int(self.data_in.dt)
				
				# update soil moisture
				if self.data_in.run_GW > 0:
					self.swb.run_soil_aquifer_one_step(self.env_state,
						self.env_state.grid.at_node['topographic__elevation'],
						self.env_state.SZgrid.at_node['water_table__elevation'],
						self.env_state.Duz,
						self.swb.tht_dt)
				
				# update rooting depth
				if self.data_in.run_GW > 0:
					self.env_state.Duz = self.swb.Duz
				
				# estimate groundwater storage change for delta t
				if self.data_in.run_GW > 0:
					self.gws_mb.append(storage_uz_sz(self.env_state,
							np.array(self.swb.tht_dt), self.data_in.run_GW)
							-self.aux_ssz)
				else:
					self.gws_mb.append(0.0)
				
				# save groundwater discharge at outlet for water balance [m]
				self.gbf_mb.append(np.mean(
					self.env_state.SZgrid.at_node['discharge'][self.env_state.act_nodes])
					#+ gw.flux_out
					)
				# save boundary condition flow from saturated component [m]
				self.bcsz_mb.append(self.gw.flux_out*self.env_state.area_cth)
				
				# Extract average state and fluxes
				self.outavg.extract_avg_var_pre(self.env_state.basin_nodes, self.rain, self.PET)
				self.outavg.extract_avg_var_UZ_inf(self.env_state.basin_nodes, self.inf)
				self.outavg.extract_avg_var_UZ_swb(self.env_state.basin_nodes, self.swb)
				self.outavg_rip.extract_avg_var_UZ_swb(self.env_state.basin_nodes, self.swb_rip)
				self.outavg.extract_avg_var_OF(self.env_state.basin_nodes, self.ro)
				self.outavg.extract_avg_var_SZ(self.env_state.basin_nodes, self.gw)
				
				# Extract point state and fluxes
				self.outpts.extract_point_var_UZ_inf(self.env_state.gaugeidUZ, self.inf)
				self.outpts.extract_point_var_UZ_swb(self.env_state.gaugeidUZ, self.swb)				
				self.outpts.extract_point_var_OF(self.env_state.gaugeidOF, self.ro)
				self.outpts.extract_point_var_SZ(self.env_state.gaugeidGW, self.gw)
				
				# extract point variables from riparian zone
				self.outptsRZ.extract_point_var_UZ_swb(self.env_state.gaugeidRZ, self.swb_rip)
				
				if self.data_in.run_GW > 1:						  
					self.outpts.extract_point_var_SZ_L2(
						self.env_state.gaugeidGW, self.env_state.SZgrid)
				
				# get all state and flux variables to grid storage
				self.state_var.get_env_state(self.rf.date_sim_dt, self.t_pre, self.rain, self.PET,
									self.inf, self.swb, self.ro, self.gw, self.swb_rip, self.env_state)
				
				# update soil water content for next iteration
				self.env_state.L_0 = np.array(self.swb.L_0)
				
				# update time steps indices
				self.t_pre += 1
				self.t_savi += 1
				self.t_kc += 1
				self.t_abs +=1
				
			self.t_eto += 1		
		self.t += 1

	def save_output(self):
		
	# store average state and fluxes for water balance calculation
	
		if self.env_state.av is not None:
			self.mb = [self.pre_mb, self.exs_mb, self.tls_mb, self.rch_mb, self.gws_mb,
				self.uzs_mb, self.gbf_mb, self.aet_mb, self.egw_mb, self.chs_mb, self.rzs_mb,
				self.bcsz_mb, self.pth_mb, self.eca_mb, self.lai_mb, self.kc_mb,
				self.kcrip_mb, self.ecar_mb, self.pthr_mb]
		else:
			self.mb = [self.pre_mb, self.exs_mb, self.tls_mb, self.rch_mb, self.gws_mb,
				self.uzs_mb, self.gbf_mb, self.aet_mb, self.egw_mb, self.chs_mb, self.rzs_mb,
				self.bcsz_mb]
		
		# save catchment and riparian average results
		self.outavg.save_avg_var(self.env_state.fnameTS_avg+'.csv',
			self.rf.date_sim_dt)
		self.outavg_rip.save_avg_var(self.env_state.fnameTS_avg+'rip.csv',
			self.rf.date_sim_dt)
		
		# save point results
		self.outpts.save_point_var(self.env_state.fnameTS_OF, self.rf.date_sim_dt,
				self.ro.carea[self.env_state.gaugeidOF],
				self.env_state.rarea[self.env_state.gaugeidOF],
				self.data_in.save_dis_depth)

		# save point results from riparian zone
		self.outptsRZ.save_point_var(self.env_state.fnameTS_RZ, self.rf.date_sim_dt,
				self.ro.carea[self.env_state.gaugeidOF],
				self.env_state.rarea[self.env_state.gaugeidOF],
				self.data_in.save_dis_depth)
		
		# save grided model result datasets 
		self.state_var.save_netCDF_var(self.env_state.fnameTS_avg+'.nc')
		
		# check mass balance
		check_mass_balance(self.env_state.fnameTS_avg, self.outavg, self.outpts,
				self.outavg_rip, self.mb, self.rf.date_sim_dt,
				self.ro.carea[self.env_state.gaugeidOF[0]])
		
		# Save water table for initial conditions
		save_map_to_rastergrid(self.env_state.SZgrid,
				'water_table__elevation',
				self.env_state.fnameTS_avg + '_wte_ini.asc')
		
		# Save soil moisture for initial conditions
		save_map_to_rastergrid(self.env_state.grid,
				'Soil_Moisture',
				self.env_state.fnameTS_avg + '_tht_ini.asc')