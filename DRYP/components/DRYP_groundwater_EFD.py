
import os
import numpy as np
from landlab import RasterModelGrid
from landlab.grid.mappers import (
	map_mean_of_link_nodes_to_link,
	map_max_of_node_links_to_node,
	map_max_of_link_nodes_to_link,
	map_min_of_link_nodes_to_link)
#from landlab.io import read_esri_ascii

from DRYP.components.DRYP_GW_SW_conector import call_update_soil

import time
#Global variables
REG_FACTOR = 0.001 # Regularisation factor
COURANT_2D = 0.250 # Courant Number 2D flow
COURANT_1D = 0.50 # Courant number 1D flow
STR_RIVER = 0.00001 # Riverbed storage factor
# provisional
a_faq = 150
b_faq = 131
# ratio between vertical and horizontal Ksat
fkxy = 0.1
lakes_is_active = 0
class gwflow_EFD(object):
		
	def __init__(self, env_state, data_in):
		"""Initialize groundwater component
		PARAMETERS
		----------
		env_state:		model estates and fluxes
		data_in:		paramters settings
		
		OUTPUT
		------
		water table depth
		groundwater discharge
		flux boundary
		"""
		# Initialize the gw component
		env_state.SZgrid.add_zeros('node', 'discharge', dtype=float)
		
		if data_in.run_GW > 0:
			
			# get active links
			act_links = env_state.SZgrid.active_links
			
			idlink = np.zeros(len(env_state.SZgrid.length_of_link), dtype=int)
			idlink[act_links] = 1
			#print(idlink)
			self.idaqtype = []			
			# check if gw component have different aquifer types
			if env_state.gwtype is not None:
				# find active nodes gw component
				gwact_nodes = np.array(env_state.SZgrid.core_nodes)
				
				# find aquifer types for model domain
				# makes non active nodes none				
				aux_mask = np.zeros_like(env_state.gwtype)
				aux_mask[gwact_nodes] = np.array(env_state.gwtype[gwact_nodes])
				env_state.gwtype = aux_mask
				
				# nodes for each aquifer type
				# Type 1: exponential model
				# Type 2: constant
				# Type 3: linear relation with aquifer thickness
				idnodes_m1 = np.where(env_state.gwtype == 1)[0]
				idnodes_m2 = np.where(env_state.gwtype == 2)[0]
				idnodes_m3 = np.where(env_state.gwtype == 3)[0]
				
				idlink_m1 = env_state.SZgrid.links_at_node[idnodes_m1]
				idlink_m2 = env_state.SZgrid.links_at_node[idnodes_m2]
				idlink_m3 = env_state.SZgrid.links_at_node[idnodes_m3]
				
				# array of aquifer type
				self.idaqtype = [idlink_m1, idlink_m2, idlink_m3]
			#print(casa)
			env_state.SZgrid.add_zeros('node', 'recharge', dtype=float)		
					
			env_state.SZgrid.add_zeros('node',
					'river_stage__elevation', dtype=float)		
			env_state.SZgrid.add_zeros('node',
					'water_storage_anomaly', dtype=float)
			
			env_state.SZgrid.at_node['surface__elevation'] = np.array(
							env_state.grid.at_node['topographic__elevation']
							)
			
			env_state.SZgrid.at_node['topographic__elevation'] = np.array(
							env_state.z_lakes)
					
			Kmax = map_max_of_link_nodes_to_link(
					env_state.SZgrid, 'Hydraulic_Conductivity')		
			Kmin = map_min_of_link_nodes_to_link(
					env_state.SZgrid, 'Hydraulic_Conductivity')		
			Ksl = map_mean_of_link_nodes_to_link(
					env_state.SZgrid, 'Hydraulic_Conductivity')
			
			self.Ksat = np.zeros_like(Ksl)		
			self.Ksat[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]
			
			self.hriv = np.array(env_state.SZgrid.at_node['water_table__elevation'])		
			self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
			self.faq = np.ones_like(Ksl)
			self.faq_node = np.ones_like(self.hriv)
			self.qo = np.ones_like(self.hriv)
			self.Ks = np.array(env_state.SZgrid.at_node['Hydraulic_Conductivity']
					*env_state.fc/env_state.grid.at_node['saturated_water_content']
					)
			print('************************************************************')		
			if env_state.func == 1:
				print('Exponential function for Transmissivity')
			elif env_state.func == 2:
				print('Constant transmissivity')
			elif env_state.func == 3:
				print('Multi-fuction for transmissivity')
			else:
				print('Unconfined conditions: variable thickness')
			print('Change approach in setting_file: line 26')
			print('************************************************************')
			
			# Calculate parameters for expomemtially variable transmissivity
			# Following Fan et al. (2013)
			# f = a / (1 + b) where:
			# f is efective aquifer depth
			# a and b are empirical parameters
			if env_state.func == 1 or env_state.func == 2 or env_state.func == 3:
				# calculate hydraulic gradient
				dzdl = env_state.SZgrid.calc_grad_at_link(
						env_state.SZgrid.at_node['topographic__elevation'])
				# calulate mean paramter a and b at links
				a_faq = map_mean_of_link_nodes_to_link(
						env_state.SZgrid, 'SZ_a_aq')
				b_faq = map_mean_of_link_nodes_to_link(
						env_state.SZgrid, 'SZ_b_aq')
				
				# calculate aquifer effective depth
				self.faq = a_faq/(1+b_faq*np.abs(dzdl))
				# map max effective depth at node
				self.faq_node = map_max_of_node_links_to_node(
						env_state.SZgrid, self.faq)
			
			# calculate mean surface elevation
			self.zm = map_mean_of_link_nodes_to_link(
					env_state.SZgrid, 'topographic__elevation')
			
			# calculate mean soil depth
			Duz_aux = map_max_of_link_nodes_to_link(
					env_state.grid, 'Soil_depth')
			
			# calculate top aquifer elevation for excponential model
			self.zm = np.array(self.zm - Duz_aux*0.001 - 0*self.faq)
			
			# makes fluxes equal to zero
			self.flux_out = 0
			self.act_fix_link = 0
			
			# calculate cell area
			A = np.power(env_state.SZgrid.dx, 2)		
			
			# Calculate river area
			Ariv = (env_state.grid.at_node['river_width']
					* env_state.grid.at_node['river_length'])
			
			# calculate river width
			self.W =  Ariv / env_state.SZgrid.dx	
			
			# calcualte river factor to reduce number of calculations
			kriv = Ariv
			kriv[Ariv != 0] = 1/Ariv[Ariv != 0]
			self.kriv = kriv
			
			# calcualte aquifer-river factor to reduce calculations
			self.kaq = 1/A

			self.f = 30
			self.dh = np.zeros_like(Ariv)
			self.dtSZ = data_in.dtSZ/data_in.dt
			if len(env_state.SZgrid.open_boundary_nodes) > 0:			
				self.fixed_links = env_state.SZgrid.links_at_node[
					env_state.SZgrid.open_boundary_nodes]
				self.act_fix_link = 1
			self.C_factor = np.array(data_in.GW_Cond_factor)
		else:
			self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
			self.dh = np.array(self.wte_dt*0.0)
			self.flux_out = 0.0
		
		# Add a second layer for groundwater 
		if data_in.run_GW > 1:
			
			# Saturated thickness confined conditions
			env_state.SZgrid.add_zeros('node', 'Vb', dtype=float)
			
			self.Vb = (env_state.SZgrid.at_node['BOT']
				-env_state.SZgrid.at_node['BOTb'])
			
			# Calulate specfic storage (storativity)
			env_state.SZgrid.at_node['Ss_2'][:] = (env_state.SZgrid.at_node['Ss_2']
				* self.Vb)
			
			act_links = env_state.SZgrid.active_links		
			Kmax = map_max_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')		
			Kmin = map_min_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')		
			Ksl = map_mean_of_link_nodes_to_link(env_state.SZgrid, 'Ksat_2')
			
			self.Ksat_2 = np.zeros_like(Ksl)		
			self.Ksat_2[act_links] = Kmax[act_links]*Kmin[act_links]/Ksl[act_links]		

		self.lakes_is_active = data_in.lakes
		
	#def add_second_layer_gw(self, env_state, thickness, Ksat, Sy, Ss):	
	#	""" This function add the second layer of the groundwater model
	#	PARAMETERS
	#	----------
	#	thickness:	Thickness of the deep aquifer
	#	Ksat:			Saturated hydraulic conductivity of the deep aquifer
	#	Sy:			Specific yield of the second layer
	#	Ss:			Specific storage of the second layer	
	#	
	#	OUTPUT
	#	------
	#	Landlab raster grid field
	#	"""
		
			

	def run_one_step_gw_2Layer(self, env_state, dt, tht_dt, Droot):	
				
		"""
		Function to update water table depending on the unsaturated zone.
		Parameters:
		Droot:		Rooting depth [mm]
		tht_dt:		Water content at time t [-]
		Duz:		Unsaturated zone depth
		env_state:	grid:	z:	Topograhic elevation
							h:	water table
							fs:	Saturated water content
							fc:	Field capacity
							Sy:	Specific yield
							dq:	water storage anomaly
					
		Groundwater storage variation
		"""
		# Calculate time step ---------------------------------------------------
		# Calculate transmissivity bottom layer
		aux_T = np.minimum(
				env_state.SZgrid.at_node['HEAD_2'],
				env_state.SZgrid.at_node['BOT']
				)
		
		# calculate saturated thickness
		aux_T += -env_state.SZgrid.at_node['BOTb']
		
		# Specify specific storage for unconfined conditons
		sy_aux = np.where(env_state.SZgrid.at_node['HEAD_2'] >
				env_state.SZgrid.at_node['BOT'],
				env_state.SZgrid.at_node['Ss_2'],
				env_state.SZgrid.at_node['Sy_2']
				)
		
		# Calculate transmissivity
		aux_T = aux_T*env_state.SZgrid.at_node['Ksat_2']
		
		# Calculate time step
		dts = time_step_confined(COURANT_2D, sy_aux,
						aux_T, env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes
						)
		
		# Calculate transmissivity upper layer
		aux_T = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
		
		# Calculate saturated thickness
		aux_T += -env_state.SZgrid.at_node['BOT']
		aux_T = aux_T*env_state.SZgrid.at_node['Hydraulic_Conductivity']
		
		# Calculate minimal time step upper layer
		# Only if water table is above the top bottom layer
		if len(aux_T > 0) > 0:			
			dts_aux = time_step_confined(COURANT_2D,
						env_state.SZgrid.at_node['Sy_2'],
						aux_T, env_state.SZgrid.dx,
						env_state.SZgrid.core_nodes
						)
			
			dts = np.nanmin([dts_aux, dts])
						
		# Select the minimal time step
		dtp = np.nanmin([dt, dts])
		dtsp = dtp
		
		# Find active links
		act_links = env_state.SZgrid.active_links
		core_nodes = env_state.SZgrid.core_nodes
		# Make zero all arrays
		env_state.SZgrid.at_node['water_storage_anomaly'][:] = 0.0
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		self.flux_out = 0.0
		stage = env_state.grid.at_node['Q_ini'] * self.kriv *0.0
		aux_riv = np.ones_like(stage)
		aux_riv[stage > 0.0] = 0.0
		
		ti = 0
		
		startTime = time.time()
		
		while dtp <= dt:
			# adjusting heads at the bottom of the model domain
			# WARNING! this could lead to increases in mass balance errors
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
			
			# adjusting head at the surface of the model domain
			# WARNING! this could lead to increases in mass balance errors
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			
			# adjusting bottom heads at the bottom of the model domain
			# WARNING! this could lead to increases in mass balance errors
			env_state.SZgrid.at_node['HEAD_2'][:] = np.maximum(
				env_state.SZgrid.at_node['HEAD_2'],
				env_state.SZgrid.at_node['BOTb']
				)
			
			# calculate saturated thickness upper layer
			dv_u = np.maximum(
				env_state.SZgrid.at_node['water_table__elevation'],
				env_state.SZgrid.at_node['BOT']
				)
			
			dv_u += -env_state.SZgrid.at_node['BOT']
			
			# find active cell
			p = np.where((env_state.SZgrid.at_node['water_table__elevation']
				- env_state.SZgrid.at_node['BOT']) > 0, 1, 0)
						
			# calculate saturated thickness lower layer
			dv_b = np.minimum(
				env_state.SZgrid.at_node['HEAD_2'],
				env_state.SZgrid.at_node['BOT']
				)
			
			dv_b += -env_state.SZgrid.at_node['BOTb']			
			#dv_b[dv_b < 0] = 0

			# Calculate vertical head difference
			dhz = np.minimum(
				env_state.SZgrid.at_node['HEAD_2'],
				env_state.SZgrid.at_node['BOT']
				)
			
			dhz += -env_state.SZgrid.at_node['BOT']
			
			dhz = dv_u - dhz
			#print(dhz[core_nodes], dv_u[core_nodes])
			
			#haux = np.array(env_state.SZgrid.at_node['BOT'])
			#haux[env_state.SZgrid.at_node['HEAD_2'][:] > haux] = (
			#	env_state.SZgrid.at_node['HEAD_2'][env_state.SZgrid.at_node['HEAD_2'][:] > haux]
			#	)
			#dhy = env_state.SZgrid.at_node['water_table__elevation'] - haux
			
			# Calculate vertical distance bottom layer
			dv_by = np.array(dv_b)
			dv_by[env_state.SZgrid.at_node['HEAD_2'][:] < env_state.SZgrid.at_node['BOT'][:]] = 0
			
			# Calculate vertical distance between two layers			
			#dy = np.array(dv_u)
			#dy[dv_u > 0] += dv_by[dv_u > 0]
			
			# Calculate Vertical conductivity
			CV = (0.5*dv_u/(fkxy*env_state.SZgrid.at_node['Hydraulic_Conductivity'])
				+ 0.5*dv_by/(fkxy*env_state.SZgrid.at_node['Ksat_2'])
				)
			
			CV[dv_by < self.Vb] = (fkxy*env_state.SZgrid.at_node['Ksat_2'])[dv_by < self.Vb]
			#print(CV[env_state.SZgrid.core_nodes])
			#print(env_state.SZgrid.at_node['recharge'][env_state.SZgrid.core_nodes])
			#print(env_state.SZgrid.at_node['HEAD_2'][env_state.SZgrid.core_nodes])			
			#CV[dv_by < env_state.SZgrid.at_node['Vb'][:]] = (1
			#	/(fkxy*env_state.SZgrid.at_node['Hydraulic_Conductivity'][
			#	dv_by < env_state.SZgrid.at_node['Vb'][:]]))
			
			# Adjusting the vertical transmissivity in contact layers
			ksat_aux = dv_u/0.1			
			ksat_aux[dv_u < 0] = 1
			ksat_aux[ksat_aux > 1] = 1
			ksat_aux = np.exp(-100*(1-ksat_aux))
			ksat_aux[dhz > 0] = 1
			
			dhz[dv_by < self.Vb] = 1
			
			# Vertical flux
			dqz = (CV)*dhz*ksat_aux
			#print(dqz[env_state.SZgrid.core_nodes])			
			# Calculate horizontal hydraulic gradients: Top layer
			dhdl_u = env_state.SZgrid.calc_grad_at_link(
					env_state.SZgrid.at_node['water_table__elevation'])
			
			# Calculate horizontal hydraulic gradients: Bottom layer
			dhdl_b = env_state.SZgrid.calc_grad_at_link(
					env_state.SZgrid.at_node['HEAD_2'])
			
			# Calculate mean hydraulic head at the face of the of the patch
			bm_u = map_mean_of_link_nodes_to_link(env_state.SZgrid, dv_u)#'water_table__elevation')
			bm_b = map_mean_of_link_nodes_to_link(env_state.SZgrid, dv_b)#'BOT')
						
			# Calculate horizontol flux: upper layer
			qxy_u = np.zeros_like(self.Ksat)
			ksat_aux = bm_u/0.1			
			#ksat_aux[dhdl_u < 0] = 1
			ksat_aux[ksat_aux > 1] = 1
			ksat_aux = np.exp(-100*(1-ksat_aux))
			qxy_u[act_links] = -self.Ksat[act_links]*bm_u[act_links]*ksat_aux[act_links]*dhdl_u[act_links]
						
			# Calculate horizontol flux: Bottom layer
			#bb = np.minimum(hm-bm, self.thickness)		
			qxy_b = np.zeros_like(self.Ksat_2)
			ksat_aux = bm_b/0.1			
			#ksat_aux[dhdl_b < 0] = 1
			ksat_aux[ksat_aux > 1] = 1
			ksat_aux = np.exp(-100*(1-ksat_aux))
			qxy_b[act_links] = -self.Ksat_2[act_links]*bm_b[act_links]*ksat_aux[act_links]*dhdl_b[act_links]
			
			# Calculate divergence: top layer
			dqxy_u = (-env_state.SZgrid.calc_flux_div_at_node(qxy_u)
					- dqz + p*env_state.SZgrid.at_node['recharge']/dt)#*dtsp
			
			# Calculate seepage surface
			dqs = regularization(
					env_state.SZgrid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT'],
					dqxy_u, REG_FACTOR)
			
			# Calculate channel cell conductivity
			Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)

			diff_stage = (env_state.SZgrid.at_node['water_table__elevation']
				- self.hriv)
			
			stage_aux = np.array(stage)
			stage_aux[diff_stage < 0.0] = 0.0
						
			# Calculate river cell flux [m3 h-1]
			qs_riv = -(Tch*(diff_stage-stage_aux)*self.C_factor)#/
			qs_riv[qs_riv < 0.0] = qs_riv[qs_riv < 0.0]*aux_riv[qs_riv < 0.0]
			
			# river mass balance
			dqxy_u += self.kaq*qs_riv			
			
			# update water storage anomaly: upper layer
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqxy_u-dqs)*dtsp
			
			if self.act_fix_link == 1:
				self.flux_out += (np.sum(qxy_u[self.fixed_links]
								+ qxy_b[self.fixed_links])
								/env_state.SZgrid.dx)*dtsp
			
			
			# Update storage change for soil-gw interactions
			env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] = (
				fun_update_UZ_SZ_depth(
				np.array(env_state.SZgrid.at_node['water_storage_anomaly'][env_state.SZgrid.core_nodes]),#dS
				np.array(env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes]),#h0
				np.array(tht_dt[env_state.SZgrid.core_nodes]),#tht_dt
				np.array(env_state.grid.at_node['saturated_water_content'][env_state.SZgrid.core_nodes]),#tht_sat
				np.array(env_state.fc[env_state.SZgrid.core_nodes]),#tht_fc
				np.array(env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes]),#Sy
				np.array(env_state.SZgrid.at_node['topographic__elevation'][env_state.SZgrid.core_nodes]
				- Droot[env_state.SZgrid.core_nodes])#zr)
				))			
			
			# Calculate divergence: bottom layer
			dqxy_b = (-env_state.SZgrid.calc_flux_div_at_node(qxy_b)
					+ dqz + (1-p)*env_state.SZgrid.at_node['recharge']/dt)#*dtsp
					
			# Calculate smooth function parameters
			FSs = smoth_func_L2(env_state.SZgrid.at_node['HEAD_2'][:],
				env_state.SZgrid.at_node['BOT'][:], 0.5,
				REG_FACTOR, env_state.SZgrid.core_nodes)
			#print(FSs[env_state.SZgrid.core_nodes])
			
			# Regularization approach for river cells
			dqs_riv = regularization_T(
				env_state.grid.at_node['river_topo_elevation'],
				self.hriv, 	self.f, -qs_riv, REG_FACTOR
				)
			
			# Calculate total discharge
			env_state.SZgrid.at_node['discharge'][:] += (dqs + dqs_riv*self.kaq)*dtsp
			#print(dqxy_b[env_state.SZgrid.core_nodes])
			# Update hydraulic head lower layer
			env_state.SZgrid.at_node['HEAD_2'][env_state.SZgrid.core_nodes] += np.array(
				dqxy_b[env_state.SZgrid.core_nodes]/
				((1-FSs[env_state.SZgrid.core_nodes])
				*env_state.SZgrid.at_node['Sy_2'][env_state.SZgrid.core_nodes]
				+ FSs[env_state.SZgrid.core_nodes]
				* env_state.SZgrid.at_node['Ss_2'][env_state.SZgrid.core_nodes])
				)*dtsp
						
			
			# Calculate maximum time step --------------------------------
			# Calculate transmissivity bottom layer
			aux_T = np.minimum(
					env_state.SZgrid.at_node['HEAD_2'],
					env_state.SZgrid.at_node['BOT']
					)
			# Specify specific storage if confined
			sy_aux = np.where(env_state.SZgrid.at_node['HEAD_2'] >
					env_state.SZgrid.at_node['BOT'], env_state.SZgrid.at_node['Ss_2'],
					env_state.SZgrid.at_node['Sy_2'])
			# Saturated thickness
			aux_T += -env_state.SZgrid.at_node['BOTb']
			aux_T = aux_T*env_state.SZgrid.at_node['Ksat_2']
			
			dts = time_step_confined(COURANT_2D, sy_aux,
							aux_T, env_state.SZgrid.dx,
							env_state.SZgrid.core_nodes)
			
			# Calculate transmissivity upper layer
			aux_T = np.maximum(
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT']
					)
			aux_T += -env_state.SZgrid.at_node['BOT']
			aux_T = aux_T*env_state.SZgrid.at_node['Hydraulic_Conductivity']
			
			# Calculate maximum time step, upper layer
			if len(aux_T > 0) > 0:
				dts_aux = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['Sy_2'],
							aux_T, env_state.SZgrid.dx,
							env_state.SZgrid.core_nodes)
				
				dts = np.nanmin([dts_aux, dts])			
			
			if dtsp <= 0:
				raise Exception("invalid time step", dtsp)
			if dtp == dt:
				dtp += dtsp
			elif (dtp + dtsp) > dt:
				dtsp = dt - dtp
				dtp += dtsp
			else:
				dtp += dtsp
			
			ti += 1
			
		#print ('time=', ti, time.time() - startTime, dtsp)
		# Update state variables
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		env_state.SZgrid.at_node['discharge'][:] *= (1/self.dtSZ)
		
		env_state.grid.at_node['riv_sat_deficit'][:] = (
				np.power(env_state.grid.dx, 2)
				* np.array(env_state.grid.at_node['river_topo_elevation'][:]
				- env_state.SZgrid.at_node['water_table__elevation'][:])
				)
		
		env_state.grid.at_node['riv_sat_deficit'][env_state.grid.at_node['riv_sat_deficit'][:] < 0] = 0.0
		
		env_state.grid.at_node['riv_sat_deficit'] *= env_state.SZgrid.at_node['SZ_Sy']
		
		if self.act_fix_link == 1:
			self.flux_out *= 1/self.dtSZ
		
		pass
	#@profile
	def run_one_step_gw(self, env_state, dt, tht_dt, Droot):
		"""Function to update water table depending on the unsaturated zone.
		PARAMETERS:
		-----------
		Droot:		Rooting depth [mm]
		tht_dt:		Water content at time t [-]
		Duz:		Unsaturated zone depth [m]
		env_state:	grid:	z:	Topograhic elevation [m]
							h:	water table [m]
							fs:	Saturated water content [-]
							fc:	Field capacity [-]
							Sy:	Specific yield [-]
							dq:	water storage anomaly [m]
							f:	effective aquifer depth [m]
							SS_loss: transmission losses [m3 h-1]
		OUTPUT
		------
		Groundwater storage variation [m]
		water table elevation [m]
		groundwater discharge [m]
		"""
		# Calculate time step ---------------------------------------------------
		#print('===============================================================')
		# select active link of model domain
		act_links = env_state.SZgrid.active_links
		core_nodes = np.array(env_state.SZgrid.core_nodes, dtype=int)
		
		# Calculate transmissivity
		T = transmissivity(env_state, self.Ksat, act_links, self.faq,
						self.zm, self.idaqtype
						)
		
		# Calculate time step
		dts = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
			map_max_of_node_links_to_node(env_state.SZgrid, T),
			env_state.SZgrid.dx, core_nodes
			)
					
		# Calculate minimal time step		
		dtp = np.nanmin([dt, dts])		
		dtsp = dtp
		
		# Make zero all arrays
		env_state.SZgrid.at_node['discharge'][:] = 0.0
		env_state.SZgrid.at_node['water_storage_anomaly'][:] = 0.0
		self.flux_out = 0.0		
		stage = env_state.grid.at_node['Q_ini'] * self.kriv	*0.0
		aux_riv = np.ones_like(stage)		
		aux_riv[stage > 0.0] = 0.0
		#print(env_state.SZgrid.at_node['topographic__elevation'][219])
		ti = 0
		while dtp <= dt:
			#startTime = time.time()
			# adjusting heads at the bottom of the model domain
			# WARNING! this could lead to increases in mass balance errors
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			
			# adjusting head at the surface of the model domain
			# WARNING! this could lead to increases in mass balance errors
			if env_state.func == 2:
				env_state.SZgrid.at_node['water_table__elevation'][:] = np.maximum(
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT']
					)
			
			# Make river water table always below or equal surface elevation
			self.hriv = np.minimum(self.hriv,
				env_state.grid.at_node['river_topo_elevation']
				)
			
			# Calculate transmissivity
			T = transmissivity(env_state, self.Ksat, act_links, self.faq,
							self.zm, self.idaqtype
							)
					
			# Calculate the hydraulic gradients			
			dhdl = env_state.SZgrid.calc_grad_at_link(
				env_state.SZgrid.at_node['water_table__elevation']
				)
			#print(dhdl)
			# Calculate flux per unit length at each face
			qs = np.zeros_like(self.Ksat)
			qs[act_links] = -T[act_links]*dhdl[act_links]
			
			# calculate total flux at constant head boundary condition cells
			# units are m per time step
			if self.act_fix_link == 1:
				self.flux_out += (np.sum(qs[self.fixed_links])
								/env_state.SZgrid.dx)*dtsp
				
			# Add flux at boundary conditions
			dfhbc = env_state.SZgrid.at_node['SZ_FHB']
			#print(len(core_nodes))
			# save flux boundary condition as average for the whole basin domain
			# units are in m per time step
			self.flux_out += np.mean(dfhbc[core_nodes])*dtsp
			#print(self.flux_out)
			# Calculate flux gradient
			dqsdxy = (-env_state.SZgrid.calc_flux_div_at_node(qs)
					- dfhbc + env_state.SZgrid.at_node['recharge']/dt)
			
			# Calculate channel cell conductivity
			Tch = exponential_T(env_state.grid.at_node['SS_loss'], STR_RIVER,
				env_state.grid.at_node['river_topo_elevation'], self.hriv)

			diff_stage = (env_state.SZgrid.at_node['water_table__elevation']
				- self.hriv)
			
			stage_aux = np.array(stage)
			stage_aux[diff_stage < 0.0] = 0.0
						
			# Calculate river cell flux [m3 h-1]
			qs_riv = -(Tch*(diff_stage-stage_aux)*self.C_factor)#/
			qs_riv[qs_riv < 0.0] = qs_riv[qs_riv < 0.0]*aux_riv[qs_riv < 0.0]
			
			# Regularization approach for river cells
			dqs_riv = regularization_T(
				env_state.grid.at_node['river_topo_elevation'],
				self.hriv, 	self.f, -qs_riv, REG_FACTOR
				)
			
			# add river out/inflow to the mass balance
			# change river flow units m3 -> m
			dqsdxy += -self.kaq*dqs_riv
			#print(dqs_riv[219], dqsdxy[219])
			# Regularization approach for aquifer cells
			if env_state.func == 1 or  env_state.func == 2:
				dqs = regularization_T(
					env_state.SZgrid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					self.faq_node, dqsdxy, REG_FACTOR
					)
				
			else:
				dqs = regularization(
					env_state.SZgrid.at_node['topographic__elevation'],
					env_state.SZgrid.at_node['water_table__elevation'],
					env_state.SZgrid.at_node['BOT'],
					dqsdxy, REG_FACTOR)
					
			
			
			# Calculate storage change			
			env_state.SZgrid.at_node['water_storage_anomaly'][:] = (dqsdxy-dqs)*dtsp
			#print(dqs[219]*dtsp, (dqsdxy[219]-dqs[219])*dtsp, dqs_riv[219], dtsp)
			# update river ghost cell
			#self.hriv += -(qs_riv+dqs_riv)*dtsp/env_state.SZgrid.at_node['SZ_Sy']
			#print(env_state.SZgrid.at_node['water_table__elevation'][219])#[core_nodes[211]])
			#print(env_state.SZgrid.at_node['SZ_Sy'][219])
			# enable lakes layer
			if self.lakes_is_active == 0:
				# Update storage change for soil-gw interactions
				env_state.SZgrid.at_node['water_table__elevation'][core_nodes] = (
					fun_update_UZ_SZ_depth(
					np.array(env_state.SZgrid.at_node['water_storage_anomaly'][core_nodes]),#dS
					np.array(env_state.SZgrid.at_node['water_table__elevation'][core_nodes]),#h0
					np.array(tht_dt[core_nodes]),#tht_dt
					np.array(env_state.grid.at_node['saturated_water_content'][core_nodes]),#tht_sat
					np.array(env_state.fc[core_nodes]),#tht_fc
					np.array(env_state.SZgrid.at_node['SZ_Sy'][core_nodes]),#Sy
					np.array(env_state.SZgrid.at_node['topographic__elevation'][core_nodes]
					- Droot[core_nodes])#zr)
					))
			else:
				#print('a',env_state.SZgrid.at_node['water_storage_anomaly'][core_nodes[40]])
				# Update storage change for soil-gw interactions
				env_state.SZgrid.at_node['water_table__elevation'][core_nodes] = (
					call_update_soil(
					np.array(env_state.SZgrid.at_node['topographic__elevation'][core_nodes]),# lake
					np.array(env_state.SZgrid.at_node['surface__elevation'][core_nodes]),# lake
					(np.array(env_state.SZgrid.at_node['surface__elevation'][core_nodes]
					-Droot[core_nodes])),
					1, 0, 0,# assuming lake conditons
					np.array(env_state.grid.at_node['saturated_water_content'][core_nodes]),#tht_sat
					np.array(env_state.fc[core_nodes]),#tht_fc
					np.array(tht_dt[core_nodes]),#tht_dt
					np.array(env_state.SZgrid.at_node['water_storage_anomaly'][core_nodes]),#dS
					np.array(env_state.SZgrid.at_node['water_table__elevation'][core_nodes]),#h0
					np.array(env_state.SZgrid.at_node['SZ_Sy'][core_nodes]),#Sy
					))
			#print(env_state.SZgrid.at_node['water_table__elevation'][219])#[core_nodes[40]])
			# Calculate total discharge
			env_state.SZgrid.at_node['discharge'][:] += (dqs + dqs_riv*self.kaq)*dtsp
			#print('discharge',env_state.SZgrid.at_node['discharge'][219])
			# Calculate maximum time step
			dtsp = time_step_confined(COURANT_2D, env_state.SZgrid.at_node['SZ_Sy'],
				map_max_of_node_links_to_node(env_state.SZgrid, T),
				env_state.SZgrid.dx, core_nodes
				)
			
			#dtsp_riv = time_step_confined(COURANT_1D, env_state.SZgrid.at_node['SZ_Sy'],
			#			Tch/self.W, 0.02*(env_state.SZgrid.dx - self.W), env_state.riv_nodes)
			
			#dtsp = np.min([dtsp, dtsp_riv])			
			
			
			# adjusting head at the surface of the model domain
			# WARNING! this could lead to increases in mass balance errors
			env_state.SZgrid.at_node['water_table__elevation'][:] = np.minimum(
				env_state.SZgrid.at_node['topographic__elevation'],
				env_state.SZgrid.at_node['water_table__elevation']
				)
			#print(dtsp)
			# Update time step
			if dtsp <= 0:
				raise Exception("invalid time step", dtsp)			
			if dtp == dt:			
				dtp += dtsp			
			elif (dtp + dtsp) > dt:			
				dtsp = dt - dtp				
				dtp += dtsp				
			else:			
				dtp += dtsp
		
		# Update state variables
		self.wte_dt = np.array(env_state.SZgrid.at_node['water_table__elevation'])
		
		# change discharge to model time step
		env_state.SZgrid.at_node['discharge'][:] *= (1/self.dtSZ)
		
		# calculate available storage for groundwater [m3]
		env_state.grid.at_node['riv_sat_deficit'][:] = (
				np.power(env_state.grid.dx, 2)
				* np.array(env_state.grid.at_node['river_topo_elevation'][:]
				- env_state.SZgrid.at_node['water_table__elevation'][:])
				)
		
		# if water table is above the surface, make storage zero
		env_state.grid.at_node['riv_sat_deficit'][env_state.grid.at_node['riv_sat_deficit'][:] < 0] = 0.0

		# estimate availble water storage of the aquifer
		env_state.grid.at_node['riv_sat_deficit'] *= env_state.SZgrid.at_node['SZ_Sy']
		
		# change time step of flux leaving the basin to model time step
		if self.act_fix_link == 1:
			self.flux_out *= 1/self.dtSZ
		
		pass
	
	def recharge(self, env_state, dt):
		k = (env_state.SZgrid.at_node['water_table__elevation']
			- env_state.grid.at_node['topographic__elevation']
			+ env_state.Droot*0.001)	
		k[k > 0] = 0
		aux = np.array(k)
		aux[aux < 0] = 1
		k[k != 0] = np.exp(dt*self.Ks[k != 0]#env_state.grid.at_node['Hydraulic_Conductivity'][k != 0]/
					/k[k != 0])
		env_state.SZgrid.at_node['recharge'] = (self.qo*k 
				+ env_state.grid.at_node['recharge']*(1-k))+self.qo*(1-aux)
		
		self.qo = np.array(env_state.SZgrid.at_node['recharge'])*aux
		
	def SZ_potential_ET(self, env_state, pet_sz):
		"""Capillary rise - plant groundwater uptake
		Linear relation depending on water table depth
		PARAMETERS
		----------
		pet:		potential evapotranspiration for GW
		env_state:	model state variables (Droot)
		
		OUTPUT
		------
		capillary rise
		"""
		# calculate saturated zone for uptake
		depth_aet = (env_state.SZgrid.at_node['water_table__elevation']
			- env_state.grid.at_node['topographic__elevation']
			+ env_state.Droot*0.001)*1000		
		depth_aet[depth_aet < 0] = 0		
		
		# evapotranspiration proportion
		f = depth_aet/(env_state.gwet_lim*env_state.Droot)
		f[f > 1] = 1
		
		return f*pet_sz

def transmissivity(env_state, Ksat, act_links, f, zm, idlink):
	"""Calculate aquifer transmissivity
	PARAMETERS:
	-----------
	env_state:		environmental variables
	Ksat:			Saturated hydraulic conductivity aquifer
	act_links:		array of active links of SZ domain
	f:				effective aquifer depth
	zm:				surface elevation at link (node average)
	OUTPUT:
	-------
	T:				Transmissivity
	"""
	T = np.zeros_like(Ksat)
	
	# Calculate mean water table elevation at link
	if env_state.func != 2:
		hm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'water_table__elevation')
	
	# calculate mean bottom elevation
	if env_state.func == 0 or env_state.func == 3:
		bm = map_mean_of_link_nodes_to_link(env_state.SZgrid,'BOT')
	
	# Calculate transmisivity for especific aquifer types
	if env_state.func == 1:
		# exponential transmissivity
		T = exponential_T(Ksat, f, zm, hm)
	
	elif env_state.func == 2:
		# Constant transmissivity
		T[act_links] = Ksat[act_links]*300
	
	elif env_state.func == 3:
		#print(idlink)
		# multi type aquifer
		# Type 1: exponential model
		if len(idlink[0]) > 0:
			T[idlink[0]] = exponential_T(Ksat[idlink[0]],
						f[idlink[0]], zm[idlink[0]],
						hm[idlink[0]]
						)
		# Type 2: constant
		if len(idlink[1]) > 0:
			T[idlink[1]] = Ksat[idlink[1]]*100
		
		# Type 3: linear relation with aquifer thickness
		if len(idlink[2]) > 0:
			T[idlink[2]] = Ksat[idlink[2]]*(
					hm[idlink[2]]-bm[idlink[2]]
					)
		#print(T[act_links])
	else:
		# linear relation with aquifer thickness		
		T[act_links] = Ksat[act_links]*(hm[act_links]-bm[act_links])
	#print(T[act_links])
	#print(act_links)
	
	return T

def fun_update_UZ_SZ_depth(dS, h0, tht_dt, tht_sat, tht_fc, Sy, zr):
	"""Function to update water table depending on both water content of
	the unsaturated zone.
	Parameters:
	INPUT:
	------
	dS:			water storage anomaly [m]
	h0:			initail water table [m]
	tht_dt:		water content at t0 [--]
	tht_sat:	water content at saturated conditions [--]
	tht_fc:		water content at field capacity [--]
	Sy:			specific yield [-]
	zr:			root zone elevation [m]
	OUTPUT:
	-------
	h:	updated water table
	"""	
	
	tht_dt = np.where(dS >= 0.0,
		tht_sat - tht_dt,
		tht_sat - tht_fc,
		)
	
	alpha = np.where(dS >= 0.0,
		1 - Sy/tht_dt,
		1 - tht_dt/Sy,
		)
	
	beta = np.where(dS >= 0.0,
		dS/tht_dt, dS/Sy
		)
	
	dSp = np.where(dS >= 0.0,
		(zr-h0)*Sy, tht_dt*(h0-zr)
		)
	
	dSp[dSp <= 0] = 0.0
	
	C = np.where(np.abs(dSp) < np.abs(dS), 0, 1)	
	alpha = np.where(np.abs(dSp) < np.abs(dS), alpha, 0)	
	beta = np.where(np.abs(dSp) < np.abs(dS), beta, 0)
	
	alpha = np.where(np.abs(dSp) == 0, 0, alpha)	
	beta = np.where(np.abs(dSp) == 0, 0, beta)
		
	D = np.where(h0 > zr, 0, 1)
	
	D = np.where(dS > 0, 0, D)
	
	C[np.abs(dSp) == 0] = 1.0
	
	gama = dS/tht_dt
	lambd = dS/Sy,
	
	# Update water table elevation
	h = h0 +(zr-h0)*alpha + beta + ((1-D)*gama + D*lambd)*C
	
	return h

def storage(env_state):
	storage = np.sum((env_state.SZgrid.at_node['water_table__elevation'][env_state.SZgrid.core_nodes] -\
		env_state.SZgrid.at_node['BOT'][env_state.SZgrid.core_nodes])*\
		env_state.SZgrid.at_node['SZ_Sy'][env_state.SZgrid.core_nodes])
	return storage
	
def storage_uz_sz(env_state, tht, *two_layer):
	""" Total storage in the saturated zone
	Parameters:
		env_state:	state variables and model parameters			
	Output:
		total:		Volume of water stored in the saturated zone [mm]
	"""
	
	# estimate storage water available in the rooting zone
	str_usz = (env_state.SZgrid.at_node['water_table__elevation']
		- (env_state.grid.at_node['topographic__elevation']
		- env_state.Droot*0.001)
		)
	
	# estimate saturated-unsaturated storage
	str_usz[str_usz < 0] = 0.0
	
	# estimate rooting depth storage
	str_uz = env_state.Droot*0.001 - str_usz
	
	# estimate saturated storage
	str_sz = np.array(env_state.SZgrid.at_node['water_table__elevation']
		- np.array(str_usz)-env_state.SZgrid.at_node['BOT'])
	
	#print(np.mean((str_uz*tht)[env_state.SZgrid.core_nodes]))#[219])
	#print(np.mean((str_usz*env_state.grid.at_node['saturated_water_content'])[env_state.SZgrid.core_nodes]))#[219]))
	#print(np.mean((str_sz*env_state.SZgrid.at_node['SZ_Sy'])[env_state.SZgrid.core_nodes]))#[219])
	
	# total storage
	total = (str_uz*tht
			+ str_usz*env_state.grid.at_node['saturated_water_content']
			+ str_sz*env_state.SZgrid.at_node['SZ_Sy']
			)
	
	if two_layer[0] == 2:
	
		# saturated thickness (unconfined)
		b = np.minimum(env_state.SZgrid.at_node['HEAD_2'],
			env_state.SZgrid.at_node['BOT'])
			
		b = b - env_state.SZgrid.at_node['BOTb']	
		
		# Storage from unconfined conditions
		b = b*env_state.SZgrid.at_node['Sy_2']
		
		# Specific torage from confined conditions
		Ss = np.array(env_state.SZgrid.at_node['HEAD_2']
			- env_state.SZgrid.at_node['BOT'])
		
		Ss[Ss <= 0] = 0
		
		# Update storage in case of confined conditions
		b[Ss > 0] = 0
		
		# Update head for Specific storage
		Ss[Ss > 0] += (env_state.SZgrid.at_node['BOT'][Ss > 0]
			- env_state.SZgrid.at_node['BOTb'][Ss > 0])
		
		# Water from confined store
		str_sz2l = np.array(Ss*env_state.SZgrid.at_node['Ss_2'])

		# storage second layer
		str_sz2l = b + str_sz2l
		total += str_sz2l
	#print(total[env_state.SZgrid.core_nodes])
	#print(np.mean(total[env_state.SZgrid.core_nodes])*1000)
	return np.mean(total[env_state.SZgrid.core_nodes])

def smoth_func_L1(h, hr, r, dq, *nodes):

	aux = np.power(h-hr, 3)/r	
	aux = np.where(aux > 0, aux, 0)
	aux = np.where(aux > 1, 1, aux)
	aux = np.where(dq > 0, 1, aux)
	
	if nodes:
		p = np.zeros_like(aux)
		p[nodes] = 1
		aux *= p
	
	return  aux
	
def smoth_func_L2(h, zb, D, r, *nodes):
	""" Reduce the hydraulic conductivity Ksat
	when the water table is close to the bottom of the layer
	
	PARAMETERS
	----------
	h:	head elevation [m]
	hr:	bottom elevation [m]
	OUTPUT
	------
	D:	distance for smoothing [m]
	r:	smoothing parameter [-]
	"""
	
	# calculate ration of smoothing
	u = (h-zb)/D
	#aux = (1-u)/r
	#
	u[u < 0] = 0
	u[u > 1] = 1
	#
	#FSy = 1 - np.exp(-aux)
	#FSs = 1 - np.exp(aux)
	#
	#FSy = np.where(u >= 1, 0, FSy)
	#FSs = np.where(u >= 1, FSs, 0)
	#	
	#if nodes:
	#	aux = np.zeros_like(aux)
	#	aux[nodes] = 1
	#	FSs *= aux
	#	FSy *= aux
	
	return  u

def river_flux(h, hriv, C, A, *nodes):
	q_riv = (h-hriv)*C/A
	if nodes:
		p = np.zeros_like(aux)
		p[nodes] = 1
		aux *= p
	return  q_riv

def smoth_func_T(h, hriv, r, f, dq, *nodes):
	SF = (h-hriv+f)/f
	SF = np.where(SF > 1, SF, 0)
	SF = np.where(SF > 0, 1 - np.exp(SF/r), 0)
	if nodes:
		p = np.zeros_like(SF)
		p[nodes] = 1
		SF *= p
	return SF

def smoth_func(h, hriv, r, dq, *nodes):
	aux = np.power(h-hriv, 3)/r	
	aux = np.where(aux > 0, aux, 0)
	aux = np.where(aux > 1, 1, aux)
	if nodes:
		p = np.zeros_like(aux)
		p[nodes] = 1
		aux *= p
	return  np.where(aux <= 0, 0, aux)
	
 
def regularization(zm, hm, bm, dq, r):
	"""regularization function for unconfined
	zm:	surface elevation
	hm:	hydraulic head
	bm:	bottom elevation aquifer
	dq:	flux per unit area
	r:	regularization factor
	"""
	aux = (hm-bm)/(zm-bm)
	aux = np.where((aux-1) > 0, 1, aux)
	return np.exp((aux-1)/r)*dq*np.where(dq > 0, 1, 0)


def regularization_T(zm, hm, f, dq, r):
	"""regularization function for confined aquifers
	zm:	surface elevation
	hm:	hydraulic head
	f:	e-folding depth
	dq:	flux per unit area
	r:	regularization factor
	"""
	aux = (hm-zm)/f+1	
	aux = np.where(aux > 0,aux,0)
	return np.exp((aux-1)/r)*dq*np.where(dq > 0,1,0)
	
def exponential_T(Ksat, f, z, h):
	"""Calculate aquifer transmissivity following
	Fan et. al. (2013)
	
	PARAMETERS:
	-----------
	Ksat:	Saturated hydraulic conductivity aquifer
	f:		effective aquifer depth
	z:		elevation at link (node average)
	h:		water table elevation
	OUTPUT:
	-------
	transmissivity
	"""
	
	return Ksat*f*np.exp(-np.maximum(z-h, 0)/f)


def time_step(D, Sy, Ksat, h, zb, dx, *nodes):
	""" Maximum time step for unconfined aquifers
	INPUT
	-----
	D:		Courant number
	Sy:		specific yield
	Ksat:	saturated hydraulic conductivity
	h:		water table elevation
	zb:		aquifer bottom elevation
	dx:		grid size
	nodes:	nodes at which the calculation will be applied
	"""
	T = (h - zb)*Ksat
	#print(T[nodes])
	dt = D*Sy*np.power(dx, 2)/(T)
	
	if nodes:		
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
		#print('a',dt)
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt

def time_step_confined(D, Sy, T, dx, *nodes):
	"""Maximum time step for confined aquifers
	D:	Courant number
	"""
	dt = D*Sy*np.power(dx, 2)/(T)
	
	if nodes:		
		dt = np.nanmin((dt[nodes])[dt[nodes] > 0])
	else:
		dt = np.nanmin(dt[dt > 0])
	return dt

class recharge_routing(object):
	def __init__(self, grid_size):
		"""Apply a damping effect ot the percolation
		INPUT:
		------
		OUTPUT:
		-------
		"""
		self.Susz0 = np.zeros(grid_size)
		

	def run_recharge_routing(self, soil, R, Dusz):
		"""Apply a damping effect ot the percolation
		INPUT:
		------
		soil:	lamda parameter
		Ksat: 	soil hydraulic conductivity
		R:		percolation 
		Dusz:	distance between the routing depth and the
				water table [m]
		OUTPUT:
		-------
		Qusz:	recharge []
		"""
		# update the storage of the unsaturated zone
		Susz = self.Susz0 + R
		
		# Maximum storage of the unsaturated zone
		Suszmax = Dusz*0.01#soil.theta_sat
		
		# flow velocity of through the unsaturated zone
		aux = np.zeros_like(Dusz)
		
		aux[Dusz > 0] = Susz[Dusz > 0]/Suszmax[Dusz > 0]
		
		v = soil.Ksat_uz*np.power(aux, soil.c_SOIL)
		
		aux = np.ones_like(Dusz)
		
		aux[Dusz > 0] = v[Dusz > 0]/Dusz[Dusz > 0]
		
		# calulate the inital flow
		Qusz0 = Susz*aux
				
		aux[Dusz <= 0] = 0
		
		# calculate the flow out of the unsaturated zone
		Qusz = Qusz0*(1-np.exp(-aux))
		
		aux = np.ones_like(Dusz)
		aux[v > 0] = Dusz[v > 0]/v[v > 0]
		Qusz = Qusz*aux
		
		# Update storage of the unsaturated zone
		self.Susz0 = Susz - Qusz
		
		self.Susz0[self.Susz0 < 0] = 0
		
		return Qusz