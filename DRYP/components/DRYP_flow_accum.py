"""Flow accumulator DRYP
"""
import numpy as np
#from numba import jit
#from numba import guvectorize
#import pyximport; pyximport.install()
#from components.TransLoss import find_discharge_and_losses # for windows
#from TransLoss import find_discharge_and_losses # for linux
from landlab.components.flow_accum import flow_accum_bw#, make_ordered_node_array
from landlab.core.utils import as_id_array
from landlab import RasterModelGrid
from landlab.components import FlowDirectorD8
#from landlab.components.flow_director.flow_director_to_one import(
#	_FlowDirectorToOne)
#from landlab.components.flow_director.flow_director_to_one import(
#	_FlowDirectorToOne)

#Kloss = None
class runoff_routing(object):
	"""Function to discharge and transmission losses discharge.
	Running run_one_step() results in the following to occur:
		1. Flow directions are updated (unless update_flow_director is set
		as False).
		2. Intermediate steps that analyse the drainage network topology
		and create datastructures for efficient drainage area and discharge
		calculations.
		3. Calculation of discharge and transmission losses.
	"""
	def __init__(self, env_state, data_in):
		#global Kloss
		#if Kloss is None:
		#self.Kloss = data_in.Kloss
		# Creates numpy arrays for passing model variables
		#env_state.grid.add_zeros('node', 'surface_water__discharge', dtype=float)
		Create_parameter_WV(env_state.grid)#, data_in.Kloss)
		self.dis_dt = np.zeros(env_state.grid_size)
		self.qfl_dt = np.zeros(env_state.grid_size)
		self.tls_dt = np.zeros(env_state.grid_size)
		self.flow_tls_dt = np.zeros(env_state.grid_size)
		self.carea = None
		
		# 1. Check if flow director is needed
		if env_state.act_update_flow_director:
			fd = FlowDirectorD8(env_state.grid, 'topographic__elevation')
			fd.run_one_step()
		
		# 2. Creates drainage networks, flowpaths and id arrays
		self.r = as_id_array(env_state.grid["node"]["flow__receiver_node"])
		#nd = as_id_array(flow_accum_bw._make_number_of_donors_array(self.r))
		#delta = as_id_array(flow_accum_bw._make_delta_array(nd))
		#D = as_id_array(flow_accum_bw._make_array_of_donors(self.r, delta))
		self.s = as_id_array(flow_accum_bw.make_ordered_node_array(self.r))
		self.carea = find_drainage_area(self.s, self.r,
					env_state.area_cells,
					env_state.grid.boundary_nodes)
		
		
	def run_runoff_one_step(self, exs, swb, aof, env_state, data_in):
		"""Function to make FlowAccumulator calculate drainage area and
		discharge.
		Running run_one_step() results in the following to occur:
			1. Flow directions are updated (unless update_flow_director is set
			as False).
			2. Intermediate steps that analyse the drainage network topology
			and create datastructures for efficient drainage area and discharge
			calculations.
			3. Calculation of drainage area and discharge.
			4. Depression finding and mapping, which updates drainage area and
			discharge.
		"""
		
		# dis_dt:	Discharge [mm]
		# exs_dt:	Infiltration excess [mm]
		# tls_dt	Transmission losses [mm]
		# cth_area:	Cell factor area, it reduces or increases the area
		# Q_ini:	Initial available water in the channel
		# aof:		River flow abstraction [mm]
		env_state.grid.at_node['AOF'][:] = aof
		act_nodes = env_state.act_nodes
		
		# Update runoff with all inputs:
		# Runoff in units [m]
		env_state.grid.at_node['runoff'][act_nodes] = (
				exs[act_nodes]*0.001*env_state.cth_area[act_nodes]
				+ (env_state.SZgrid.at_node['discharge'][act_nodes])
				)
		
		# create array with bolean values for running streamflow routing
		check_dry_conditions = len(np.where(
				(env_state.grid.at_node['runoff'][act_nodes]
				+ env_state.grid.at_node['Q_ini'][act_nodes]
				) > 0.0)[0])
		
		if check_dry_conditions > 0:
			# run this section for any runoff
			# this function return
			# discharge, QTL, Q_ini, Q_aof
			discharge, QTL, Q_ini, Qaof = find_discharge_and_losses(
				self.s, self.r,
				np.array(env_state.grid.at_node['runoff']),#Qw
				np.array(env_state.grid.at_node['SS_loss']), #Criv
				np.array(env_state.grid.at_node['decay_flow']), #Kt
				np.array(env_state.grid.at_node['Transmission_losses']), #QTL
				np.array(env_state.grid.at_node['Q_ini']), #Q_ini
				np.array(env_state.grid.at_node['river']), #riv
				np.array(env_state.grid.at_node['AOF']), #Qaof
				np.array(env_state.grid.at_node['AOFT']), #Qaoft
				np.array(env_state.grid.at_node['riv_sat_deficit']), #riv_std
				np.array(env_state.grid.at_node['par_3']), #P3
				np.array(env_state.grid.at_node['par_4']), #P4
				np.array(env_state.area_cells),
				np.array(env_state.grid.boundary_nodes)
				)
			
			# Update landlab grids of environmental states/fluxes
			env_state.grid.at_node["surface_water__discharge"][:] = discharge
			env_state.grid.at_node['Transmission_losses'][:] = QTL
			env_state.grid.at_node['Q_ini'][:] = Q_ini
			env_state.grid.at_node['AOF'][:] = Qaof
			
			#env_state.grid.at_node['AOF'][:] = aux[3] #aof
			#print(discharge[23], QTL[23], Q_ini[23])
			self.dis_dt[act_nodes] = np.array(
					env_state.grid.at_node["surface_water__discharge"][act_nodes])
			
			self.dis_dt[self.dis_dt < 0] = 0
			
		else:
			# no runoff over the entire cells
			env_state.grid.at_node['Transmission_losses'][env_state.river_ids_nodes] = 0.0				
			self.dis_dt[:] = 0				
			noflow = 0
		
		# save runoff for mass balance
		self.run_dt = np.zeros(len(exs))
		self.run_dt[act_nodes] = np.array(
				env_state.grid.at_node['runoff'][act_nodes])
		
		# get transmission losses in [mm]
		self.tls_dt = np.array(
				env_state.grid.at_node['Transmission_losses']
				*1000.0/env_state.area_cells)
		
		# get transmission losses in [m3]
		self.tls_flow_dt = np.array(
				env_state.grid.at_node['Transmission_losses'])
		
		# get channel storage in [mm]
		self.qfl_dt[act_nodes] = np.array(
				env_state.grid.at_node['Q_ini'][act_nodes]
				*1000.0/env_state.area_cells[act_nodes])
		#print(self.tls_dt[act_nodes])

def Create_parameter_WV(grid):#, kKloss):
	"""additional parameters to save computational time
	parameters:
	INPUT:
	------
	grid:	landlab grid
			Ksat_ch: saturated hydraulic conductivity [m/dt]
			decay_flow: exponential factor -> residence time
			SS_loss: transmission losses at steady state
			river_width: channel width
	Kloss:	not used
	OUTPUT:
	-------
	par_3:	grid variable tp estimate time
	par_4:	grid variable to estimate transmission losses
	"""
	# Calculate parameter p4 for transmission losses estimation
	grid.at_node['par_4'] = (2.0*grid.at_node['Ksat_ch']
			/(grid.at_node['decay_flow']*grid.at_node['river_width']))
	
	# Calculate parameter p3 for estimating time to get dry
	# conditions in the channel	
	grid.at_node['par_3'] = (grid.at_node['river_width']
							* grid.at_node['SS_loss']
							/ (grid.at_node['river_width']-
							2*grid.at_node['Ksat_ch']))
	return		


def find_drainage_area(s, r, node_cell_area=1.0, boundary_nodes=None):

	"""Calculate the drainage area and water discharge at each node, permitting
	discharge to fall (or gain) as it moves downstream according to some
	function. Note that only transmission creates loss, so water sourced
	locally within a cell is always retained. The loss on each link is recorded
	in the 'surface_water__discharge_loss' link field on the grid; ensure this
	exists before running the function.
	Parameters
	----------
	s : ndarray of int
		Ordered (downstream to upstream) array of node IDs
	r : ndarray of int
		Receiver node IDs for each node
	boundary_nodes: list, optional
		Array of boundary nodes to have discharge and drainage area set to zero.
		Default value is None.
	Returns
	-------
	tuple of ndarray
		drainage area and discharge
	Notes
	-----
	-  If node_cell_area not given, the output drainage area is equivalent
	to the number of nodes/cells draining through each point, including
	the local node itself.
	-  Give node_cell_area as a scalar when using a regular raster grid.
	-  If runoff is not given, the discharge returned will be the same as
	drainage area (i.e., drainage area times unit runoff rate).
	-  If using an unstructured Landlab grid, make sure that the input
	argument for node_cell_area is the cell area at each NODE rather than
	just at each CELL. This means you need to include entries for the
	perimeter nodes too. They can be zeros.
	
	Examples
	--------
	>>> import numpy as np
	>>> from landlab import RasterModelGrid
	>>> from landlab.components.flow_accum import (
	...     find_drainage_area_and_discharge)
	>>> r = np.array([2, 5, 2, 7, 5, 5, 6, 5, 7, 8])-1
	>>> s = np.array([4, 1, 0, 2, 5, 6, 3, 8, 7, 9])
	>>> l = np.ones(10, dtype=int)  # dummy
	>>> nodes_wo_outlet = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
	
	"""
	# Number of points
	npoint = len(s)
	
	# Initialize the drainage_area and discharge arrays. Drainage area starts
	# out as the area of the cell in question, then (unless the cell has no
	# donors) grows from there. Discharge starts out as the cell's local runoff
	# rate times the cell's surface area.
	drainage_area = np.zeros(npoint, dtype=int) + node_cell_area
	#discharge = np.zeros(npoint, dtype=int) + node_cell_area
	# note no loss occurs at a node until the water actually moves along a link
	
	# Optionally zero out drainage area and discharge at boundary nodes
	if boundary_nodes is not None:
		drainage_area[boundary_nodes] = 0
	
	# Iterate backward through the list, which means we work from upstream to
	# downstream.
	for i in range(npoint - 1, -1, -1):
		donor = s[i]
		recvr = r[donor]
		if donor != recvr:
			drainage_area[recvr] += drainage_area[donor]
			
	return drainage_area		

#@jit(nopython=True)
#numba.jit('float64(float64[:])')
def find_discharge_and_losses(s, r, runoff, Criv, Kt,
				QTL, Q_ini, riv, Q_aof, Q_aoft, riv_std, P3, P4,
				node_cell_area, boundary_nodes):
				#node_cell_area=1.0, boundary_nodes=None):

	"""Calculate the drainage area and water discharge at each node, permitting
	discharge to fall (or gain) as it moves downstream according to some
	function. Note that only transmission creates loss, so water sourced
	locally within a cell is always retained. The loss on each link is recorded
	in the 'surface_water__discharge_loss' link field on the grid; ensure this
	exists before running the function.
	Parameters
	----------
	s : ndarray of int
		Ordered (downstream to upstream) array of node IDs
	r : ndarray of int
		Receiver node IDs for each node
	l : ndarray of int
		Link to receiver node IDs for each node
	loss_function : Python function(Qw, nodeID, linkID, grid)
		Function dictating how to modify the discharge as it leaves each node.
		nodeID is the current node; linkID is the downstream link, grid is a
		ModelGrid. Returns a float.
	grid : Landlab ModelGrid (or None)
		A grid to enable spatially variable parameters to be used in the loss
		function. If no spatially resolved parameters are needed, this can be
		a dummy variable, e.g., None.
	node_cell_area : float or ndarray
		Cell surface areas for each node. If it's an array, must have same
		length as s (that is, the number of nodes).
	runoff : float or ndarray
		Local runoff rate at each cell (in water depth per time). If it's an
		array, must have same length as s (that is, the number of nodes).
	boundary_nodes: list, optional
		Array of boundary nodes to have discharge and drainage area set to zero.
		Default value is None.
	Returns
	-------
	tuple of ndarray
		drainage area and discharge
	Notes
	-----
	-  If node_cell_area not given, the output drainage area is equivalent
	to the number of nodes/cells draining through each point, including
	the local node itself.
	-  Give node_cell_area as a scalar when using a regular raster grid.
	-  If runoff is not given, the discharge returned will be the same as
	drainage area (i.e., drainage area times unit runoff rate).
	-  If using an unstructured Landlab grid, make sure that the input
	argument for node_cell_area is the cell area at each NODE rather than
	just at each CELL. This means you need to include entries for the
	perimeter nodes too. They can be zeros.
	-  Loss cannot go negative.
	Examples
	--------
	>>> import numpy as np
	>>> from landlab import RasterModelGrid
	>>> from landlab.components.flow_accum import (
	...     find_drainage_area_and_discharge)
	>>> r = np.array([2, 5, 2, 7, 5, 5, 6, 5, 7, 8])-1
	>>> s = np.array([4, 1, 0, 2, 5, 6, 3, 8, 7, 9])
	>>> l = np.ones(10, dtype=int)  # dummy
	>>> nodes_wo_outlet = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
	
	"""
	# Number of points
	npoint = len(s)
	
	# Initialize the drainage_area and discharge arrays. Drainage area starts
	# out as the area of the cell in question, then (unless the cell has no
	# donors) grows from there. Discharge starts out as the cell's local runoff
	# rate times the cell's surface area.
	#drainage_area = np.zeros(npoint, dtype=int) + node_cell_area
	discharge = node_cell_area * runoff
	# note no loss occurs at a node until the water actually moves along a link
	
	# Optionally zero out drainage area and discharge at boundary nodes
	if boundary_nodes is not None:
		discharge[boundary_nodes] = 0
	
	# Iterate backward through the list, which means we work from upstream to
	# downstream.
	for i in range(npoint - 1, -1, -1):
		donor = s[i]
		recvr = r[donor]
		if donor != recvr:
			#print('a',discharge[donor])	
			# this function calculate:
			# discharge_remaining, Q_TLp, Q_inip, Q_aofp	
			aux = TransLossWV(
					discharge[donor],
					Criv[donor],
					Kt[donor],
					QTL[donor],
					Q_ini[donor],
					riv[donor],
					Q_aof[donor],
					Q_aoft[donor],
					riv_std[donor],
					P3[donor], P4[donor]
					)
				
			discharge[recvr] += aux[0]#discharge_remaining
			QTL[donor] = aux[1]#Q_TLp
			Q_ini[donor] = aux[2]#Q_inip
			Q_aof[donor] = aux[3]#Q_aofp
			#print(aux)
	return discharge, QTL, Q_ini, Q_aof
	#return np.array([discharge, QTL, Q_ini, Q_aof])

#@jit(nopython=True)
def TransLossWV(Qw, Criv, Kt, QTL, Q_ini,
				riv, Qaof, Qaoft, riv_std, P3, P4):
	"""Transmission losses function
	Parameters
	----------
	Qw :		flow entering the cell
	Kloss :		Transmission losses factor
	Criv :		Conductivity
	Kt :		Decay parameter
	QTL :		Transimission losses
	Q_ini :		Initial flow
	riv :		river cell id
	Qaof :		Abstraction flux
	Qaoft :		Threshold abstraction
	riv_std :	River saturation deficit
	P3 :		Parameter
	P4 :		Parameter
	Returns
	-------
	Qout :		flow leaving the cell
	QTL :		Transimission losses
	Q_ini :		Initial flow
	Qaof :		Abstraction flux
	"""
	Qout = Qw	
	if riv != 0:		
		Qin = (Qw+Q_ini)
		# abstractions, it can be modified
		if Qin <= Qaof:
			Qaof = Qaoft*Qin
		Qin += -Qaof
			
		Q_ini = 0
		QTL = 0
		#print(Qin,riv_std,Q_ini,Qw,Qaoft)	
		if Qin > 0.0:		
			if riv_std <= 0.0:		
				aux = exp_decay_wp(Qin, Kt)
				Qout = aux[0]
				Qo = aux[1]
				TL = 0				
			else:
				#Con = np.array(Criv)
				aux_l = exp_decay_loss_wp(Criv, Qin, Kt, P3, P4)
				
				#print(aux_l)
				Qout = aux_l[0]
				TL = aux_l[1]
				Qo = aux_l[2]
								
				if TL > riv_std:					
					Qo += TL-riv_std
					TL = riv_std
			
			Q_ini = Qo			
			QTL = TL
		#print([Qout, QTL, Q_ini, Qin])				
	return np.array([Qout, QTL, Q_ini, Qaof])
	#return Qout, QTL, Q_ini, Qaof

#@jit(nopython=True)	
def exp_decay_loss_wp(TL, Qin, k, P3, P4):
	"""Calculate transmission losses, flow leaving the cell, and
	channel storage
	
	INPUT:
	------
	TL:		steady-state transmission losses [m2/dt]
	Qin:	Total amount of water entering the cell [m3]
	k:		decay factor - time residence [1/dt]
	par_3:	grid variable tp estimate time
	par_4:	grid variable to estimate transmission losses
	
	OUTPUT:
	-------
	Qout:	flow leaving the cells [m3/dt]
	Qtl:	transmission losses [m3/dt]
	Sch:	storage [m3]
	"""
	Qout, Qo = 0, 0
	
	# Calcualte the time to reach dry condition on
	# the stream
	t = -(1/k)*np.log(P3/(Qin*k))
	
	# test if channel dry-up during the time step
	if t > 0:	
		if t > 1:		
			# Dry conditions are not developed
			t = 1
		
		# potential amount of water leaving the cell
		Q1 = Qin*(1-np.exp(-k*t))		
		
		# transmission losses
		Qtl = Qin*k*P4*(1-np.exp(-k*t))+TL*t	
		
		# water leaving the cell
		Qout = Q1 - Qtl
	
	# test 	
	if t >= 1:
		# water stored in the channel
		Sch = Qin-Q1
	else:
		# channel get dry, no storage
		Sch = 0.0		
		Qtl = Qin - Qout
		
	return np.array([Qout, Qtl, Sch])

#@jit(nopython=True)
def exp_decay_wp(Qin, k):
	"""Calculate flow leaving the cell and channel storage
	
	INPUT:
	------
	Qin:	Total amount of water entering the cell [m3]
	k:		decay factor - time residence [1/dt]
		
	OUTPUT:
	-------
	Qout:	flow leaving the cells [m3/dt]
	Sch:	storage [m3]
	"""
	Qout = Qin*(1-np.exp(-k))
	Sch = Qin-Qout
	
	return np.array([Qout, Sch])