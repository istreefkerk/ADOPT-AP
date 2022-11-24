# -*- coding: utf-8 -*-
"""
DRYP: Dryland WAter Partitioning Model
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DRYP.components.DRYP_io import inputfile, model_environment_status
from DRYP.components.DRYP_infiltration import infiltration
from DRYP.components.DRYP_rainfall import rainfall
from DRYP.components.DRYP_ABM_connector import ABMconnector
from DRYP.components.DRYP_routing import runoff_routing
from DRYP.components.DRYP_soil_layer import swbm
from DRYP.components.DRYP_groundwater_EFD import gwflow_EFD, storage, storage_uz_sz
from DRYP.components.DRYP_Gen_Func import (
	GlobalTimeVarPts, GlobalTimeVarAvg, GlobalGridVar,
	save_map_to_rastergrid, check_mass_balance)
#from datetime import datetime, timedelta

# Structure and model components ---------------------------------------
# data_in:	Input variables
# env_state:Model state and fluxes
# rf:		Precipitation
# abc:		Anthropic boundary conditions
# inf:		Infiltration
# swbm:	Soil water balance
# ro:		Routing - Flow accumulator
# gw:		Groundwater flow

class DRYP_Model(object):

    def __init__(self, filename_input):
      
    #initializing model, based on DRYP.io (inputfile + model_environment_status)
    
        self.data_in = inputfile(filename_input, self.config)
        self.daily = 1
        
        # setting model fluxes and state variables
        self.env_state = model_environment_status(self.data_in)
        self.env_state.set_output_dir(self.data_in)
        self.env_state.points_output(self.data_in)
        
        # setting model components
        self.rf = rainfall(self.data_in, self.env_state)
        self.abc = ABMconnector(self.data_in, self.env_state)
        self.inf = infiltration(self.env_state, self.data_in)
        self.swb = swbm(self.env_state, self.data_in)
        self.swb_rip = swbm(self.env_state, self.data_in)
        self.ro = runoff_routing(self.env_state, self.data_in)
        self.gw = gwflow_EFD(self.env_state, self.data_in)
        
        # Output variables and location
        self.outavg = GlobalTimeVarAvg(self.env_state.area_catch_factor)
        self.outavg_rip = GlobalTimeVarAvg(self.env_state.area_river_factor)
        self.outpts = GlobalTimeVarPts()
        self.state_var = GlobalGridVar(self.env_state, self.data_in)

        self.gw_level = []
        self.pre_mb = []
        self.exs_mb = []
        self.tls_mb = []
        self.gws_mb = []
        self.uzs_mb = []
        self.dis_mb = []
        self.rch_mb = []
        self.aet_mb = []
        self.egw_mb = []

        self.dt_GW = np.int(self.data_in.dt)

        self.t_eto = 0	
        self.t_pre = 0
        
        self.rch_agg = np.zeros(len(self.swb.L_0))
        self.etg_agg = np.zeros(len(self.swb.L_0))
        
    def step(self):
        
        # while t < t_end...
                
        for UZ_ti in range(self.data_in.dt_hourly):
            
            for dt_pre_sub in range(self.data_in.dt_sub_hourly):
                
                self.swb.run_soil_aquifer_one_step(self.env_state,
                    self.env_state.grid.at_node['topographic__elevation'],
                    self.env_state.SZgrid.at_node['water_table__elevation'],
                    self.env_state.Duz,
                    self.swb.tht_dt)
                                                            
                self.env_state.Duz = self.swb.Duz
                
                self.rf.run_rainfall_one_step(self.t_pre, self.t_eto, self.env_state, self.data_in)
                
                self.abc.run_ABM_one_step(self.agents.farmers, self.env_state,
                    self.rf.rain, self.env_state.Duz, self.swb.tht_dt, self.env_state.fc,
                    self.env_state.grid.at_node['wilting_point'],
                    self.env_state.SZgrid.at_node['water_table__elevation'],self.swb.aet_dt, self.rf.PET, self.config)

                self.rf.rain += self.abc.auz
                
                self.inf.run_infiltration_one_step(self.rf, self.env_state, self.data_in)
                
                self.aux_usz = np.sum((self.swb.L_0*self.env_state.hill_factor)[self.env_state.act_nodes])
                self.aux_usp = np.sum((self.swb_rip.L_0*self.env_state.riv_factor)[self.env_state.act_nodes])

                self.swb.run_swbm_one_step(self.inf.inf_dt, self.rf.PET, self.abc.kc,
                    self.env_state.grid.at_node['Ksat_soil'], self.env_state, self.data_in)

                self.env_state.grid.at_node['riv_sat_deficit'][:] *= (self.swb.tht_dt)
                
                self.ro.run_runoff_one_step(self.inf, self.swb, self.abc.aof, self.env_state, self.data_in)
                
                self.tls_aux = self.ro.tls_flow_dt*self.env_state.rip_factor
				
                self.rip_inf_dt = self.inf.inf_dt + self.tls_aux
                                
                self.swb_rip.run_swbm_one_step(self.rip_inf_dt, self.rf.PET, self.abc.kc,
                        self.env_state.grid.at_node['Ksat_ch'], self.env_state,
                        self.data_in,self.env_state.river_ids_nodes)
                
                self.swb_rip.pcl_dt *= self.env_state.riv_factor
                self.swb_rip.aet_dt *= self.env_state.riv_factor
                self.swb.pcl_dt *= self.env_state.hill_factor
                self.swb.aet_dt *= self.env_state.hill_factor
                self.rech = self.swb.pcl_dt + self.swb_rip.pcl_dt - self.abc.asz# [mm/dt]
                self.etg_dt = self.gw.SZ_potential_ET(self.env_state, self.swb.gwe_dt)
                self.etg_agg += np.array(self.etg_dt) # [mm/h]
                self.rch_agg += np.array(self.rech) # [mm/dt]

                # Water balance storage and flow

                self.pre_mb.append(np.sum(self.rf.rain[self.env_state.act_nodes]))
                self.exs_mb.append(np.sum(self.inf.exs_dt[self.env_state.act_nodes]))
                self.tls_mb.append(np.sum(self.tls_aux[self.env_state.act_nodes]))
                self.aux_usz1 = np.sum((self.swb.L_0*self.env_state.hill_factor)[self.env_state.act_nodes])
                self.aux_usp1 = np.sum((self.swb_rip.L_0*self.env_state.riv_factor)[self.env_state.act_nodes])
                
                self.uzs_mb.append(self.aux_usp1+self.aux_usz1-self.aux_usp-self.aux_usz)
                self.aet_mb.append(np.sum((self.swb_rip.aet_dt+self.swb.aet_dt)[self.env_state.act_nodes]))
                self.egw_mb.append(np.sum(self.etg_dt[self.env_state.act_nodes]))
                self.rch_mb.append(np.sum(self.rech[self.env_state.act_nodes]))				
                #aux_ssz = storage_uz_sz(env_state, 0, 0)

                if self.dt_GW == self.data_in.dtSZ:
                    # Change units to m/h
                    self.env_state.SZgrid.at_node['discharge'][:] = 0.0
                    self.env_state.SZgrid.at_node['recharge'][:] = (self.rch_agg - self.etg_agg)*0.001 #[mm/dt]
                    self.gw.run_one_step_gw(self.env_state, self.data_in.dtSZ/60, self.swb.tht_dt,
                        self.env_state.Droot*0.001)
                    self.rch_agg = np.zeros(len(self.swb.L_0))
                    self.etg_agg = np.zeros(len(self.swb.L_0))
                    self.dt_GW = 0

                self.dt_GW += np.int(self.data_in.dt)	

                self.gws_mb.append(storage_uz_sz(self.env_state, np.array(self.swb.tht_dt), self.gw.dh))			
                self.dis_mb.append(np.sum(self.env_state.SZgrid.at_node['discharge'][self.env_state.act_nodes])-self.gw.flux_out)
                
                #Extract average state and fluxes				
                self.outavg.extract_avg_var_pre(self.env_state.basin_nodes,self.rf)				
                self.outavg.extract_avg_var_UZ_inf(self.env_state.basin_nodes,self.inf)
                self.outavg.extract_avg_var_UZ_swb(self.env_state.basin_nodes,self.swb)
                self.outavg_rip.extract_avg_var_UZ_swb(self.env_state.basin_nodes,self.swb_rip)
                self.outavg.extract_avg_var_OF(self.env_state.basin_nodes,self.ro)
                self.outavg.extract_avg_var_SZ(self.env_state.basin_nodes,self.gw)
                
                #Extract point state and fluxes
                self.outpts.extract_point_var_UZ_inf(self.env_state.gaugeidUZ,self.inf)
                self.outpts.extract_point_var_UZ_swb(self.env_state.gaugeidUZ,self.swb)
                self.outpts.extract_point_var_OF(self.env_state.gaugeidOF,self.ro)
                self.outpts.extract_point_var_SZ(self.env_state.gaugeidGW,self.gw)
                self.state_var.get_env_state(self.t_pre, self.rf, self.inf, self.swb, self.ro, self.gw, self.swb_rip, self.env_state)
                
                self.env_state.L_0 = np.array(self.swb.L_0)		
	
                self.t_pre += 1
            self.t_eto += 1

    def save_output(self):

        mb = [self.pre_mb, self.exs_mb, self.tls_mb, self.rch_mb, self.gws_mb,
        self.uzs_mb, self.dis_mb, self.aet_mb, self.egw_mb]
                    
        self.outavg.save_avg_var(self.env_state.fnameTS_avg+'.csv', self.rf.date_sim_dt)
        self.outavg_rip.save_avg_var(self.env_state.fnameTS_avg+'rip.csv', self.rf.date_sim_dt)	
        self.outpts.save_point_var(self.env_state.fnameTS_OF, self.rf.date_sim_dt,
            self.ro.carea[self.env_state.gaugeidOF],
            self.env_state.rarea[self.env_state.gaugeidOF])	
        self.state_var.save_netCDF_var(self.env_state.fnameTS_avg+'.nc')
        check_mass_balance(self.env_state.fnameTS_avg, self.outavg, self.outpts,
            self.outavg_rip, mb, self.rf.date_sim_dt,
            self.ro.carea[self.env_state.gaugeidOF[0]])
        
        # Save water table for initial conditions
        fname_out = self.env_state.fnameTS_avg + '_wte_ini.asc'	
        save_map_to_rastergrid(self.env_state.SZgrid, 'water_table__elevation', fname_out)