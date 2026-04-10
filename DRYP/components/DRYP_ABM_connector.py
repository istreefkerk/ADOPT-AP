
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import utm
import pyproj
import time
# Global parameters
ABC_RIVER = 0.99 # River abstraction parameter

# seed = 22
# random.seed(seed)
# np.random.seed(seed)

#------------------------------------------------
#           INITIALIZE ABM CONNECTOR
#------------------------------------------------

class ABMconnector(object):
    def __init__(self, inputfile, env_state, agents):
        # initialize grass layer
        
        if inputfile:
            # For surface grids:
            env_state.grid.at_node['AOFT'][:] = ABC_RIVER
                         
            print('Add any additional parameter for running ABM')
            if agents.model.config['general']['ABM'] == True:
                self.initial_kc = agents.initial_kc 
            if agents.model.config['general']['ABM'] == False:
                self.initial_kc = np.full(env_state.grid_size, 1.0)

            self.kc = self.initial_kc.copy()
            self.aof = np.zeros(env_state.grid_size)
            self.auz = np.zeros(env_state.grid_size)
            self.asz = np.zeros(env_state.grid_size)
            self.data_provided = 1

            self.width = 201  #this information can be found in input data of DRYP
            self.height = 174 
            self.grid = self.width * self.height
            self.x_grid = 925.145414825871
            self.y_grid = 925.145414825871 
            self.I_capacity = np.zeros(self.grid)

            self.crop_yield_factors = self.crop_yield_factors()
            self.max_yield_maize = 2.4
            self.max_yield_casava = 4
            self.max_yield_grass = 3.5 #*1000kg /hectare

            # index related to crops (in excel sheet)
            self.grass_nr = 24
            self.maize_nr = 1
            self.wheat_nr = 0

            self.length_crop_season = agents.harvest_date - agents.planting_date # length of crop season (harvest - start of season)
            self.length_milk_season = 150 # length of 'milk' season (half year)

            self.crop_data = self.get_crop_data()

            self.crop_stage_data = np.zeros((26, 4), dtype=np.float32)
            self.crop_stage_data[:, 0] = self.crop_data['L_ini']
            self.crop_stage_data[:, 1] = self.crop_data['L_dev']
            self.crop_stage_data[:, 2] = self.crop_data['L_mid']
            self.crop_stage_data[:, 3] = self.crop_data['L_late']

            self.kc_crop_stage = np.zeros((26, 3), dtype=np.float32)
            self.kc_crop_stage[:, 0] = self.crop_data['kc_ini']
            self.kc_crop_stage[:, 1] = self.crop_data['kc_mid']
            self.kc_crop_stage[:, 2] = self.crop_data['kc_end']

            self.grid_size =self.x_grid * self.y_grid

            grid_coordinates = []

            for y in range(self.height):
                for x in range(self.width): 
                    grid_coordinates.append((x,y)) 

            self.grid_x_coordinates = self.Extract_x(grid_coordinates) 
            self.grid_y_coordinates = self.Extract_y(grid_coordinates) 

            self.distance = []
            self.groundwater_distance = []
            self.closest_groundwater_distance = []
            self.index_groundwater_distance = []

            self.abstraction_points = np.where(agents.abstraction_p == True, 1, 1000) # array with 1 where there is an abstraction point and 1000 where there is no abstraction point
            self.river_network = np.where(agents.river_network.flatten() >= 1, 1, 1000)

            for i in range(agents.n):
                distance = 1 + np.sqrt(abs(([x[1] for x in agents.coords_neighbours[i]] - agents.coordinates[i,0])**2 + ([y[0] for y in agents.coords_neighbours[i]]  - agents.coordinates[i,1])**2) )
                self.distance.append(distance)

            self.max_gw_abs_depth = 100 #[m] maximum depth of groundwater abstraction
            self.water_demand_hh = 50 #L/day/person
            self.water_demand_cow = 25 #L/day
            self.water_demand_goat = 9.6  #L/day

            self.storage_size = 5 #m3 or 5000L for water harvesting storage capacity
            self.roof_size = 50 #m2

            # empty grid for water storage communities

            self.irrigation_from_storage = np.zeros((self.height, self.width))
            self.water_from_surface = np.zeros((self.height, self.width))
            self.water_from_precipitation = np.zeros((self.height, self.width))
            self.water_storage_agri = np.zeros((self.height, self.width))
            self.storage = np.zeros((self.height, self.width))
            self.storage_rain = np.zeros((self.height, self.width))
            self.livestock_from_storage = np.zeros((self.height, self.width))
            
            self.grass_yield_ = np.zeros((self.height, self.width))

            self.grass_yield_ratio = np.zeros((env_state.grid_size))
            self.milk_yield_ratio = np.zeros((env_state.grid_size))

            self.actual_transpiration_crop = np.empty((env_state.grid_size, 177)) # half year
            self.actual_transpiration_crop[:] = np.nan
            self.potential_transpiration_crop = np.empty((env_state.grid_size, 177)) # half year
            self.potential_transpiration_crop[:] = np.NaN
            self.latest_rain = np.empty((env_state.grid_size, 30), dtype=np.float32)
            self.latest_rain[:] = np.NaN
            
            self.abstraction_gw_hh = np.zeros(env_state.grid_size)
            self.abstraction_gw_irr = np.zeros(env_state.grid_size)
            self.abstraction_gw_liv = np.zeros(env_state.grid_size)
            self.abstraction_riv_hh = np.zeros(env_state.grid_size)
            self.abstraction_riv_irr = np.zeros(env_state.grid_size)
            self.abstraction_riv_liv = np.zeros(env_state.grid_size)
            
            self.distance_hh_water = np.zeros((agents.n))
            self.grid_adapt_measure_3 = np.zeros((self.height, self.width)) # irrigation grid

            self.drought_perception = np.nan
            self.soil_moisture_deficit = np.zeros(env_state.grid_size)
            self.yield_crops = agents.yield_crops.copy()

        else:
            self.data_provided = 0
            print('Not available water extractions file')
        self.netcdf_file = int(inputfile.netcf_ABC)

    def run_ABM_one_step(self, agents, env_state, rain, Droot, theta, theta_fc, wte, aet, pet, config, ro, inf_args):
        """
        Call this to execute a step in the model.
                
        Parameters:
            i_ABM:		it is counter in case it is needed
            env_state:	all state parameters of the model
            rain:		to check if irrigation is needed (mm)
            Duz:		Soil depth (mm)
            Droot:      rooting depth
            theta:		water content (-)
            theta_fc:	water at field capacity (-)
            theta_wp:	water at wilting point (-)
            wte:        water table elevation
            aet:        actual evapotranspiration of crop
            pet:        potential evapotranspiration of crop
            ->
            Kc: crop factor (-)
        
        Outputs:
            aof:	fluw rate of stream flow abstracted [m3/dt] (*dt is model timestep)
            auz:	fluw rate of irrigation (+) or storage (-) [mm/dt]
            asz:	fluw rate of groundwater abstraction [mm/dt]
            kc:     crop factor
        """
        if config['general']['ABM'] == False: 
            self.data_provided == 0
            self.aof = np.zeros(env_state.grid_size)
            self.auz = np.zeros(env_state.grid_size)
            self.asz = np.zeros(env_state.grid_size)
            
        else:

            self.data_provided =  1

            crop_map = agents.crop_map

            self.I_capacity[env_state.act_nodes] = (env_state.Lsat[env_state.act_nodes]- env_state.L_0[env_state.act_nodes])* inf_args # infiltration capacity
            
            self.kc = self.get_crop_factor(agents.current_day_of_year, crop_map, env_state.grid_size, self.initial_kc, agents.planting_date, agents.harvest_date) 

            self.water_demand = self.abstract_water(theta, theta_fc, Droot, env_state, agents, wte, rain)
            
            self.latest_rain[:, 1:] = self.latest_rain[:, 0:-1]
            self.latest_rain[:, 0] = rain

            self.large_scale_agriculture(agents, theta, theta_fc, Droot, rain, env_state)
            
            self.water_storage(agents, rain, env_state)

            self.aof = (self.water_demand[3] + self.water_demand[4] + self.water_demand[5] + self.water_from_surface.flatten()) * self.grid_size / 1000  # abstraction and runoff to storage
            self.auz = self.water_demand[1] + self.water_demand[4] + self.irrigation_from_storage.flatten() - self.water_from_precipitation.flatten() # irrigation from storage, but precipitation captured by storage
            self.asz = self.water_demand[0] + self.water_demand[1] + self.water_demand[2] # abstraction

            agents.aof = self.aof
            agents.auz = self.auz
            agents.asz = self.asz
            

            self.actual_transpiration_crop[:, 1:] = self.actual_transpiration_crop[:, 0:-1]
            self.actual_transpiration_crop[:, 0] = aet #self.swb.aet_dt

            self.potential_transpiration_crop[:, 1:] = self.potential_transpiration_crop[:, 0:-1]
            self.potential_transpiration_crop[:, 0] = pet #self.rf.PET

            self.grass_yield_ = self.grass_yield(agents) # daily calcalation


#------------------------------------------------
#                 CROP MODULE- KC
#------------------------------------------------

    def interpolate_kc(self, stage_start, stage_end, crop_progress, stage_start_kc, stage_end_kc):
        """
        Interpolate kc for stage 1 and 3 (developing and late stage)
        
        stage start = day number at start of that stage
        stage_start_kc = kc at start of that stage

        Return kc at that current crop&stage progress
        """

        stage_progress = (crop_progress - stage_start) / (stage_end - stage_start)
        return (stage_end_kc - stage_start_kc) * stage_progress + stage_start_kc

    def get_crop_kc(self, crop_map, crop_age_days, crop_harvest_day, crop_stage_data, kc_crop_stage, ini_land_use):
        """
        Return kc for entire grid based on growing stage and land use.
        """
        
        kc = np.full(crop_map.size, ini_land_use, dtype=np.float32)

        for i in range(crop_map.size):
            crop = crop_map[i]
            if crop != -1: # if crop map is not -1, then there is a crop
                age_days = crop_age_days[i]
                harvest_day = crop_harvest_day[i]
                crop_progress = age_days / harvest_day
                #if crop_progess > 1: set crop to -1 (and communicate harvest to ABM?)
                stage = np.searchsorted(crop_stage_data[crop], crop_progress, side='left')
                if stage == 0:
                    field_kc = kc_crop_stage[crop, 0]
                elif stage == 1:
                    field_kc = self.interpolate_kc(
                        stage_start=crop_stage_data[crop, 0],
                        stage_end=crop_stage_data[crop, 1],
                        crop_progress=crop_progress,
                        stage_start_kc=kc_crop_stage[crop, 0],
                        stage_end_kc=kc_crop_stage[crop, 1]
                    )
                elif stage == 2:
                    field_kc = kc_crop_stage[crop, 1]
                elif stage == 3:
                    field_kc = self.interpolate_kc(
                        stage_start=crop_stage_data[crop, 2],
                        stage_end=1,
                        crop_progress=crop_progress,
                        stage_start_kc=kc_crop_stage[crop, 1],
                        stage_end_kc=kc_crop_stage[crop, 2]
                    )
                else:
                    assert stage == 4
                    field_kc = 1.0  # pasture land if there is no crop
            elif crop == -1:
                field_kc = ini_land_use[i] # other land if there is no crop, should specify this further based on land use
            
            kc[i] = field_kc
        return kc

    def get_crop_data(self):
        """
        https://doi.org/10.1016/j.jhydrol.2009.07.031
        """
        df = pd.read_csv(os.path.join('DataDrive', 'crop_factors.csv'), delimiter = ';')
        df['L_ini'] = df['L_ini']
        df['L_dev'] = df['L_ini'] + df['L_dev']
        df['L_mid'] = df['L_dev'] + df['L_mid']
        df['L_late'] = df['L_mid'] + df['L_late']
        assert np.allclose(df['L_late'], 1.0)
        return df

    def get_crop_factor(self, current_day_of_year, crop_map, grid, ini_land_use, planting_date, harvest_date):
        """
        Calculate kc based on crop data and day of the year.
        """
        
        # For now only Maize, with same planting date
        
        self.start_day = np.full(grid, planting_date) # start_day 
        
        self.crop_age_days = current_day_of_year - self.start_day
        self.crop_harvest_day = np.full(grid, harvest_date) # harvest_day

        if self.start_day[0] <= current_day_of_year: # large scale agriculture
            self.cropKC = self.get_crop_kc(crop_map, self.crop_age_days, self.crop_harvest_day, self.crop_stage_data, self.kc_crop_stage, ini_land_use) # Only for agents locations!

        else:
            self.cropKC = ini_land_use 

        return self.cropKC

#------------------------------------------------
#                CROP YIELD MODULE
#------------------------------------------------

    def crop_yield_factors(self):
        """
        Source of yield ratio data: https://doi.org/10.1016/j.jhydrol.2009.07.031
        """
        df = pd.read_csv(os.path.join('DataDrive','yield_ratios.csv'), delimiter = ';')
        return df[['alpha', 'beta', 'P0', 'P1']].to_dict(orient='list')

    def _get_yield_ratio(self, crop_map, evap_ratios, alpha, beta, P0, P1):
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031""" # monfreda dataset
        yield_ratios = np.empty(evap_ratios.size, dtype=np.float32)
            
        for i in range(evap_ratios.size): 
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            if alpha[crop] * evap_ratio + beta[crop] > 1:
                yield_ratio = 1
            elif P0[crop] < evap_ratio < P1[crop]:
                yield_ratio = alpha[crop] * P1[crop] + beta[crop] - (P1[crop] - evap_ratio) * (alpha[crop] * P1[crop] + beta[crop]) / (P1[crop] - P0[crop])
            elif evap_ratio < P0[crop]:
                yield_ratio = 0
            else:
                yield_ratio = alpha[crop] * evap_ratio + beta[crop]
            yield_ratios[i] = yield_ratio
            
        return yield_ratios

    def get_yield_ratio(self, actual_transpiration, potential_transpiration, crop_map):
        
        
        return self._get_yield_ratio(
            crop_map,
            np.where(potential_transpiration == 0, 0, actual_transpiration /potential_transpiration),
            self.crop_yield_factors['alpha'],
            self.crop_yield_factors['beta'],
            self.crop_yield_factors['P0'],
            self.crop_yield_factors['P1'],
        )

    def get_yield(self, actual_transpiration, potential_transpiration, crop_map):
    
        yield_ratio = np.maximum(np.minimum(self.get_yield_ratio(actual_transpiration, potential_transpiration, crop_map), 1),0)
        yield_irr_maize = (yield_ratio*0+1) * self.max_yield_maize
        yield_irr_casava = (yield_ratio*0+1) * self.max_yield_casava
        yield_actual_maize = yield_ratio * yield_irr_maize
        yield_actual_casava = yield_ratio * yield_irr_casava
        
        return yield_actual_maize, yield_irr_maize, yield_actual_casava, yield_irr_casava #, yield_grass

#------------------------------------------------
#                CROP PRODUCTION
#------------------------------------------------

    def Crop_production(self, crop_map, type_of_crop, coordinates, n):
        ''' 
        Return the the production of crops [*1000 kg], based on crop yields [kg / hectare], land size [hectare] , and rainfed/irrigated [0/1]
        '''
        average_aet = np.where(np.nanmean(self.actual_transpiration_crop[:, 0:self.length_crop_season], axis = 1) > 0, np.nanmean(self.actual_transpiration_crop[:, 0:self.length_crop_season], axis = 1), 0)
        average_pet = np.where(np.nanmean(self.potential_transpiration_crop[:, 0:self.length_crop_season], axis = 1) > 0, np.nanmean(self.potential_transpiration_crop[:, 0:self.length_crop_season], axis = 1), 0)

        self.yield_crops = self.get_yield(average_aet, average_pet, crop_map)
        
        crop_produce = np.zeros(n)
        # land_size [hectare]
        # yield_crops = [*1000 kg / hectare]

        for i in range (n):
            if type_of_crop[i] == 0: #rainfed maize
                crop_produce[i] = self.yield_crops[0].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* 1000 # kg/hectare
            elif type_of_crop[i] == 1: #rainfed casava
                crop_produce[i] = self.yield_crops[2].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* 1000 # kg/hectare

        assert (crop_produce >= 0).all()
        assert (crop_produce != np.nan).all()
        return crop_produce
    
#------------------------------------------------
#               LIVESTOCK AND GRASS
#------------------------------------------------

    def grass_yield(self, agents):
        '''
        Grass yield per grid cell
        Based on doi.org/10.1002/2015WR017841 + Pande & Savenije
        '''

        # calculate grass yield, per half year
        yield_max = self.max_yield_grass #*1000kg /hectare, ICPAC forage hazard watch

        average_aet = np.where(np.nanmean(self.actual_transpiration_crop[:, 0:self.length_crop_season], axis = 1) > 0, np.nanmean(self.actual_transpiration_crop[:, 0:self.length_crop_season], axis = 1), 0)
        average_pet = np.where(np.nanmean(self.potential_transpiration_crop[:, 0:self.length_crop_season], axis = 1) > 0, np.nanmean(self.potential_transpiration_crop[:, 0:self.length_crop_season], axis = 1), 0)

        average_aet_milk = np.where(np.nanmean(self.actual_transpiration_crop[:, 0:self.length_milk_season], axis = 1) > 0, np.nanmean(self.actual_transpiration_crop[:, 0:self.length_milk_season], axis = 1), 0)
        average_pet_milk = np.where(np.nanmean(self.potential_transpiration_crop[:, 0:self.length_milk_season], axis = 1) > 0, np.nanmean(self.potential_transpiration_crop[:, 0:self.length_milk_season], axis = 1), 0)

        fodder_grass = np.full(self.grid, self.grass_nr)

        self.grid_adapt_measure_3[agents.coordinates[:,1],agents.coordinates[:,0]] = agents.adapt_measure_3.copy() # where people irrigate

        self.grass_yield_ratio = np.maximum(np.minimum(self.get_yield_ratio(average_aet, average_pet, fodder_grass), 1),0) 
        
        self.milk_yield_ratio = np.maximum(np.minimum(self.get_yield_ratio(average_aet_milk, average_pet_milk, fodder_grass), 1),0)

        yield_grass =  np.where(self.grid_adapt_measure_3.flatten() != 1, self.grass_yield_ratio  * yield_max, self.grass_yield_ratio * yield_max * 0.1) #where commercial farms and irrigated crop farmers are -> yield 10%
        
        assert (yield_grass != np.nan).all()
        assert (yield_grass >= 0).all()

        return yield_grass.reshape(self.height, self.width)


    def Livestock_production(self, grass_yield, sum_livestock, nr_livestock, feed_required, feed_residue, net_birth_rate, weight_gain_rate): # yearly timestep, should differiate between types of livestock
        ''' 
        Livestock production as a function of grass_yield and animal specific characteristics. 
        There is a first order preference for grass availability, implemented in migration decisions in agents.py.
        Based on doi.org/10.1002/2015WR017841
        '''

        grass_availability = np.maximum(grass_yield *1000 *(np.where(sum_livestock > 0, nr_livestock /sum_livestock, 1)),0) #- (grass_consumed * self.agent_population),0) # *1000 to convert to kg from grid to every farmer, ratio to # agent population/household size.. (minimum 1)

        carrying_capacity = grass_availability / (feed_required * (1 - feed_residue)) # grass availability to cows.. * nr cows??
        rate_of_growth = net_birth_rate + np.minimum(np.maximum((weight_gain_rate * (grass_availability / ((1 - feed_residue) * np.maximum(nr_livestock, 1)))), 0), 2) # CHECK with FGD -> max number of babies born per year
        livestock_produce = np.maximum(np.maximum(nr_livestock, 1) + (rate_of_growth * (1 - (np.where(carrying_capacity >0, (nr_livestock/carrying_capacity),0)) * nr_livestock)) , 1) #per agent

        assert (livestock_produce >= 0).all()
        assert (livestock_produce != np.nan).all()
        return livestock_produce


#------------------------------------------------
#                   WATER DEMAND 
#------------------------------------------------

    def Drought_perception(self, rain, agents):
        '''Returns whether an agent has experience drought in the last 30 days (1 months), yes (1) or no (0)'''

        self.drought_threshold = agents.model.data.drought_threshold_pre.sel(dayofyear=agents.current_day_of_year)['pre'].values

        drought = np.where(rain.reshape(self.height, self.width) - self.drought_threshold <= 0.0 , 1, 0)[agents.coordinates[:,1],agents.coordinates[:,0]]

        return drought

    # DISTANCE MATRIX FOR ABSTRACTION
    def Extract_x(self, lst):
        return [item[0] for item in lst]

    def Extract_y(self, lst):
        return [item[1] for item in lst]

    def distance_river(self, agents, wte, env_state):
        ''' 
        Calculation distance to water source (river and groundwater abstraction point).
        Return the distance (number of cells) and the coordinates of nearest water sources that has water. 
        '''
        
        groundwater_level = np.where((agents.elevation_grid.flatten() - wte) < self.max_gw_abs_depth, self.abstraction_points, 1000).reshape(self.height,self.width) 
                
        #set value of river_network to 1 if river_network >=1 and dis_dt > 0 else set it to 1000
        river_network = np.where((( env_state.grid.at_node['Q_ini'] + env_state.grid.at_node['runoff']) *1000.0/env_state.area_cells ) * ABC_RIVER > 0, self.river_network, 1000).reshape(self.height,self.width)

        river_distance = []
        groundwater_distance = []

        for i in range(agents.n):
            # find river points in the grid
            river_distance_ = river_network[tuple(np.array(agents.coords_neighbours[i]).T)] * self.distance[i]
            river_distance.append(river_distance_)

            groundwater_distance_ = groundwater_level[tuple(np.array(agents.coords_neighbours[i]).T)] * self.distance[i]
            groundwater_distance.append(groundwater_distance_)


        closest_river_distance = np.min(river_distance, axis =1)
        index_river_distance = np.argmin(river_distance, axis =1)

        self.closest_groundwater_distance = np.min(groundwater_distance, axis =1)
        self.index_groundwater_distance = np.argmin(groundwater_distance, axis =1)

        coordinates_river = []
        coordinates_gw = []

        for i in range(agents.n):
            coordinates_river.append(agents.coords_neighbours[i][index_river_distance[i]])
            coordinates_gw.append(agents.coords_neighbours[i][self.index_groundwater_distance[i]])

        # return for every agent their abstraction grid location, use that in the abstraction function
        return  closest_river_distance, coordinates_river , self.closest_groundwater_distance , coordinates_gw

    def water_household(self, agents):
        ''' 
        Water consumption of household [mm]. Assume 50 L/day for rural areas mm*gridsize [m2] mm = L/m2
        ''' 
        water_household = self.water_demand_hh / self.grid_size  * agents.agent_population
        return water_household

    def water_livestock(self,agents):
        ''' 
        Water consumption of livestock [mm]. From Steinfeld et al., 2006. Method based on Wada et al.
        ''' 
        water_livestock = np.where(agents.agent_population == 0.0, 0.0, (self.water_demand_cow * agents.nr_livestock[:,0] + self.water_demand_goat * agents.nr_livestock[:,1]) / agents.grid_size * agents.agent_population) / np.where(agents.HH_size < 1.0, 1.0, agents.HH_size)# L/m2 = mm L/1000 per livestock type, based on 25degrees Celsius. 25 L/day cow, 9.6 L/day goat
        return water_livestock

    def water_storage(self, agents, rain, env_state):
        '''Add precipitation and runoff water to storage (with maximum of storage capacity), and use for irrigation and livestock'''

        storage_capacity = np.zeros((self.height, self.width))
        storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]] = np.where(agents.adapt_measure_4 == 1, self.storage_size / self.grid_size * agents.agent_population * 1000, 0) #[mm] # to do # 5000L = 5 m3 ADDED  /1000

        #water demand in mm -> abstraction in m3

        #1: catch water 

        #can catch from 50 m2/gridsize [m2] roof = (proportion of gridcell)
        self.water_from_precipitation = np.maximum(np.minimum(np.minimum(self.roof_size/self.grid_size * rain.reshape(self.height, self.width) * agents.density_grid, rain.reshape(self.height, self.width)), storage_capacity - self.storage - self.storage_rain), 0) # [mm]
        
        self.storage_rain += self.water_from_precipitation

        #HH can only use rainwater

        household_shortage = np.maximum(self.hh_demand - (self.water_demand[0] + self.water_demand[3]).reshape(self.height, self.width)[agents.coordinates[:,1],agents.coordinates[:,0]], 0) # [mm]
        water_from_storage_household = np.minimum(np.minimum(household_shortage,  self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]]), storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]]) # maximum of storage (capacity) [mm]

        self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]] -= water_from_storage_household # current status of storage, after abstraction

        #river water that can still be harvested after abstracted water #household, livestock and irrigation
        self.water_from_surface = np.minimum(np.maximum((((( env_state.grid.at_node['Q_ini'] + env_state.grid.at_node['runoff']) *1000.0/env_state.area_cells ) * ABC_RIVER ) - self.water_demand[3] - self.water_demand[4] - self.water_demand[5]).reshape(self.height, self.width), 0), storage_capacity - self.storage- self.storage_rain) # [mm]
        
        self.storage += self.water_from_surface

        # 2: calculate water needs and fill if possible with tankwater [m3]

        livestock_shortage = np.maximum(self.liv_demand - (self.water_demand[2] + self.water_demand[5]).reshape(self.height, self.width)[agents.livestock_coords[:,1],agents.livestock_coords[:,0]] , 0) # mm, calculated at livestock coordinates water demand
        irrigation_shortage = np.maximum((self.irrigation_demand.reshape(self.height, self.width)[agents.coordinates[:,1], agents.coordinates[:,0]] / self.grid_size * 1000 * agents.land_size * agents.agent_population/ 10 ) - (self.water_demand[1] + self.water_demand[4]).reshape(self.height, self.width)[agents.coordinates[:,1],agents.coordinates[:,0]], 0) # [mm]
        household_shortage = np.maximum(self.hh_demand - water_from_storage_household - (self.water_demand[0] + self.water_demand[3]).reshape(self.height, self.width)[agents.coordinates[:,1],agents.coordinates[:,0]], 0) # [mm]
        
        # 3: Abstract from water tank

        water_from_storage_rain_livestock = np.minimum(np.minimum(livestock_shortage, self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]]), (storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]] - self.storage[agents.coordinates[:,1],agents.coordinates[:,0]])) # maximum of storage (capacity) [mm]
        water_from_storage_livestock = np.minimum(np.minimum(np.maximum(livestock_shortage - water_from_storage_rain_livestock, 0), self.storage[agents.coordinates[:,1],agents.coordinates[:,0]]), (storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]] - self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]])) # maximum of storage (capacity) [mm]
        self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]] -= water_from_storage_rain_livestock # current status of storage, after abstraction
        self.storage[agents.coordinates[:,1],agents.coordinates[:,0]] -= water_from_storage_livestock # current status of storage, after abstraction

        self.livestock_from_storage[agents.coordinates[:,1],agents.coordinates[:,0]] = water_from_storage_livestock + water_from_storage_rain_livestock

        water_from_storage_rain_irrigation = np.minimum(np.minimum(irrigation_shortage, self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]]), (storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]] - self.storage[agents.coordinates[:,1],agents.coordinates[:,0]])) 
        water_from_storage_irrigation = np.minimum(np.minimum(np.maximum(irrigation_shortage - water_from_storage_rain_irrigation, 0),  self.storage[agents.coordinates[:,1],agents.coordinates[:,0]]), (storage_capacity[agents.coordinates[:,1],agents.coordinates[:,0]] - self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]])) # maximum of storage (capacity) [mm]
        self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]] -= water_from_storage_rain_irrigation
        self.storage[agents.coordinates[:,1],agents.coordinates[:,0]] -= water_from_storage_irrigation # current status of storage, after abstraction

        self.irrigation_from_storage[agents.coordinates[:,1],agents.coordinates[:,0]] = water_from_storage_irrigation + water_from_storage_rain_irrigation

        return self.irrigation_from_storage, self.water_from_surface, self.water_from_precipitation, self.livestock_from_storage # communicate to abstract water
    
    def large_scale_agriculture(self):
        pass
    
    def industry(self):
        pass

    def abstract_water(self, theta, theta_fc, Droot, env_state, agents, wte, rain):
        ''' 
        Calculation water abstracted from nearest source per type of usage (household, irrigation, livestock).
        Return abstraction on grid for river and groundwater use per type of usage.
        '''

        distance_river = self.distance_river(agents, wte, env_state)
        
        dtheta = (theta_fc-theta)
        dtheta[dtheta < 0] = 0
        SMD = np.maximum( Droot*dtheta - rain, 0) # soil moisture demand
        self.soil_moisture_deficit = SMD
        self.irrigation_demand = np.maximum(0, SMD.reshape(self.height,self.width) * agents.irrigation_demand_factor)  # Soil moisure deficit on entire grid [mm]
        self.hh_demand = self.water_household(agents)# mm per household 
        self.liv_demand = self.water_livestock(agents) # mm per household 
        gw_hh = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        gw_irr = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        gw_liv = np.zeros(env_state.grid_size).reshape(self.height, self.width) # to from to coordinates of livestock positions
        riv_hh = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        riv_irr = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        riv_liv = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        discharge = ((( env_state.grid.at_node['Q_ini'] + env_state.grid.at_node['runoff']) *1000.0/env_state.area_cells ) * ABC_RIVER ).reshape(self.height, self.width) #)*  .reshape(self.height, self.width) #mm

        coords_river = tuple(np.array(distance_river[1]).T)
        coords_groundwater = tuple(np.array(distance_river[3]).T)


        self.water_available = np.where(self.closest_groundwater_distance >= distance_river[0], discharge[coords_river], 0.6) #communicated to agents

        # 1)
        gw_hh[coords_groundwater] = np.where((self.closest_groundwater_distance < distance_river[0]), self.hh_demand, 0)

        # 2)
        gw_liv[agents.livestock_coords[:,1],agents.livestock_coords[:,0]] = np.where( (self.closest_groundwater_distance < distance_river[0]), np.minimum(self.liv_demand, 0.6 - self.hh_demand), 0)

        # 3)
        gw_irr[coords_groundwater] = np.where(((self.closest_groundwater_distance < distance_river[0]) & (agents.adapt_measure_3 == 1) & (agents.current_day_of_year >= agents.planting_date) & (agents.current_day_of_year <= agents.harvest_date)), 
                                np.maximum(0, (np.minimum(self.irrigation_demand[coords_groundwater], 0.6) / self.grid_size * 1000 * agents.land_size * agents.agent_population/ 10)), 0) # max 5 mm,  mm * [1000 m * 10 m]ha * pop / 10  = m3 -> m3 /grid [m2] = m -> m/1000 = [mm]
        

        riv_hh[coords_river] = np.where((self.closest_groundwater_distance >= distance_river[0]), np.minimum(discharge[coords_river], self.hh_demand), 0)

        riv_liv[agents.livestock_coords[:,1],agents.livestock_coords[:,0]] = np.where((self.closest_groundwater_distance >= distance_river[0]) , np.minimum(np.maximum(0, discharge[coords_river] - self.hh_demand), self.liv_demand), 0)
        
        riv_irr[coords_river] = np.where(((self.closest_groundwater_distance >= distance_river[0]) ) & (agents.adapt_measure_3 == 1) & (agents.current_day_of_year >= agents.planting_date) & (agents.current_day_of_year <= agents.harvest_date), 
                                np.maximum(0, (np.minimum(self.irrigation_demand[coords_river], (discharge[coords_river] - riv_hh[coords_river] - riv_liv[coords_river])) / self.grid_size * 1000 * agents.land_size * agents.agent_population/ 10)), 0)



        self.distance_hh_water = np.where((self.storage_rain[agents.coordinates[:,1],agents.coordinates[:,0]]).flatten() > self.hh_demand, 0, np.where((self.closest_groundwater_distance < distance_river[0]), self.closest_groundwater_distance, distance_river[0]))

        #Communicate water abstractions to agents for plotting
        agents.abstraction_gw_hh = gw_hh.flatten()
        agents.abstraction_gw_irr = gw_irr.flatten()
        agents.abstraction_gw_liv = gw_liv.flatten()
        agents.abstraction_riv_hh = riv_hh.flatten()
        agents.abstraction_riv_irr = riv_irr.flatten()
        agents.abstraction_riv_liv = riv_liv.flatten()

        return gw_hh.flatten(), gw_irr.flatten(), gw_liv.flatten(), riv_hh.flatten(), riv_irr.flatten(), riv_liv.flatten()
