
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
# Global parameters
ABC_RIVER = 0.9 # River abstraction parameter

#------------------------------------------------
#           INITIALIZE ABM CONNECTOR
#------------------------------------------------

class ABMconnector(object):
    def __init__(self, inputfile, env_state):
        # initialize grass layer
        
        if inputfile:
            # For surface grids:
            # env_state.grid.add_zeros('node', 'name_variable', dtype=float)
            # e.g. maximun fraction of streamflow taken from river,
            # for now it is to avoid streams to become dry
            env_state.grid.at_node['AOFT'][:] = ABC_RIVER
            # For groundwater grids:
            # env_state.SZgrid.add_zeros('node', 'name_variable', dtype=float)
                         
            print('Add any additional parameter for running ABM')
            self.initial_kc = env_state.Kc.copy() #np.full(env_state.grid_size, 1.0)
            self.aof = np.zeros(env_state.grid_size)
            self.auz = np.zeros(env_state.grid_size)
            self.asz = np.zeros(env_state.grid_size)
            self.data_provided = 1

            self.width = 45 
            self.height = 92 
            self.x_grid = 925.145414826388 
            self.y_grid = 925.145414826388

            self.grid_size =self.x_grid * self.y_grid

            self.actual_transpiration_crop = np.empty((env_state.grid_size, 177)) # lenght of season (harvest - start of season)
            self.actual_transpiration_crop[:] = np.nan
            self.potential_transpiration_crop = np.empty((env_state.grid_size,177)) # lenght of season (harvest - start of season)
            self.potential_transpiration_crop[:] = np.NaN
            self.actual_transpiration_grass = np.empty((env_state.grid_size, 90)) # 3 months dry season
            self.actual_transpiration_grass[:] = np.nan
            self.potential_transpiration_grass = np.empty((env_state.grid_size,90)) # 3 months dry season
            self.potential_transpiration_grass[:] = np.NaN
            
            self.abstraction_gw_hh = np.zeros(env_state.grid_size)
            self.abstraction_gw_irr = np.zeros(env_state.grid_size)
            self.abstraction_gw_liv = np.zeros(env_state.grid_size)
            self.abstraction_riv_hh = np.zeros(env_state.grid_size)
            self.abstraction_riv_irr = np.zeros(env_state.grid_size)
            self.abstraction_riv_liv = np.zeros(env_state.grid_size)

        else:
            self.data_provided = 0
            print('Not available water extractions file')
        self.netcdf_file = int(inputfile.netcf_ABC)

    def run_ABM_one_step(self, agents, env_state, rain, Duz, theta, theta_fc, theta_wp, wte, aet, pet, config):
        """
        Call this to execute a step in the model.
                
        Parameters:
            i_ABM:		it is counter in case it is needed
            env_state:	all state parameters of the model
            rain:		to check if irrigation is needed (mm)
            Duz:		Soil depth (mm)
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
            auz:	fluw rate of irrigation [mm/dt]
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

            water_demand = self.abstract_water(theta, theta_fc, Duz, agents.abstraction_p, env_state, agents)

            crop_map = agents.crop_map 
            
            self.kc = self.get_crop_factor(agents.current_day_of_year, crop_map, env_state.grid_size, self.initial_kc, agents.planting_date, agents.harvest_date) # initial kc can be taken from 
            
            self.aof = (water_demand[3] + water_demand[4] + water_demand[5]) * self.grid_size /1000 # to m3 #+ agents.large_scale_agriculture() 
            self.auz = water_demand[1] + water_demand[4]
            self.asz = water_demand[0] + water_demand[1] + water_demand[2]

            if agents.current_day_of_year >= agents.start_dry_season and agents.current_day_of_year <= agents.end_dry_season: # how does this work with adaptation efficacy calculation?? -> take of previous year?
                   
                self.actual_transpiration_grass[:, 1:] = self.actual_transpiration_grass[:, 0:-1] 
                self.actual_transpiration_grass[:, 0] = aet #swb.aet_dt

                self.potential_transpiration_grass[:, 1:] = self.potential_transpiration_grass[:, 0:-1] 
                self.potential_transpiration_grass[:, 0] = pet #self.rf.PET

            if agents.current_day_of_year >= (agents.planting_date - 1) and agents.current_day_of_year <= agents.harvest_date:

                self.actual_transpiration_crop[:, 1:] = self.actual_transpiration_crop[:, 0:-1] 
                self.actual_transpiration_crop[:, 0] = aet #self.swb.aet_dt

                self.potential_transpiration_crop[:, 1:] = self.potential_transpiration_crop[:, 0:-1] 
                self.potential_transpiration_crop[:, 0] = pet #self.rf.PET


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
        
        kc = np.full(crop_map.size, np.nan, dtype=np.float32)

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
        # df['L_ini'] = df['L_ini']
        df['L_dev'] = df['L_ini'] + df['L_dev']
        df['L_mid'] = df['L_dev'] + df['L_mid']
        df['L_late'] = df['L_mid'] + df['L_late']
        assert np.allclose(df['L_late'], 1.0)
        return df

    def get_crop_factor(self, current_day_of_year, crop_map, grid, ini_land_use, planting_date, harvest_date):
        """
        Calculate kc based on crop data and day of the year.
        """

        crop_data = self.get_crop_data()
            
        self.crop_stage_data = np.zeros((26, 4), dtype=np.float32)
        self.crop_stage_data[:, 0] = crop_data['L_ini']
        self.crop_stage_data[:, 1] = crop_data['L_dev']
        self.crop_stage_data[:, 2] = crop_data['L_mid']
        self.crop_stage_data[:, 3] = crop_data['L_late']

        self.kc_crop_stage = np.zeros((26, 3), dtype=np.float32)
        self.kc_crop_stage[:, 0] = crop_data['kc_ini']
        self.kc_crop_stage[:, 1] = crop_data['kc_mid']
        self.kc_crop_stage[:, 2] = crop_data['kc_end']

        # For now only Maize, with same planting date
        
        self.start_day = np.full(grid, planting_date) # start_day 
        
        self.crop_age_days = current_day_of_year  - self.start_day
        self.crop_harvest_day = np.full(grid, harvest_date) # harvest_day

        if self.start_day[0] <= current_day_of_year:
            self.cropKC = self.get_crop_kc(crop_map, self.crop_age_days, self.crop_harvest_day, self.crop_stage_data, self.kc_crop_stage, ini_land_use)
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
        crop_yield_factors = self.crop_yield_factors()
        
        return self._get_yield_ratio(
            crop_map,
            actual_transpiration / potential_transpiration,
            crop_yield_factors['alpha'],
            crop_yield_factors['beta'],
            crop_yield_factors['P0'],
            crop_yield_factors['P1'],
        )

    def get_yield(self, actual_transpiration, potential_transpiration, crop_map):
    
        yield_ratio = self.get_yield_ratio(actual_transpiration, potential_transpiration, crop_map)
        yield_irr_maize = (yield_ratio*0+1) * 2.4 
        yield_irr_casava = (yield_ratio*0+1) * 4.0
        yield_actual_maize = yield_ratio * yield_irr_maize
        yield_actual_casava = yield_ratio * yield_irr_casava
        return yield_actual_maize, yield_irr_maize, yield_actual_casava, yield_irr_casava

#------------------------------------------------
#                CROP PRODUCTION
#------------------------------------------------

    def Crop_production(self, crop_map, type_of_management, type_of_crop, coordinates, n, land_size):
        ''' 
        Return the the production of crops [*1000 kg], based on crop yields [kg / hectare], land size [hectare] , and rainfed/irrigated [0/1]
        '''
        average_aet = np.where(np.nanmean(self.actual_transpiration_crop, axis = 1) > 0, np.nanmean(self.actual_transpiration_crop, axis = 1), 0)
        average_pet = np.where(np.nanmean(self.potential_transpiration_crop, axis = 1) > 0, np.nanmean(self.potential_transpiration_crop, axis = 1), 0)

        self.yield_crops = self.get_yield(average_aet, average_pet, crop_map)

        plt.imshow((average_aet/average_pet).reshape(92,45))
        plt.show()
        
        crop_produce = np.ones(n)
        # land_size [hectare]
        # yield_crops = [*1000 kg / hectare]

        for i in range (n):
            if type_of_management[i] == 0 and type_of_crop[i] == 0: #rainfed maize
                crop_produce[i] = self.yield_crops[0].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* land_size[i] *1000 # kg
            elif type_of_management[i] == 1 and type_of_crop[i] == 0: #irrigated maize
                crop_produce[i] = self.yield_crops[1].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* land_size[i] *1000 # kg
            elif type_of_management[i] == 0 and type_of_crop[i] == 1: #rainfed casava
                crop_produce[i] = self.yield_crops[2].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* land_size[i] *1000 # kg
            else: #irrigated casava
                crop_produce[i] = self.yield_crops[3].reshape(self.height, self.width)[coordinates[i,1],coordinates[i,0]]* land_size[i] *1000 # kg

        assert (crop_produce >= 0).all()
        assert (crop_produce != np.nan).all()
        return crop_produce
    
#------------------------------------------------
#               LIVESTOCK AND GRASS
#------------------------------------------------

    def grass_yield(self):
        '''
        Grass yield per grid cell (no grass consumed yet) 
        Based on doi.org/10.1002/2015WR017841 + Pande & Savenije
        '''

        # calculate grass yield, per half year
        Kg = 0.4 # yield response factor
        yield_max = 400 #*1000kg /hectare , ICPAC forage hazard watch

        average_aet = np.where(np.nanmean(self.actual_transpiration_grass, axis = 1) > 0, np.nanmean(self.actual_transpiration_grass, axis = 1), 0)
        average_pet = np.where(np.nanmean(self.potential_transpiration_grass, axis = 1) > 0, np.nanmean(self.potential_transpiration_grass, axis = 1), 0)
        
        yield_grass =  np.maximum(0, np.minimum(1 - ( Kg * ( 1 - np.where(average_pet <= 0, 0, (average_aet / average_pet).reshape(self.height, self.width) ))) , 1)) * yield_max
        #self.yield_grass = np.where(self.land_use.reshape(self.grid) != 50 or 40 or 0, yield_grass, np.zeros(self.grid)) # national parks may be added...
        
        assert (yield_grass != np.nan).all()
        assert (yield_grass >= 0).all()

        return yield_grass.reshape(self.height, self.width)


    def Livestock_production(self, grass_yield, sum_livestock, nr_livestock, feed_required, feed_residue, net_birth_rate, weight_gain_rate): # yearly timestep, should differiate between types of livestock
        ''' 
        Livestock production as a function of grass_yield grass consumed,, and animal specific characteristics
        Based on doi.org/10.1002/2015WR017841
        '''
        #-> first order preference!
        # can also scale based on feed_requirement... cows need more food. average_feed_required = self.feed_required.mean(axis=1)

        grass_consumed = np.minimum(nr_livestock * feed_required* (1- feed_residue), grass_yield *1000*(nr_livestock /(np.where((nr_livestock >0) & (sum_livestock > 0), nr_livestock /sum_livestock, 1))))
        grass_availability = np.maximum(grass_yield *1000 *(np.where(sum_livestock > 0, nr_livestock /sum_livestock, 1)),0) #- (grass_consumed * self.agent_population),0) # *1000 to convert to kg from grid to every farmer, ratio to # agent population/household size.. (minimum 1)

        carrying_capacity = grass_availability / (feed_required * (1 - feed_residue)) # grass availability to cows.. * nr cows??
        rate_of_growth = net_birth_rate + np.minimum(np.maximum((weight_gain_rate * (grass_availability / ((1 - feed_residue) * nr_livestock))), 0), 2) # CHECK with FGD -> max number of babies born per year
        livestock_produce = np.maximum(nr_livestock + (rate_of_growth * (1 - (nr_livestock/carrying_capacity)) * nr_livestock), 1) #per agent

        assert (livestock_produce >= 0).all()
        assert (livestock_produce != np.nan).all()
        return livestock_produce

#------------------------------------------------
#                   WATER DEMAND 
#------------------------------------------------

    # DISTANCE MATRIX FOR ABSTRACTION
    def Extract_x(self, lst):
        return [item[0] for item in lst]

    def Extract_y(self, lst):
        return [item[1] for item in lst]

    def distance_water(self, abstraction_points_DRYP, ro, agents):
        ''' 
        Calculation distance to water source (river and groundwater abstraction point).
        Return the distance (number of cells?) and the coordinates of nearest water sources that has water. 
        '''
        index_groundwater = np.zeros(agents.n, dtype = int)
        index_river = np.zeros(agents.n, dtype = int)
        

        coordinates = []

        for y in range(self.height):
            for x in range(self.width):
                coordinates.append((x,y)) 

        abstraction_points = abstraction_points_DRYP

        self.dis_groundwater = np.zeros(agents.n)
        self.dis_river = np.zeros(agents.n)
 
        abstraction_points = np.where(abstraction_points == True, 1, 1000)
        #abstraction_points = np.where(groundwater_depth < 100, 1, 1000)
        river_network = agents.river_network.flatten()
        discharge = ro.dis_dt

        river_ = np.where(river_network >= 1, 1, 1000)
        river_network = np.where(discharge >0, river_, 1000)

        for i in range(agents.n):
            agent_coordinates_x = agents.coordinates[i,0]
            agent_coordinates_y = agents.coordinates[i,1]

            distance = 1 + np.sqrt(abs((self.Extract_x(coordinates) - agent_coordinates_x)**2 + (self.Extract_y(coordinates) - agent_coordinates_y)**2) ) #distance in relation to other grid points
            
            groundwater_distance = distance * abstraction_points #> high numbers is large distance
            river_distance = distance * river_network
            closest_groundwater = np.min(groundwater_distance) 
            closest_river = np.min(river_distance) # distance to closest source

            index_closest_groundwater = np.argmin(groundwater_distance) #argmin gives index of array where value is minimal (closest source)
            index_closest_river = np.argmin(river_distance) # index of min value

            index_groundwater[i] = index_closest_groundwater
            index_river[i] = index_closest_river

            self.dis_groundwater[i] = closest_groundwater
            self.dis_river[i] = closest_river

        
        # return for every agent their abstraction grid location, use that in the abstraction function
        return self.dis_groundwater, self.dis_river, index_groundwater, index_river

    def water_household(self, agents):
        ''' 
        Water consumption of household. Assume 50 L/day for rural areas mm*gridsize [m2] mm = L/m2
        ''' 
        water_household = 50 / self.grid_size  * agents.agent_population
        return water_household

    def water_livestock(self,agents):
        ''' 
        Water consumption of livestock. From Steinfeld et al., 2006. Method based on Wada et al.
        ''' 
        water_livestock = np.where(agents.agent_population ==0.0, 0.0, ( 25 * agents.nr_livestock[:,0] +  9.6 * agents.nr_livestock[:,1]) / agents.grid_size * agents.agent_population ) / np.where(agents.HH_size < 1.0, 1.0, agents.HH_size)# L/m2 = mm L/1000 per livestock type, based on 25degrees Celsius. 25 L/day cow, 9.6 L/day goat
        return water_livestock

    def irrigation_schemes(sefl):
        ''' Define the irrigation schemes in the catchment'''
        
        #rapsu:0.266724, 38.233585, 250 acres, 101.171411 hectares
        #Malkadaka: 0.850197, 38.500191, 70 acres
        #Oldonyiro, planned. 215 acres
        pass

    def large_scale_agriculture(self):
        ''' 
        Calculate water abstraction of large-scale agriculture
        ''' 
        farms = 7
        locations = np.zeros((farms, 2), dtype=np.float32)

        locations[:, 0] = np.array([308372, 323798.28, 309419.36, 308641.78, 318151.49, 314672.65, 310992.84])
        locations[:, 1] = np.array([9445.55, 13376.73, 8876.57, 11674.72, 10805.31, 9839.26, 9335.96]) 

        grid_large_scale_agriculture = np.zeros(self.grid).reshape(self.height, self.width)

        rate_of_absraction = 37 #m3/hectare
        hectares = np.array([138, 22, 20, 40, 12, 13 , 5 ])
        #locations = [308372.04,9445.55], [323798.28,13376.73], [309419.36,8876.57], [308641.78, 11674.72], [318151.49,10805.31], [314672.65,9839.26], [310992.84,9335.96], 
        
        # coordinates = self.location_index(locations)
        # #bring locations to coordinates in grid
        # for i in range(farms):
        #     grid_large_scale_agriculture[coordinates[0][i],coordinates[1][i]] = hectares[i] * rate_of_absraction /1000 # from m3 to mm (per grid cell of 1km2)
        #timaflor (0.085418,37.2781470), kisima (0.120977, 37.416719), lobelia (0.080273, 37.287555), lolomarik (0.105577, 37.280569), uhuru (0.097719, 37.365994), bemack (0.088981, 37.334744), kikwetu (0.084428, 37.301689)

        # , mt kenya flowers/kariki (11 ha) (-0.013945, 37.121485)[290930.61,9998457.82], chulu (9 ha) (0.037395, 37.157242)[294911.62, 4135.42], Kongoni River farm (20 ha) (0.069339, 37.159346)[295145.97, 7668.03] zitten niet in sub-ewaso catchment?
        return grid_large_scale_agriculture.flatten()

    def industry(self):
        pass

    def abstract_water(self, theta, theta_fc, Duz, abstraction_points_DRYP, env_state, agents, ro):
        ''' 
        Calculation water abstracted from nearest source per type of usage (household, irrigation, livestock).
        Return abstraction on grid for river and groundwater use per type of usage.
        ''' 

        distance_water = self.distance_water(abstraction_points_DRYP, env_state, ro, agents)
        dtheta = (theta_fc-theta)
        dtheta[dtheta < 0] = 0
        SMD = Duz*dtheta # soil moisture demand
        irr_demand = agents.irrigation_demand_factor * (SMD) # Soil moisure deficit on entire grid [mm]
        hh_demand = self.water_household(agents)# per household 
        liv_demand = self.water_livestock(agents) # per household 
        abstraction_points = abstraction_points_DRYP.reshape(self.height, self.width)
        gw_hh = np.zeros(env_state.grid_size)
        gw_irr = np.zeros(env_state.grid_size)
        gw_liv = np.zeros(env_state.grid_size).reshape(self.height,self.width) # to from to coordinates of livestock positions
        riv_hh = np.zeros(env_state.grid_size)
        riv_irr = np.zeros(env_state.grid_size)
        riv_liv = np.zeros(env_state.grid_size).reshape(self.height, self.width)
        discharge = env_state.SZgrid.at_node['discharge']

        for i in range(agents.n):
            if distance_water[0][i] < distance_water[1][i]: # if distance water groundwater is smaller than river -> can also do half half, if distance is equal
                index_gw = distance_water[2][i]
                if agents.adapt_measure_3[i] == 1 and agents.current_day_of_year >= agents.planting_date and agents.current_day_of_year <= agents.harvest_date:
                    gw_irr[index_gw] = np.maximum(0, (np.minimum(irr_demand[index_gw] * agents.land_size[i] * agents.agent_population[i] / 100, 5.0))) #hectare to km2 #add irrigation, Max 1 mm
                
                gw_hh[index_gw] = hh_demand[i] # assign individual household to gg_hh grid
                gw_liv[agents.livestock_coords[i][1]][agents.livestock_coords[i][0]] = liv_demand[i] # 0 and 0...??

            else:
                index_riv = distance_water[3][i]
                if agents.adapt_measure_3[i] == 1 and agents.current_day_of_year >= agents.planting_date and agents.current_day_of_year <= agents.harvest_date:
                    riv_irr[index_riv] = np.maximum(0, (np.minimum( irr_demand[index_riv] * agents.land_size[i] * agents.agent_population[i] /100, 5.0))) # discharge in that cell #discharge[distance_water[3][i]], Max 5 mm
        
                riv_hh[index_riv] = hh_demand[i]
                riv_liv[agents.livestock_coords[i][1]][agents.livestock_coords[i][0]] = liv_demand[i]

        #Communicate water abstractions to agents for plotting
        agents.abstraction_gw_hh = gw_hh
        agents.abstraction_gw_irr = gw_irr
        agents.abstraction_gw_liv = gw_liv.flatten()
        agents.abstraction_riv_hh = riv_hh
        agents.abstraction_riv_irr = riv_irr
        agents.abstraction_riv_liv = riv_liv.flatten()

        return gw_hh, gw_irr, gw_liv.flatten(), riv_hh, riv_irr, riv_liv.flatten()

