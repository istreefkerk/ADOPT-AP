import numpy as np
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import coords_to_pixels
import pandas as pd
import os
import pyproj
import stats


class Init_(AgentBaseClass):
    '''
    Class for the initialisation settings of the agents. Called apon in the Agents class.
    '''

    def _initiate_locations(self):
        """
        Initiate location of agents and grid characteristics.

        Sets up the spatial grid, agent coordinates, population, and
        neighborhood relationships.
        """

        self.width = 201 #this information can be found in input data of DRYP
        self.height = 174 
        self.x_grid = 925.145414825871 
        self.y_grid = 925.145414825871 
        self.grid = self.height * self.width
        self.grid_size =self.x_grid * self.y_grid
        self.gt = (self.model.xmin, self.x_grid, 0.0, self.model.ymin, 0.0, self.y_grid)
        self.catchment =  np.flip(self.model.data.mask.get_data_array(), axis = 0)
        self.density_grid = np.flip(self.model.data.density.get_data_array(),axis = 0) 
        self.elevation_grid = np.flip(self.model.data.elevation.get_data_array(),axis = 0)
        self.farm_agents = np.flip(self.model.data.farm_agents.get_data_array(),axis = 0) 

        #self.seeds = 1 + self.model.config['general']['scenario_1'] # set seeds for every run

        self.land_use = np.flip(self.model.data.land_cover.get_data_array(),0) # gives (y,x)

        self.large_scale_agri = self.get_large_scale_agriculture()

        self.lonlat_agri = np.zeros((len(self.large_scale_agri[0]), 2), dtype=int)

        self.lonlat_agri[:, 0] = self.large_scale_agri[1]
        self.lonlat_agri[:, 1] = self.large_scale_agri[2] # x, y

        self.coordinates_agri = self.location_index(self.lonlat_agri)

        
        self.surface_abstraction_agri = np.where(self.large_scale_agri[0]['surface'] > 0,self.large_scale_agri[0]['surface'], 0)
        self.ground_abstraction_agri = np.where(self.large_scale_agri[0]['ground'] > 0, self.large_scale_agri[0]['ground'], 0)
        self.storage_agri = np.where(self.large_scale_agri[0]['Volume_storage [m3]'] > 0,self.large_scale_agri[0]['Volume_storage [m3]'], 0)
        self.greenhouses_agri = np.where(self.large_scale_agri[0]['Area_greenhouses [m2]'] > 0, self.large_scale_agri[0]['Area_greenhouses [m2]'], 0)
        
        self.grid_greenhouses = np.zeros((self.height, self.width))
        self.grid_greenhouses[self.coordinates_agri[:][1],self.coordinates_agri[:][0]] = self.greenhouses_agri

        self.lonlat = []
        self.agent_population = []

        if self.model.config['general']['scenario'] == 'forest' or self.model.config['general']['scenario'] == 'large_scale_agriculture':

            for i in range(self.height):
                for j in range(self.width):
                    if self.catchment[i,j] > 0.0 and self.farm_agents[i,j] !=1 and self.grid_greenhouses[i,j] <= 1 and self.land_use[i,j] <= 110 and self.land_use[i,j] != 50:
                        if self.density_grid[i,j] > 0.0:
                            self.lonlat.append(np.array([self.model.xmin + (j * self.x_grid + 0.5 * self.x_grid), self.model.ymin + (i * self.y_grid + 0.5 * self.y_grid)], dtype=np.float32)) # lat lon to grid: height-i?
                            self.agent_population.append(self.density_grid[i,j]) 

        self.lonlat = np.array(self.lonlat)
        self.agent_population = np.asarray(self.agent_population) 
        self.n = len(self.lonlat)

        self.coordinates = np.zeros((self.n, 2), dtype=int)
        self.coordinates[:, 0] = self.location_index(self.lonlat)[0][:]
        self.coordinates[:, 1] = self.location_index(self.lonlat)[1][:]

        #Neigbourhood
        
        self.radius_neighbourhood = self.model.config['parameters']['neighbourhood_radius']

        self.radius = np.full(self.n, self.radius_neighbourhood/1000)
        self.radius = self.radius.astype(int)

        self.coords_neighbours = self.coords_neighbourhood(self.lonlat, self.radius)
        self.range_lands =  self.coords_neighbourhood(self.lonlat, self.radius * 10)

        self.livestock_coords = self.coordinates.copy()

        ### define distance to other agents:
        self.distance_to_neighbours, self.index_neighbours = self.Distance()

        assert self.lonlat[:,0].min() >= self.model.xmin
        assert self.lonlat[:,0].max() <= self.model.xmax
        assert self.lonlat[:,1].min() >= self.model.ymin
        assert self.lonlat[:,1].max() <= self.model.ymax

    @property
    def activation_order(self):
        """
        Get the activation order for agents.

        Returns:
            np.ndarray: Array of agent indices.
        """
        return np.arange(self.n, dtype=np.int32)

    def _initiate_attributes(self):
        """
        Initiate attributes of agents.

        Initializes demographic, economic, and adaptation-related attributes
        for each agent based on climate zone and scenario configuration.
        """
        #Characteristics of agents

        self.climate_zone = self.model.data.climate_zone.sample_coords(self.lonlat)
        self.elevation = self.model.data.elevation.sample_coords(self.lonlat)
        self.land_cover_agent = self.model.data.land_cover.sample_coords(self.lonlat)

        # Age of household head
        self.age = np.where((self.climate_zone == 53), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.2757150057973239, 29.115534677184037, 8.097626181715015, size=self.n),0),60), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.23603638275095812, 29.624383990553913, 8.57732319340807, size=self.n),0),60), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.16930110091100253, 20.296920968632236, 13.919713052593494, size=self.n), 0), 60))) 

        # Education level (6 classes)
        self.edu = np.where((self.climate_zone == 53), np.minimum(np.maximum(stats.fixed_gamma(self.seeds, 0.330834834093068, -1.2362664483930804e-20, 1.614618293830849, size=self.n),0),6), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), np.minimum(np.maximum(stats.fixed_gamma(self.seeds, 0.5104698844225849, -2.7651053091078577e-26, 0.9417021243154988, size=self.n),0),6), np.minimum(np.maximum(stats.fixed_gamma(self.seeds, 3.7881205176443022, -0.7573919824074247, 0.7030015831427152, size=self.n), 0), 6))) 
        # Assets in USD
        self.assets = np.where(((self.climate_zone == 53) | (self.climate_zone == 18) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone ==43)), np.minimum(np.maximum(stats.fixed_pareto(self.seeds, 0.14385711751652053, -2.6556085108463225, 2.6387186240468656, size=self.n),0),55000), np.minimum(np.maximum(stats.fixed_pareto(self.seeds, 0.19038972588311004, -2.3583499310215865, 2.3583484337792386, size=self.n), 0), 55000)) 
        # Household size
        self.HH_size = np.where((self.climate_zone == 53), np.minimum(np.maximum(stats.fixed_dweibull(self.seeds, 1.2086355484512648, 5.7070585354476835, 2.3633941978015462, size=self.n), 1), 15), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), np.minimum(np.maximum(stats.fixed_dweibull(self.seeds, 1.4492085474802183, 5.552990398419958, 2.5000221880013616, size=self.n),1),15), np.minimum(np.maximum(stats.fixed_dweibull(self.seeds, 1.3728123574283657, 5.406527251268886, 1.6090867657184331, size=self.n), 1), 15)))
        # Sex of household head
        self.sex = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [0, 1], size=(self.n), p = [0.5345455, 1 - 0.5345455]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [0, 1], size=(self.n), p = [0.49334, 1 - 0.49334]), stats.fixed_choice(self.seeds, [0, 1], size=(self.n), p = [0.6379, 1 - 0.6379])))
        # 0 = no extension services/ external info , 1 = extension services/ external info (e.g. radio)
        if self.model.config['general']['scenario_2'] == 'extension_twice':
            self.receive_extension = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1 , 0], size=(self.n), p = [0.42545 *2, 1 - (0.42545*2)]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.23788 *2, 1 - (0.23788*2)]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.094545 *2, 1 - (0.094545*2)])))
        elif self.model.config['general']['scenario_2'] == 'extension_half':
            self.receive_extension = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1 , 0], size=(self.n), p = [0.42545 /2, 1 - (0.42545/2)]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.23788 /2, 1 - (0.23788/2)]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.094545 /2, 1 - (0.094545/2)])))
        else:
            self.receive_extension = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1 , 0], size=(self.n), p = [0.42545, 1 - 0.42545]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.23788, 1 - 0.23788]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.094545, 1 - 0.094545])))
        
        self.receive_aid = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.271, 0.421, 0.253, 0.055]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.053, 0.220, 0.445, 0.282]), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.271, 0.421, 0.253, 0.055])))
        self.forecast_information = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.293, 0.305, 0.353, 0.049]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.018, 0.088, 0.204, 0.690]), stats.fixed_choice(self.seeds, [4, 3, 2, 1], size=(self.n), p = [0.293, 0.305, 0.353, 0.049])))

        self.intention_to_behavior = self.model.config['parameters']['intention_to_behavior']

        # Adaptation measures

        # If agents obtain land, give land size [acres]
        self.own_land = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [62/274, 1 - 62/274]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [2/227, 1 - 2/227]), 1))

        # Migration 
        self.adapt_measure_0 = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.094545, 1 - 0.094545]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.246695, 1 - 0.246695]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.094545, 1 - 0.094545])))  
        # Different livestock (goats) 
        self.adapt_measure_1 = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.094545, 1 - 0.094545]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.6079, 1 - 0.6079]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.482759, 1 - 0.482759])))
        # Different crop (casava)
        self.adapt_measure_2 = np.where(((self.climate_zone == 53) & (self.own_land == 1)), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.145454 / (self.own_land ==1).mean(), 1 - (0.145454 / (self.own_land ==1).mean())]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71) & (self.own_land == 1)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.0001/ (self.own_land ==1).mean(), 1 - (0.0001 / (self.own_land ==1).mean())]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [0.155 / (self.own_land ==1).mean(), 1 - (0.155/ (self.own_land ==1).mean())]))) 
        # Irrigation 
        self.adapt_measure_3 = np.where(((self.climate_zone == 53) & (self.own_land == 1)), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.08727 / (self.own_land ==1).mean(), 1 - (0.08727 / (self.own_land ==1).mean())]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71) & (self.own_land == 1)), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.088/ (self.own_land ==1).mean(), 1 - (0.088 / (self.own_land ==1).mean())]), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.172414 / (self.own_land ==1).mean(), 1 - (0.172414/ (self.own_land ==1).mean())]))) 

        # Soil moisture conservation (pasture and agroforestry)
        self.adapt_measure_5 = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.3345, 1 - 0.3345]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.20264, 1 - 0.20264]), stats.fixed_choice(self.seeds, [1,0], size=(self.n), p = [0.3345, 1 - 0.3345]))) 
        
        #create grid to communicate with DRYP
        self.adapt_measure_5_grid = np.zeros((self.height, self.width))

        self.lifetime_measures = [1, 10, 1, 10, 10, 10]
        
        self.land_acres = np.where(((self.climate_zone == 53) & (self.own_land ==1)), stats.fixed_choice(self.seeds, [0.5,1.5], size=(self.n), p = [0.5, 0.5]), np.where((((self.climate_zone == 62) | (self.climate_zone == 71)) & (self.own_land == 1)), stats.fixed_choice(self.seeds, [0.5, 1.5], size=(self.n), p = [0.5, 0.5]), np.where((((self.climate_zone == 18) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone == 43 )) & (self.own_land == 1)),np.minimum(np.maximum(stats.fixed_dweibull(self.seeds, 1.2086355484512648, 5.7070585354476835, 2.3633941978015462, size = self.n), 0), 7.5), 0)))
        self.land_size = np.minimum(self.land_acres, (self.x_grid * self.y_grid * 0.404685642 / 100 / 100 /self.agent_population * self.HH_size)) #land size [acres] -> [ha]. Can have a maximum land_size of the # of agents divided in grid cell
        
        self.nr_livestock = np.ones((self.n, 2))
        self.own_cattle = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [62/274, 1 - 62/274]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [182/227, 1 - 182/227]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [34/58, 1 - 34/58])))
        self.own_shoats = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [137/274, 1 - 137/274]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [180/227, 1 - 180/227]), stats.fixed_choice(self.seeds, [1, 0], size=(self.n), p = [28/58, 1 - 28/58])))
        self.nr_livestock[:, 0] = np.where(((self.climate_zone == 53) & (self.own_cattle ==1)), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.9766438248886353, 4.969251334847236, 5.336444937797102, size=self.n),0),100), np.where((((self.climate_zone == 62) | (self.climate_zone == 71)) & (self.own_cattle ==1)), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.8826725623909313, 2.5291453008599887, 2.128602924778958, size=self.n),0),100), np.where((((self.climate_zone == 18) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone == 43) ) & (self.own_cattle == 1)), np.minimum(np.maximum(stats.fixed_genextreme(self.seeds, -0.1680578243835209, 2.5295233701978352, 1.979227305517283, size=self.n), 0), 100), 0)))
        self.nr_livestock[:, 1] = np.where(((self.climate_zone == 53) & (self.own_shoats ==1)), np.minimum(np.maximum(stats.fixed_burr(self.seeds, 4.030252431164184, 0.16789254960226052, 1.9999998014801932, 75.12249812745972, size=self.n),0),200), np.where((((self.climate_zone == 62) | (self.climate_zone == 71)) & (self.own_shoats ==1)), np.minimum(np.maximum(stats.fixed_burr(self.seeds, 2.5608684144098905, 0.3105282710450781, 0.9999999999979682, 31.8251029009367, size=self.n),0),200), np.where(((( self.climate_zone == 18) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone == 43) ) & (self.own_shoats == 1)), np.minimum(np.maximum(stats.fixed_burr(self.seeds, 1.562205060515856, 1.398278434634138, -0.6517716962106974, 5.960715509942904, size=self.n), 0), 200), 0)))

        # off-farm income: labour and business
        self.off_farm_income = np.where((self.climate_zone == 53), np.minimum(stats.fixed_expon(self.seeds, 10.0, 34059.0405904059, size=self.n), 320000), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), np.minimum(stats.fixed_expon(self.seeds, 0.0, 23153.153153153155, size=self.n),320000), np.minimum(stats.fixed_expon(self.seeds, 0.0, 48573.07692307692, size=self.n), 320000)))
        self.food_expenditures = self.HH_size * np.where(((self.climate_zone == 53) | (self.climate_zone == 18) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone ==43)), np.minimum(np.maximum(stats.fixed_gompertz(self.seeds, 2.0722789557281347, -1.3939473040928873e-06, 29802.877297070056, size=self.n),0),15000), np.minimum(np.maximum(stats.fixed_gompertz(self.seeds, 8595893638.1942, 277.77777776779703, 55116882102044.66, size=self.n),0), 15000))
        self.other_expenditures = self.HH_size * np.where(((self.climate_zone == 53) | (self.climate_zone == 8) | (self.climate_zone == 25) | (self.climate_zone == 35) | (self.climate_zone ==43)), np.minimum(np.maximum(stats.fixed_burr(self.seeds, 2.0866433228747714, 0.8045258584619412, -38.92183286861429, 6134.655398495872, size=self.n),0),60000), np.minimum(np.maximum(stats.fixed_burr(self.seeds, 1.7983491526718325, 1.0493954898488884, -12.174012461201514, 2223.338697647964, size=self.n),0), 60000))
        
        self.expenditures_crops = 130 * self.land_size #130 dollars/hectare -> 1000 ksh/acre 0.405 = 1 acre
        # livescosts tegemeo -> depends per type of livestock. Goats are cheaper than cattle. (350 ksh/goat, 1500 ksh/cow)
        self.expenditures_livestock = 12 * self.nr_livestock[:, 0] + 3 * self.nr_livestock[:, 1] #12 dollar/cow and 3 dollars/goat

        self.food_consumption = 103  # kg/year of crops per HH member

        # PROTECTION MOTIVATION THEORY PARAMETERS/VARIABLES
        self.selfefficacy = np.zeros((self.n, 6))
        self.selfefficacy[:,0] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.selfefficacy[:,1] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.selfefficacy[:,2] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.selfefficacy[:,3] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.selfefficacy[:,4] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.selfefficacy[:,5] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.alpha = self.model.config['parameters']['alpha']
        self.beta  = 1 - self.alpha #alpha + beta = 1
        self.gamma  = np.maximum(np.minimum(stats.fixed_uniform(self.seeds, 0.2105, 0.55158, size=self.n), 1), 0)
        self.delta  = np.maximum(np.minimum(stats.fixed_uniform(self.seeds, 0.39684, 0.7263, size=self.n), 1), 0)
        self.epsilon  = np.maximum(np.minimum(1 - self.gamma - self.delta, 1), 0)

        self.IntentionToAdapt_Livestock = np.zeros((self.n))
        self.IntentionToAdapt_Crops = np.zeros((self.n))
        self.IntentionToAdapt = np.zeros((self.n))
        self.CopingAppraisal_Livestock = np.zeros((self.n))
        self.CopingAppraisal_Crops = np.zeros((self.n))
        self.CopingAppraisal = np.zeros((self.n))
        self.RiskAppraisal_Livestock = np.zeros((self.n))
        self.RiskAppraisal_Crops = np.zeros((self.n))
        self.RiskAppraisal = np.zeros((self.n))
        self.adaptation_eff = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.damage = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception = np.zeros((self.n, 6))
        self.costperception[:,0] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception[:,1] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception[:,2] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception[:,3] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception[:,4] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        self.costperception[:,5] = stats.fixed_uniform(self.seeds, 0, 1, size=self.n)
        
        self.risk_perception = stats.fixed_uniform(self.seeds, 0, 1, size=(self.n))

        self.maize_price = 0.35 # USD/kg
        self.milk_price_goats = 0.50 # USD/litre milk # 60 ksh
        self.milk_price_cows = 0.70 # USD/litre milk # 90 ksh

        self.cow_price = 160 # USD
        self.goat_price = 40 # USD
        
        self.adapt_costs = [10 * self.model.config['parameters']['adapt_costs'], 20 * self.model.config['parameters']['adapt_costs'], 30* self.model.config['parameters']['adapt_costs'], 1000 * self.model.config['parameters']['adapt_costs'], 250 * self.model.config['parameters']['adapt_costs'], 415 *self.model.config['parameters']['adapt_costs']]


    def _initiate_socio_hydrology(self):
        """
        Initialize socio-hydrological variables for agents.

        Sets up water and land use variables, livestock characteristics,
        crop production, and abstraction/irrigation factors.
        """

        # WATER AND LAND USE VARIABLES

        self.at_river = self.model.data.river_network.sample_coords(self.lonlat)
        self.elevation_grid = np.flip(self.model.data.elevation.get_data_array(),0)
        self.sub_catchment = self.model.data.sub_catchment.sample_coords(self.lonlat)
        self.admin = self.model.data.admin.sample_coords(self.lonlat)
        
        self.distance_hh_water = np.zeros(self.n)

        #Livestock Characteristics
        
        self.feed_required_cattle = 7 * 365 #(kg/livestock/year)
        self.feed_residue_cattle = 0.3
        self.net_birth_rate_cattle = 0.15
        self.weight_gain_rate_cattle = 1 / self.feed_required_cattle

        self.feed_required_goats = 6 * 365 #(kg/livestock/year)
        self.feed_residue_goats = 0.15
        self.net_birth_rate_goats = 0.45
        self.weight_gain_rate_goats = 1 / self.feed_required_goats

        self.max_milk_production_cows = 3.0  # 3 L per cow per day
        self.max_milk_production_goats = 1.5  # 1.5 L per goat per day

        self.cow_milk_consumption = 0.25 # l/year HH member
        self.goat_milk_consumption = 0.05 # 18 l/year HH member
        
        self.livestock_produce = np.sum(self.nr_livestock, axis = 1)
        self.milk_production = self.nr_livestock[:, 0] * self.max_milk_production_cows/2.5 + self.nr_livestock[:, 1] * self.max_milk_production_goats/2.5 # L/day

        self.crop_produce = self.land_size * stats.fixed_uniform(self.seeds, 150, 1500, size=self.n) #statistics from survey data
        self.crop_production = stats.fixed_uniform(self.seeds, 150, 1500, size=self.n)

        self.livelihood = (self.crop_produce * self.maize_price) + (np.maximum(((self.nr_livestock[:, 0] * self.max_milk_production_cows/2*365) - (self.cow_milk_consumption*365 * self.HH_size)), 0) * self.milk_price_cows) + (np.maximum(((self.nr_livestock[:, 1]* self.max_milk_production_goats/2*365) - (self.goat_milk_consumption*365 * self.HH_size )), 0)  * self.milk_price_goats)
        self.livestock_livelihood = np.where((self.livelihood == 0.0), 0.5, ((np.maximum(((self.nr_livestock[:, 0] * self.max_milk_production_cows/2*365) - (self.cow_milk_consumption*365 * self.HH_size)), 0)  * self.milk_price_cows) + (np.maximum(((self.nr_livestock[:, 1]* self.max_milk_production_goats/2*365) - (self.goat_milk_consumption*365 * self.HH_size )), 0)  * self.milk_price_goats)) / self.livelihood)
        self.crop_livelihood = np.where(self.land_size == 0.0, 0 , (self.crop_produce * self.maize_price) / self.livelihood)
        
        #Abstraction and irrigation factors
        self.distribution_abstraction_points = self.model.config['parameters']['abstraction_p']
        self.irrigation_demand_factor = self.model.config['parameters']['irrigation_demand_factor']
        self.abstraction_p = stats.fixed_choice(self.seeds, [True, False], size=(self.grid), p = [self.distribution_abstraction_points, 1 - self.distribution_abstraction_points])

        #Seasons
        self.planting_date = 274 # 1st of October
        self.harvest_date = 360
        self.start_dry_season = 140 #20th of May
        self.end_dry_season = 260
        self.lenght_short_dry_season = 60 #days

        self.water_cost_dry_season = 10 # ksh
        self.water_cost_wet_season = 5 # ksh
        
        self.coordinates_max_grass = np.zeros((self.n, 2), dtype=np.int32)

        self.distance_migration = np.zeros(self.n)
        
        self.yield_grass = np.maximum(stats.fixed_uniform(self.seeds, 1,4, size = self.grid),0) # yield in *1000kg/hectare?
        self.yield_crops = np.maximum(stats.fixed_uniform(self.seeds, 1,4, size = self.grid),0) * 1000 # yield in *1000kg/hectare?

        self.grid_livestock_nrs = np.zeros((self.height, self.width))

        self.crop_map = np.full((self.grid), -1)
        self.land_use_flat = self.land_use.flatten()

        self.crop_map.reshape(self.height, self.width)[self.coordinates[:,1],self.coordinates[:,0]] = np.where((self.own_land == 1), 1, -1) # crop map at -1 if there is no crop, 1 maize
        self.initial_kc = np.where((self.land_use_flat == 40) , 1.0, np.where((self.land_use_flat >= 112), 1.2, np.where((self.grid_greenhouses.flatten() > 0), 1 - (self.grid_greenhouses.flatten() / self.x_grid / self.y_grid), 1.0 ) )) #urban and cropland to 0.5. herbeous/grass to 1.0 [default], forest to 1.2

        self.river_network = np.flip(self.model.data.river_network.get_data_array(),0)

    def _initiate_storing_variables(self):
        """
        Initialize variables for storing agent memory and time series.

        Sets up arrays for harvest memory, water availability, and
        hydrological variables.
        """

        self.latest_harvest_crops = np.empty((self.n, 10), dtype=np.float32) # array to store harvest memory
        self.latest_harvest_crops[:] = np.NaN
        self.latest_harvest_crops[:,0] = self.crop_produce.copy()

        self.latest_harvest_livestock = np.empty((self.n, 10), dtype=np.float32) # array to store harvest memory
        self.latest_harvest_livestock[:] = np.NaN
        self.latest_harvest_livestock[:,0] = self.livestock_produce.copy()  / 4

        self.yearly_harvest_livestock = np.empty((self.n, 2), dtype=np.float32) # array to store harvest memory
        self.yearly_harvest_livestock[:] = np.NaN
        self.yearly_harvest_livestock[:,0] = self.livestock_produce.copy() 
        
        self.memory_water_available = np.zeros((self.n, 8), dtype=np.float32) # array to store income memory

        self.risk_appraisal = stats.fixed_uniform(self.seeds, 0,1,(self.n,2)).copy()

        self.rainfall = np.zeros(self.grid)
        self.soil_moisture = np.zeros(self.grid)
        self.soil_moisture_deficit = np.zeros(self.grid)
        self.discharge = np.zeros(self.grid)
        self.groundwater = np.zeros(self.grid)
        self.discharge_at_outlet = np.nan

        self.abstraction_gw_hh = np.zeros(self.grid)
        self.abstraction_gw_irr = np.zeros(self.grid)
        self.abstraction_gw_liv = np.zeros(self.grid)
        self.abstraction_riv_hh = np.zeros(self.grid)
        self.abstraction_riv_irr = np.zeros(self.grid)
        self.abstraction_riv_liv = np.zeros(self.grid)

        self.storage_grid_agri = np.zeros((len(self.large_scale_agri[0])))
        self.groundwater_abstraction_grid_agri = np.zeros(self.grid)
        self.surface_abstraction_grid_agri = np.zeros(self.grid)
        self.irrigation_grid_agri = np.zeros(self.grid)
        self.rain_storage_grid_agri = np.zeros(self.grid)
        self.river_abstraction_farm_agri = np.zeros(self.grid)	
        self.groundwater_abstraction_farm_agri = np.zeros(self.grid)
        self.water_from_storage_rain_roses = np.zeros(self.grid)
        self.water_from_storage_discharge_roses = np.zeros(self.grid)
        self.water_to_be_abstracted_gw_roses = np.zeros(self.grid)
        
        self.irrigation_from_storage = np.zeros(self.grid)
        self.water_from_surface = np.zeros(self.grid)
        self.water_from_precipitation = np.zeros(self.grid)
        self.water_storage_agri = np.zeros(self.grid)
        self.storage = np.zeros(self.grid)

        self.aof = np.zeros(self.grid) #flow rate of stream flow abstracted [m3/dt] (*dt is model timestep)
        self.auz = np.zeros(self.grid) #flow rate of irrigation (+) or storage (-) [mm/dt]
        self.asz = np.zeros(self.grid) #flow rate of groundwater abstraction [mm/dt]

    def get_large_scale_agriculture(self):
        """
        Load and process large-scale agriculture data.

        Returns:
            tuple: DataFrame of farm data, x coordinates, y coordinates.
        """

        df = pd.read_csv(os.path.join('DataDrive', 'Data_commercial_farms_flat.csv'), delimiter = ';')
        p = pyproj.Proj(proj='utm', zone=37, ellps='WGS84', preserve_units=True)
        x,y = p(np.array(df['Coordinates_X']),np.array(df['Coordinates_Y']))
        df['Locations'] = list(zip(x,y))
        return df, x, y
