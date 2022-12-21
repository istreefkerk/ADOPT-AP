from logging import logProcesses
import numpy as np
from honeybees.agents import AgentBaseClass
from honeybees.library import neighbors
from honeybees.library.neighbors import find_neighbors
from honeybees.library.raster import coords_to_pixels, write_to_array
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime

class Farmers(AgentBaseClass):

#------------------------------------------------
#             INITIATION PROCEDURE
#------------------------------------------------
    def __init__(self, model, agents): #, model, agents
        self.model = model
        self.agents = agents
        self._initiate_agents()
        
    def _initiate_agents(self):
        self._initiate_locations()
        self._initiate_attributes()
        self._initiate_storing_variables()
        self._initiate_socio_hydrology()
        
        print(f'initialized {self.n} agents')

    def _load_initial_state(self):
        self.load_timestep_data()

    def _initiate_locations(self):
        '''
        Initiate location of agents and grid characteristics
        '''
        
        self.width = 45  #this information can be found in input data of DRYP
        self.height = 92 
        self.x_grid = 925.145414826388 
        self.y_grid = 925.145414826388
        self.grid = self.height * self.width
        self.grid_size =self.x_grid * self.y_grid
        self.gt = (self.model.xmin, self.x_grid, 0.0, self.model.ymin, 0.0, self.y_grid)
        self.catchment =  np.flip(self.model.data.mask.get_data_array(), axis = 0)
        self.density_grid = np.flip(self.model.data.density.get_data_array(),axis = 0)
        self.elevation_grid = np.flip(self.model.data.elevation.get_data_array(),axis = 0)
    
        self.lonlat = []
        self.agent_population = []

        for i in range(self.height):
            for j in range(self.width):
                if self.catchment[i,j] > 0.0:
                    if self.density_grid[i,j] > 0.0:
                        #if self.elevation_grid[i,j] <=1300: # run only with downstream.
                        self.lonlat.append(np.array([self.model.xmin + (j * self.x_grid + 0.5 * self.x_grid), self.model.ymin + (i * self.y_grid + 0.5 * self.y_grid)], dtype=np.float32)) # lat lon to grid: height-i?
                        self.agent_population.append(self.density_grid[i,j])

        self.lonlat = np.array(self.lonlat)
        self.agent_population = np.asarray(self.agent_population)
        self.n = len(self.lonlat)

        self.coordinates = np.zeros((self.n, 2), dtype=int)
        self.coordinates[:, 0] = self.location_index(self.lonlat)[0][:]
        self.coordinates[:, 1] = self.location_index(self.lonlat)[1][:] 

        self.livestock_coords = self.coordinates.copy()
        
        self.distance_to_neighbours = self.Distance()

        assert self.lonlat[:,0].min() >= self.model.xmin
        assert self.lonlat[:,0].max() <= self.model.xmax
        assert self.lonlat[:,1].min() >= self.model.ymin
        assert self.lonlat[:,1].max() <= self.model.ymax

    @property
    def activation_order(self):
        return np.arange(self.n, dtype=np.int32)

    def _initiate_attributes(self):
        '''
        Initiate attributes of agents
        '''

        self.age = np.maximum(np.random.normal(45, 13, size=self.n),0) # age of household head
        self.edu = np.maximum(np.random.normal(9, 3, size=self.n),0) # years of education
        self.assets = np.maximum(np.random.normal(100, 300, size=self.n),0) # assets in USD
        self.HH_size = np.maximum(np.random.normal(6, 3, size=self.n),0) # household size
        
        self.adapt_measure_0 = np.random.randint(2, size = self.n) # Migration 
        self.adapt_measure_1 = np.random.randint(2, size = self.n) # Different livestock (goats)
        self.adapt_measure_2 = np.random.randint(2, size = self.n) # Different crop (casava)
        self.adapt_measure_3 = np.random.randint(2, size = self.n) # Irrigation
       
        self.receive_extension = np.random.randint(2, size=self.n) # 0 = no extension services, 1 = extension services
        self.gender = np.random.randint(0, 2, size=self.n) # 0 = male,  1 = female household head

        self.elevation = self.model.data.elevation.sample_coords(self.lonlat)
        self.land_cover_agent = self.model.data.land_cover.sample_coords(self.lonlat)
        
        self.land_size = np.maximum(np.random.normal(0.6, 0.3, size=self.n),0.1) #np.where(self.land_cover_agent == 40,np.maximum( land size [ha] -> [km2]
        self.land_size = np.minimum(self.land_size, (self.x_grid * self.y_grid / 100 / 100 /self.agent_population)) #land size [m2]-> [ha]. Can have a maximum land_size of the # of agents divided in grid cell
        self.crop = np.random.randint(1, 25, size=self.n)
        self.nr_livestock = np.zeros((self.n, 2))
        self.nr_livestock[:, 0] = np.random.randint(1, 40, size=self.n) # Cows
        self.nr_livestock[:, 1] = np.random.randint(1, 40, size=self.n) # Goats
        self.off_farm_income = np.maximum(np.random.normal(100, 250, size=self.n),0) # off-farm income in a year 1200 +- 500 USD
        self.other_expenditures = np.maximum(np.random.normal(20, 50, size=self.n),0)* self.HH_size
        self.expenditures_crops = 110 * self.land_size #110 dollars/hectare
        self.expenditures_livestock = 20 * np.sum(self.nr_livestock, axis = 1) #20 dollars/livestock

        # PROTECTION MOTIVATION THEORY PARAMETERS/VARIABLES
        self.selfefficacy = np.zeros((self.n, 4))
        self.selfefficacy[:,0] = np.random.uniform(0, 1, size=self.n)
        self.selfefficacy[:,1] = np.random.uniform(0, 1, size=self.n)
        self.selfefficacy[:,2] = np.random.uniform(0, 1, size=self.n)
        self.selfefficacy[:,3] = np.random.uniform(0, 1, size=self.n)
        if self.model.config['general']['sensitivity'] == False:
            self.alpha = np.random.uniform(0.334, 0.666)
        if self.model.config['general']['sensitivity'] == True:
            self.alpha = self.model.config['adaptation']['alpha']
        self.beta  = 1 - self.alpha #alpha + beta = 1
        self.gamma  = np.random.uniform(0.25, 0.5, size=1) #between 0.25 and 0.5
        self.delta  = np.random.uniform(0.25, 0.5, size=1) #between 0.25 and 0.5
        self.epsilon  = 1 - self.gamma - self.delta #gamma + delta + epsilon = 1

        self.IntentionToAdapt_Livestock = np.zeros((self.n))
        self.IntentionToAdapt_Crops = np.zeros((self.n))
        self.CopingAppraisal_Livestock = np.zeros((self.n))
        self.CopingAppraisal_Crops = np.zeros((self.n))
        self.RiskAppraisal_Livestock = np.zeros((self.n))
        self.RiskAppraisal_Crops = np.zeros((self.n))
        self.adaptation_eff = np.random.uniform(0, 1, size=self.n)
        self.damage = np.random.uniform(0, 1, size=self.n)
        self.costperception = np.zeros((self.n, 4))
        self.costperception[:,0] = np.random.uniform(0, 1, size=self.n)
        self.costperception[:,1] = np.random.uniform(0, 1, size=self.n)
        self.costperception[:,2] = np.random.uniform(0, 1, size=self.n)
        self.costperception[:,3] = np.random.uniform(0, 1, size=self.n)
        
        self.risk_perception = np.random.uniform(0, 1, size=self.n)

        self.maize_price = 0.35 # USD/kg
        self.milk_price = 0.20 # USD/litre milk

        if self.model.config['general']['sensitivity'] == False:
            self.adapt_costs = [50,250,50,500]
        if self.model.config['general']['sensitivity'] == True:
            self.adapt_costs = [self.model.config['adaptation']['adapt_costs'], self.model.config['adaptation']['adapt_costs'], self.model.config['adaptation']['adapt_costs'], self.model.config['adaptation']['adapt_costs']]


    def _initiate_socio_hydrology(self):
        # WATER AND LAND USE VARIABLES

        self.at_river = self.model.data.river_network.sample_coords(self.lonlat)
        self.spei = np.zeros(self.n)
        self.elevation_grid = np.flip(self.model.data.elevation.get_data_array(),0)
        self.land_use = np.flip(self.model.data.land_cover.get_data_array(),0) # gives (y,x)
        self.livestock_produce = np.sum(self.nr_livestock, axis = 1)
        self.crop_produce = self.land_size * np.random.uniform(2000, 3000, size=self.n) # land size * average yield/hectare
        
        #Abstraction and irrigation factors
        if self.model.config['general']['sensitivity'] == False:
            self.distribution_abstraction_points = np.random.uniform(0.1, 0.9)
            self.irrigation_demand_factor = np.random.uniform(0.1, 2.0)
            print (self.distribution_abstraction_points, self.irrigation_demand_factor)
        if self.model.config['general']['sensitivity'] == True:
            self.distribution_abstraction_points = self.model.config['adaptation']['abstraction_p']
            self.irrigation_demand_factor = self.model.config['adaptation']['irrigation_demand_factor']
        self.distribution_abstraction_points = 0.1
        self.abstraction_p = np.random.choice([True, False], size=(self.grid), p = [self.distribution_abstraction_points, 1 - self.distribution_abstraction_points])

        #Neigbourhood
        if self.model.config['general']['sensitivity'] == False:
            self.radius_neighbourhood =  np.random.uniform(1000, 10000)
        if self.model.config['general']['sensitivity'] == True:
            self.radius_neighbourhood = self.model.config['adaptation']['neighbourhood_radius']

        self.radius = np.full(self.n, self.radius_neighbourhood/1000)
        self.radius = self.radius.astype(int)
        self.coords_neighbours = self.coords_neighbourhood(self.lonlat, self.radius)

        #Seasons
        self.planting_date = 274 # 1st of October
        self.harvest_date = 360
        self.start_dry_season = 140 #20th of May
        self.end_dry_season = 260

        #Livestock Characteristics
        self.feed_required_cattle = 7*365/2 #(kg/livestock/half year)
        self.feed_residue_cattle = 0.3
        self.net_birth_rate_cattle = 0.15
        self.weight_gain_rate_cattle = 1/self.feed_required_cattle/2

        self.feed_required_goats = 6*365/2 #(kg/livestock/half year)
        self.feed_residue_goats = 0.2
        self.net_birth_rate_goats = 0.25
        self.weight_gain_rate_goats = 1/self.feed_required_goats/2
        
        self.coordinates_max_grass = np.empty(self.n)

        self.distance_migration = np.zeros(self.n)
        
        self.yield_grass = np.maximum(np.random.uniform(1,4, size = self.grid),0) # yield in *1000kg/hectare?
        self.yield_crops = np.maximum(np.random.uniform(1,4, size = self.n),0) # yield in *1000kg/hectare?

        self.grid_livestock_nrs = np.zeros((self.height, self.width))

        # TO AND FROM ENVIRONMENT COMPONENT 

        self.land_use_flat = self.land_use.flatten()
        self.crop_map = np.where(self.land_use_flat == 40, 2, -1) # crop map at -1 if there is no crop, 2 is maize, 11 is cassava

        self.river_network = np.flip(self.model.data.river_network.get_data_array(),0)


    def _initiate_storing_variables(self):

        self.latest_harvest_crops = np.empty((self.n, 10), dtype=np.float32) # array to store harvest memory
        self.latest_harvest_crops[:] = np.NaN
        self.latest_harvest_crops[:,0] = self.land_size.copy() * np.random.uniform(15000, 20000, size=self.n)

        self.latest_harvest_livestock = np.empty((self.n, 10), dtype=np.float32) # array to store harvest memory
        self.latest_harvest_livestock[:] = np.NaN
        self.latest_harvest_livestock[:,0] = np.random.randint(3, 200, size=self.n)

        self.latest_income = np.empty((self.n, 10), dtype=np.float32) # array to store income memory
        self.latest_income[:] = np.NaN
        self.latest_income[:,0] = np.maximum(np.random.normal(150.0,50.0, size = self.n),0) #Random for now
        self.latest_spei = np.empty((self.n, 90), dtype=np.float32)
        self.latest_spei[:] = np.NaN

        self.average_crop_production = np.zeros(int(self.model.n_timesteps/360)) # need to make average
        self.average_livestock_production = np.zeros(int(self.model.n_timesteps/360)) # need to make average
        self.average_adoption_rate = np.zeros(int(self.model.n_timesteps/180)) # need to make average

        self.rainfall = np.zeros(self.grid)
        self.soil_moisture = np.zeros(self.grid)
        self.discharge = np.zeros(self.grid)
        self.groundwater = np.zeros(self.grid)

        self.abstraction_gw_hh = np.zeros(self.grid)
        self.abstraction_gw_irr = np.zeros(self.grid)
        self.abstraction_gw_liv = np.zeros(self.grid)
        self.abstraction_riv_hh = np.zeros(self.grid)
        self.abstraction_riv_irr = np.zeros(self.grid)
        self.abstraction_riv_liv = np.zeros(self.grid)

#------------------------------------------------
#           LOCATION AND NEIGNBOURHOOD
#------------------------------------------------

    @property
    def current_day_of_year(self):
        '''Calculate the current day of the year'''

        start_day_per_month = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        return start_day_per_month[self.model.current_time.month - 1] + self.model.current_time.day

    def location_index(self, pos):
        '''
        Returns a x,y array of the agents pixel locations (integers)

        Args:
            pos: coordinate tuple 

        '''
        gt = self.gt
        return coords_to_pixels(pos, gt)

    def coords_neighbourhood(self, pos, radius):
        
        '''Return a list of cells that are in the neighborhood of a
        certain point (for each farmer). Code based on Mesa: https://github.com/projectmesa/mesa.

        Args:
            pos: coordinate tuple for the neighborhood to get.
            radius: radius, in number of cells, of neighborhood to get.

        Returns:
            A list of coordinate tuples (column (width), row (height)) representing the neighborhood;

        '''
        coordinates_neighbourhood = []
        self.torus = True #otherwise index out of bounds..
        gt = self.gt
        pos = coords_to_pixels(pos, gt)

        for i in range(self.n): 
            co = []
            coordinates_neighbourhood.append(co)
            x, y = pos[0][i], pos[1][i]
            radius = int(self.radius[i])
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    coord = ( x + dx, y + dy)

                    if self.out_of_bounds(coord):
                    #Skip if not a torus and new coords out of bounds.
                        if not self.torus:
                            continue
                        coord = self.torus_adj(coord)
                    co.append(coord)
        return coordinates_neighbourhood

    def values_neighbourhood(self, coords, variable):
        '''
        Get values of a variable at the coordinates (in neighbourhood)

        Args:
            coords: coordinate tuples (x,y)
            variable: array of variable of interest
        '''
        array = []
        for i in range(self.n):
            values = []
            array.append(values)
            for j in range(len(coords[i])): #coords is size of agents, j gives tuple coordinates ()
                values.append(variable[coords[i][j]]) 
        return array

    def max_neighbourhood(self, coords, values):
        '''
        Returns grid position of max value in neighborhood of each agent
        '''
        indices = []
        for i in range(self.n):
            indices.append(coords[i][values[i].index(max(values[i]))])
        return indices

    def out_of_bounds(self, pos):
        '''
        Determines whether position is off the grid, returns the out of
        bounds coordinate. Code by Mesa: https://github.com/projectmesa/mesa.
        '''
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def torus_adj(self, pos):
        '''
        Convert coordinate, handling torus looping. Code by Mesa: https://github.com/projectmesa/mesa.
        '''
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            return pos[0] % self.width, pos[1] % self.height

#----------------------------------------------------
#  DECISION TO ADAPT (PROTECTION MOTIVATION THEORY)
#----------------------------------------------------

    def SelfEfficacy(self, measure):
        '''
        Calculate Self Efficacy for every agent [0-1]. 

        Args:
            measure: what adaptation measure to calculate self efficacy for
            age: age of household head
            edu: number of years education
            HH_size: number of people in household
            Knowledge(): Share of people in social network adopting the measure

        Returns:
            Self Efficacy for every agent, based on above characteristics
        
        '''

        self.selfefficacy[:,measure] = ( 0.25 * np.minimum(np.maximum(30 / self.age, 0),1) ) + ( 0.25 * np.minimum(np.maximum(self.edu / 12, 0),1) ) + ( 0.25 * np.minimum(np.maximum(self.HH_size / 6, 0),1) ) + ( 0.25 * self.Knowledge(measure))
        return self.selfefficacy[:,measure]

    def Knowledge(self, measure):
        'Share of people in social network adopting the measure'

        self.knowledge = np.minimum(np.maximum(self.Neighbours_adopted(measure), 0),1)

        return self.knowledge

    def RiskAppraisal_1_livestock(self, damage):
        'Calculate Risk Appraisal for livestock for every agent'

        self.risk_perception = np.maximum(np.where(self.Drought_SPEI()== 1, damage, np.maximum(self.risk_perception - 0.2, 0)), 0) # minimum risk perception of zero
        
        return self.risk_perception

    def RiskAppraisal_1_crops(self, damage):
        'Calculate Risk Appraisal for crops for every agent'


        self.risk_perception = np.maximum(np.where(self.Drought_SPEI()== 1, damage, np.maximum(self.risk_perception - 0.2, 0)), 0) # minimum risk perception of zero
        
        return self.risk_perception

    def Adaptationcosts(self, measure): # should include time & effort cost as well
        'Calculate relative Adapatation Costs for every agent [0-1]'

        self.costperception[:,measure] = 1 - np.maximum(np.minimum((self.adapt_costs[measure]) / (self.assets + 0.001), 1), 0)

        return self.costperception[:,measure]

    def AdaptationEfficacy_crops(self, measure):
        '''
        Calculate relative Adapatation Efficavy for every agent [0-1]. 
        Expected production gain is depended on whether an agent receives extension services/training/information.
        '''
    
        pot_harvest_neighbours_crops = self.Pot_Harvest_Neighbours_crops(measure, self.land_size)  # absolute yield that can be gained extra
        pot_harvest_neighbours_irrigation = self.Pot_Harvest_Neighbours_crops(measure, self.land_size)

        pot_harv_crops = self.Pot_Harvest_Crops() 
        pot_harv_irrigation = self.Pot_Harvest_Irrigation() 

        for i in range(self.n):
            pot_harvest_crops = pot_harv_crops - np.maximum(np.nanmean(self.latest_harvest_crops),0) # from gained total to gained extra
            pot_harvest_irrigation = pot_harv_irrigation - np.maximum(np.nanmean(self.latest_harvest_crops),0)
            if self.receive_extension[i] == 0:
                if measure == 2: # change crop types
                    if np.nanmean(self.latest_harvest_crops[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_crops[i], 1) ,0)
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_crops[i] / np.nanmean(self.latest_harvest_crops[i]) ,1) ,0)
                if measure == 3: # irrigation
                    if np.nanmean(self.latest_harvest_crops[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_irrigation[i], 1) ,0)
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_irrigation[i] / np.nanmean(self.latest_harvest_crops[i]) ,1),0)

            if self.receive_extension[i] == 1:
                if measure == 2: # change crop types
                    if np.nanmean(self.latest_harvest_crops[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_crops[i],1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_crops[i] / np.nanmean(self.latest_harvest_crops[i])),1),0)
                if measure == 3: # irrigation
                    if np.nanmean(self.latest_harvest_crops[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_irrigation[i]),1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_irrigation[i] / np.nanmean(self.latest_harvest_crops[i])),1),0)

        assert (self.adaptation_eff >= 0).all()
        assert (self.adaptation_eff <= 1).all()
        assert (self.adaptation_eff != np.nan).all()

        return self.adaptation_eff

    def AdaptationEfficacy_livestock(self, measure):
        '''
        Calculate relative Adapatation Efficacy for every agent [0-1]. 
        Expected production gain is depended on whether an agent receives extension services/training/information.
        '''
        
        pot_harvest_neighbours_migration = self.Pot_Harvest_Neighbours_livestock(measure, np.sum(self.nr_livestock, axis = 1))  # absolute number of livestock that can be gained
        pot_harvest_neighbours_livestock = self.Pot_Harvest_Neighbours_livestock(measure, np.sum(self.nr_livestock, axis = 1)) 
        
        pot_harvest_migration = self.Pot_Harvest_Migration() - np.nanmean(self.latest_harvest_livestock) # from gained total to gained extra
        pot_harvest_livestock = self.Pot_Harvest_Livestock() - np.nanmean(self.latest_harvest_livestock)

        for i in range(self.n):
            if self.receive_extension[i] == 0:
                if measure == 0: # migration
                    if np.nanmean(self.latest_harvest_livestock[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_neighbours_migration[i]),1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_migration[i] / np.nanmean(self.latest_harvest_livestock[i]) ,1),0)
                if measure == 1: # change livestock types
                    if np.nanmean(self.latest_harvest_livestock[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_neighbours_livestock[i]),1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_neighbours_livestock[i] / np.nanmean(self.latest_harvest_livestock[i]) ,1),0)

            if self.receive_extension[i] == 1:
                if measure == 0: # migration
                    if np.nanmean(self.latest_harvest_livestock[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_migration[i]),1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_migration[i] / np.nanmean(self.latest_harvest_livestock[i]),1),0)
                if measure == 1: # change livestock types
                    if np.nanmean(self.latest_harvest_livestock[i]) == 0.0:
                        self.adaptation_eff[i] = np.maximum(np.minimum((pot_harvest_livestock[i]),1),0) # cannot divide by zero
                    else:
                        self.adaptation_eff[i] = np.maximum(np.minimum(pot_harvest_livestock[i] / np.nanmean(self.latest_harvest_livestock[i]),1),0)

        assert (self.adaptation_eff >= 0).all()
        assert (self.adaptation_eff <= 1).all()
        assert (self.adaptation_eff != np.nan).all()

        return self.adaptation_eff

    def Pot_Harvest_Neighbours_crops(self, measure, relative):
        '''
        Rely on neighbours to obtain information on adaptation efficacy: expected production gain was estimated as the difference 
        between the production of neighboring households that had already adopted a specific measure and the households’ own current production.

        Args:
            measure: measure to calculate the potential for [integer]
            relative: land_size of agent to enable comparison among agents.
        '''

        yields = np.nanmean(self.latest_harvest_crops, axis = 1)

        yield_neighbours = self.Neighbours_adopted_attributes(measure, yields, relative) * self.land_size

        pot_harvest = np.maximum(yield_neighbours - yields, 0) # minimum zero
        return pot_harvest

    def Pot_Harvest_Neighbours_livestock(self, measure, relative):
        '''Rely on neighbours to obtain information on adaptation efficacy: expected production gain 
            was estimated as the difference between the production of neighboring households that 
            had already adopted a specific measure and the households’ own current production.'''
        
        yields = np.nanmean(self.latest_harvest_livestock, axis = 1)
        
        relative_yield = self.Neighbours_adopted_attributes(measure, yields, relative)

        yield_neighbours =  relative_yield* np.sum(self.nr_livestock, axis = 1) # now you have to multiply again to get to actual values (relative to farmer)
        
        pot_harvest = np.maximum(yield_neighbours - yields , 0) # minimum zero
        return pot_harvest

    def Pot_Harvest_Migration(self):
        """Calculate grass yield at other location (within vision). Calculate what the livestock production would be if you would move there (different grass yield)"""

        self.coordinates_max_grass = self.max_neighbourhood(self.coords_neighbours, self.values_neighbourhood(self.coords_neighbours, self.model.abc.grass_yield()))

        max_grass_yield = np.zeros(self.n)

        grass_yield_grid = self.model.abc.grass_yield().reshape(self.height,self.width)

        for i in range(self.n):
            max_grass_yield[i] = grass_yield_grid[self.coordinates_max_grass[i][1],self.coordinates_max_grass[i][0]] 

        pot_harvest = self.model.abc.Livestock_production(max_grass_yield, self.nr_livestock.sum(axis=1), self.nr_livestock[:,0], self.feed_required_cattle, self.feed_residue_cattle, self.net_birth_rate_cattle, self.weight_gain_rate_cattle) + self.model.abc.Livestock_production(max_grass_yield,self.nr_livestock.sum(axis=1), self.nr_livestock[:,1], self.feed_required_goats, self.feed_residue_goats, self.net_birth_rate_goats, self.weight_gain_rate_goats)
        return pot_harvest

    def Pot_Harvest_Livestock(self):
        '''Calculate the livestock procduction in case you have goats only'''

        grass_yield = self.model.abc.grass_yield()[self.coordinates[:,1],self.coordinates[:,0]] # should be only when you are at grass location...
        pot_harvest = self.model.abc.Livestock_production(grass_yield,self.nr_livestock.sum(axis=1), self.nr_livestock[:,0], self.feed_required_goats , self.feed_residue_goats, self.net_birth_rate_goats, self.weight_gain_rate_goats) + self.model.abc.Livestock_production(grass_yield,self.nr_livestock.sum(axis=1), self.nr_livestock[:,1], self.feed_required_goats, self.feed_residue_goats, self.net_birth_rate_goats, self.weight_gain_rate_goats)
        return pot_harvest

    def Pot_Harvest_Crops(self):
        '''Change crop type: change crop_map on that location'''

        crop_map = np.empty(self.grid, dtype=np.int32) 
        crop_map.fill(11) # cassava 
        type_of_management = self.adapt_measure_3[:] # irrigated field or not
        type_of_crop = np.ones(self.n, dtype=np.int32)

        pot_harvest = self.model.abc.Crop_production(crop_map, type_of_management, type_of_crop, self.coordinates, self.n, self.land_size)
        return pot_harvest

    def Pot_Harvest_Irrigation(self):
        '''Irrigation: Get irrigated yield instead of rainfed yield'''

        crop_map = self.crop_map

        type_of_management = np.ones(self.n, dtype=np.int32) 
        type_of_crop = self.adapt_measure_2[:] # type of crop
        pot_harvest = self.model.abc.Crop_production(crop_map, type_of_management, type_of_crop,self.coordinates, self.n,  self.land_size)

        return pot_harvest

    def Drought_SPEI(self):
        '''
        Returns whether an agent has experience drought in the last 90 days (3 months), yes (1) or no (0) 
        '''

        drought = np.where(self.spei<= -1, 1, 0)

        return drought

    def Harvest_crops(self):
        '''
        Return production of crops in kg
        '''

        self.yield_crops = self.model.abc.yield_crops

        crop_map = self.crop_map # flat
        type_of_management = np.zeros(self.n, dtype=np.int32)
        type_of_crop = self.adapt_measure_2[:] # either 0 (maize) or 1 (casava)
        self.crop_produce = self.model.abc.Crop_production(crop_map, type_of_management, type_of_crop, self.coordinates, self.n, self.land_size) # # land_size [hectare] # yield = [*1000 kg / hectare] 

        self.latest_harvest_crops[:, 1:] = self.latest_harvest_crops[:, 0:-1]
        self.latest_harvest_crops[:, 0] = self.crop_produce

        #For model output:
        self.average_crop_production[1:] = self.average_crop_production[0:-1]
        self.average_crop_production[0] = np.mean((self.crop_produce * self.agent_population)/ np.sum(self.agent_population))

        assert (self.crop_produce != np.nan).all()
        assert (self.crop_produce >= 0).all()
        
        return self.crop_produce

    def Grid_sum_livestock(self): 
        ''' Calculate the sum of livstock within every grid cell'''
        
        for i in range(self.n):
            self.grid_livestock_nrs[self.livestock_coords[i, 1],self.livestock_coords[i, 0]] += np.sum(self.nr_livestock[i])
        return self.grid_livestock_nrs

    def Harvest_livestock(self):
        '''
        Return production of livestock in number of livestock
        '''
        self.yield_grass = self.model.abc.grass_yield().reshape(self.height,self.width)

        grass_yield_household = self.yield_grass[self.livestock_coords[:, 1],self.livestock_coords[:, 0]]
        
        sum_livestock_grid_cell = np.zeros(self.n)
        
        for i in range (self.n):
            sum_livestock_grid_cell[i] = self.grid_livestock_nrs[self.livestock_coords[i, 1],self.livestock_coords[i, 0]]

        # livestock_production [# of livestock]
        cows = self.model.abc.Livestock_production(grass_yield_household, sum_livestock_grid_cell, self.nr_livestock[:,0], self.feed_required_cattle, self.feed_residue_cattle, self.net_birth_rate_cattle, self.weight_gain_rate_cattle)
        goats =+ self.model.abc.Livestock_production(grass_yield_household, sum_livestock_grid_cell, self.nr_livestock[:,1], self.feed_required_goats, self.feed_residue_goats, self.net_birth_rate_goats, self.weight_gain_rate_goats)
        self.livestock_produce =  cows + goats

        self.nr_livestock[:, 0] = cows
        self.nr_livestock[:, 1] = goats

        self.latest_harvest_livestock[:, 1:] = self.latest_harvest_livestock[:, 0:-1]
        self.latest_harvest_livestock[:, 0] = self.livestock_produce

        #For model output:
        self.average_livestock_production[1:] = self.average_livestock_production[0:-1]
        self.average_livestock_production[0] = np.mean((self.livestock_produce * self.agent_population)/ np.sum(self.agent_population))

        assert (self.livestock_produce != np.nan).all()
        assert (self.livestock_produce >= 0).all()

        return self.livestock_produce

    def Income_Crops(self):
        '''
        Calculate income based on crop production
        '''
        earnings_crops = np.maximum((self.Harvest_crops())- (103 * self.HH_size),0)  * self.maize_price # harvest[*1000 kg] maize_price[USD/kg] - 103kg per hh member consumed
        earnings = earnings_crops

        self.latest_income[:, 1:] = self.latest_income[:, 0:-1]
        self.latest_income[:, 0] = earnings

        return earnings

    def Income_Livestock(self):
        '''
        Calculate income based on livestock production
        '''
        earnings_livestock = np.maximum(((self.Harvest_livestock() * 365) - (100 * self.HH_size * 365)), 0)  * self.milk_price # 1 liter per day.. - 100 l/ HH member
        earnings = earnings_livestock

        self.latest_income[:, 1:] = self.latest_income[:, 0:-1]
        self.latest_income[:, 0] = earnings

        return earnings

    def Damage_livestock(self):
        '''
        Determine damage by comparing production of livestock to average production of last 10 years
        '''
        mean = np.nanmean(self.latest_harvest_livestock, axis=1)
        self.damage = np.where(mean > 0, ((mean - self.livestock_produce) / mean), 0) 
        return self.damage

    def Damage_crops(self):
        '''
        Determine damage by comparing production of crops to average production of last 10 years
        '''

        mean = np.nanmean(self.latest_harvest_crops, axis=1)
        damage =  np.where(mean > 0, ((mean - self.crop_produce) / mean), 0) 
        return damage

    def distance_livestock_migration(self, coordinates, livestock_coords):
        ''' Calculate the distance an agents has travelled with their livestock [km]'''
        distance = np.sqrt(abs((coordinates[:, 0] - livestock_coords[:,0])**2 + (coordinates[:,1] - livestock_coords[:,1])**2) )
        return distance

    def Protection_Motivation_Theory(self, measure, damage, risk_appraisal, spei):
        '''
        
        Calculate Intention to Adapt (for every measure) based on Protection Motivation Theory

        Args:
            measure: what adaptation measure to calculate intention to adopt for
            CopingAppraisal: coping apraisal based on self efficacy, adaptation efficacy and adaptation costs
            RiskAppraisal: risk apraisal based on memory (risk_appraisal) and drought damage

        Returns:
            Intention to adapt for a measure for every agent
        
        '''
        self_efficacy = self.SelfEfficacy(measure)
        adaptation_costs = self.Adaptationcosts(measure)
        drought_damage = np.maximum(spei * (1-np.exp(-damage)), 0)

        if measure == 0 or measure ==1:
            self.adaptation_efficacy = self.AdaptationEfficacy_livestock(measure)
            self.RiskAppraisal_Livestock = np.minimum(np.maximum(risk_appraisal[:, 0] + drought_damage - 0.125 * risk_appraisal[:, 1], 0),1) #drought is expressed as SPEI, risk perception based on previous timestep
            self.CopingAppraisal_Livestock = self.gamma * self_efficacy + self.delta * self.adaptation_efficacy + self.epsilon * (1 - adaptation_costs)
            self.IntentionToAdapt_Livestock = self.alpha * self.RiskAppraisal_Livestock + self.beta * self.CopingAppraisal_Livestock

        if measure == 2 or measure ==3:
            self.adaptation_efficacy = self.AdaptationEfficacy_crops(measure)
            self.RiskAppraisal_Crops = np.minimum(np.maximum(risk_appraisal[:, 0] + drought_damage - 0.125 * risk_appraisal[:, 1], 0),1) #drought is expressed as SPEI, risk perception based on previous timestep
            self.CopingAppraisal_Crops = self.gamma * self_efficacy + self.delta * self.adaptation_efficacy + self.epsilon * (1 - adaptation_costs)
            self.IntentionToAdapt_Crops = self.alpha * self.RiskAppraisal_Crops + self.beta * self.CopingAppraisal_Crops

        assert (self.IntentionToAdapt_Livestock >= 0).all()
        assert (self.IntentionToAdapt_Livestock != np.nan).all()
        assert (self.IntentionToAdapt_Crops >= 0).all()
        assert (self.IntentionToAdapt_Crops != np.nan).all()
        return self.IntentionToAdapt_Livestock, self.IntentionToAdapt_Crops

    def Decision_to_Adapt(self):
        '''Determines whether people adapt. If you can pay, and intention to adapt is greater than random number -> update assets, location, measure.'''
        Random_threshold = np.random.uniform(0, 1, size=self.n)

        abstraction_points = self.abstraction_p.reshape(self.height, self.width) 

        # DECISION MADE BEGINNING DRY PERIOD
        if self.current_day_of_year == self.start_dry_season:
            spei =  self.Drought_SPEI()
            damage = self.Damage_livestock()
            risk_appraisal = np.random.uniform(0,1,(self.n,2))
            risk_appraisal[:, 1:] = risk_appraisal[:, 0:-1] # previous timestep
            risk_appraisal[:, 0] = self.RiskAppraisal_1_livestock(damage) # curent timestep
            PMT_0 = self.Protection_Motivation_Theory(0,damage, risk_appraisal, spei)[0]
            PMT_1 = self.Protection_Motivation_Theory(1,damage, risk_appraisal, spei)[0]

            assert (PMT_0 != np.nan).all()
            assert (PMT_0 >= 0).all()

            assert (PMT_1 != np.nan).all()
            assert (PMT_1 >= 0).all()

            costs_migration = self.adapt_costs[0]
            costs_livestock = self.adapt_costs[1]

            likelihood_to_adapt_0 = ( 1 -  ( ( 1 - PMT_0 ) ** ( 1 / 1 ) ) ) 
            likelihood_to_adapt_1 = ( 1 -  ( ( 1 - PMT_1 ) ** ( 1 / 10 ) ) ) 

            self.adapt_measure_0[:] = 0 # after one year the measure expires

            for i in range(self.n):

                #[0] = MIGRATION
                if costs_migration < self.assets[i] : #and PMT_0[i] > PMT_1[i]:
                    if likelihood_to_adapt_0[i] > Random_threshold[i]:
                        self.adapt_measure_0[i] = 1
                        
                        self.livestock_coords[i, 0] = self.coordinates_max_grass[i][1] # should actually differ per season -> rainy season they go home
                        self.livestock_coords[i, 1] = self.coordinates_max_grass[i][0]
                        
                        self.assets[i] = self.assets[i] - costs_migration

                #[1] = LIVESTOCK BREED CHANGE
                elif costs_livestock < self.assets[i] : #and PMT_1[i] > PMT_0[i]
                    
                    if likelihood_to_adapt_1[i] > Random_threshold[i]:
                        self.assets[i] = self.assets[i] - costs_livestock
                        self.adapt_measure_1[i] = 1
                        self.nr_livestock[i, 1] += int(self.nr_livestock[i, 0]) #remove cows, and add to goats
                        self.nr_livestock[i, 0] = 1
            
            self.distance_migration = self.distance_livestock_migration(self.coordinates, self.livestock_coords)
            
        # DECISIONS MADE JUST BEFORE (SHORT) RAINY SEASON
        if self.current_day_of_year == (self.planting_date - 1):
            spei =  self.Drought_SPEI()
            damage = self.Damage_crops()
            risk_appraisal = np.random.uniform(0,1,(self.n,2))
            risk_appraisal[:, 1:] = risk_appraisal[:, 0:-1] # previous timestep
            risk_appraisal[:, 0] = self.RiskAppraisal_1_crops(damage) # curent timestep
            PMT_2 = self.Protection_Motivation_Theory(2,damage, risk_appraisal,spei)[1]
            PMT_3 = self.Protection_Motivation_Theory(3,damage, risk_appraisal, spei)[1]

            assert (PMT_2 != np.nan).all()
            assert (PMT_2 >= 0).all()

            assert (PMT_3 != np.nan).all()
            assert (PMT_3 >= 0).all()
            
            costs_crops = self.adapt_costs[2]
            costs_irrigation = self.adapt_costs[3]

            likelihood_to_adapt_2 = ( 1 -  ( ( 1 - PMT_2 ) ** ( 1 / 1 ) ) ) # measure is for one year
            likelihood_to_adapt_3 = ( 1 -  ( ( 1 - PMT_3 ) ** ( 1 / 10 ) ) ) # measure is for ten years

            self.adapt_measure_2[:] = 0 # after one year the measure expires

            crop_change_map = self.crop_map.reshape(self.height,self.width)

            for i in range(self.n):
                #[2] = CROP CHANGE
                if costs_crops < self.assets[i]: #and PMT_2[i] > PMT_3[i]
                    
                    if likelihood_to_adapt_2[i] > Random_threshold[i]:
                        self.assets[i] = self.assets[i] - costs_crops 
                        self.adapt_measure_2[i] = 1
                        # Set new kc values by altering crop map:
                        #to ABM connector
                        self.land_use[self.coordinates[i,1],self.coordinates[i,0]] = 40 # representing certain 
                        crop_change_map[self.coordinates[i,1],self.coordinates[i,0]] = 11 # set cropmap to casava
                
                #[3] = IRRIGATION CHANGE
                if costs_irrigation < self.assets[i]: #and PMT_3[i] > PMT_2[i]
                    
                    if self.at_river[i] >= 1 or abstraction_points[self.coordinates[i,1],self.coordinates[i,0]] == True: #NOT WORKING YET
                        if likelihood_to_adapt_3[i] > Random_threshold[i]:
                            self.assets[i] = self.assets[i] - costs_irrigation
                            self.adapt_measure_3[i] = 1

            self.crop_map = crop_change_map.flatten() # communicated to DRYP
        
        # #END OF DRY SEASON SEASON: collect income livestock
        if self.current_day_of_year == (self.end_dry_season + 1): 
            # in Income() harvest of livestock and crops are calculated
            income_livestock = self.Income_Livestock()
            self.assets += np.maximum(income_livestock - self.expenditures_livestock, 0)
            self.livestock_coords = self.coordinates.copy() # go back home
            
        # #END OF RAINY SEASON: collect income crops
        if self.current_day_of_year == (self.harvest_date + 1): 
            # in Income() harvest crops are calculated, and assest are updated
            income_crops = self.Income_Crops()
            self.assets +=  np.maximum(income_crops + self.off_farm_income - self.expenditures_crops - self.other_expenditures, 0) # update assets because of income/loss of harvest
            
        else:
            pass

#------------------------------------------------
#        SOCIAL NETWORK (NEIGHBOURS)
#------------------------------------------------
        
    def Distance(self):
        '''Distance matrix for every farmer.
        Returns a list of distance to every other agent'''
        distance_matrix = np.ones((self.n, self.n-1))

        for i in range(self.n):
            x_self = np.full(self.n, self.lonlat[i, 0])
            y_self = np.full(self.n, self.lonlat[i, 1])
            distance = np.sqrt(abs((x_self - self.lonlat[:, 0])**2 + (y_self - self.lonlat[:, 1])**2) )
            non_zeros = np.delete(distance, i) # do not include agent itself
            distance_matrix[i,:] = non_zeros

        return  distance_matrix

    def Neighbours_adopted(self, measure):
        '''Calculate the share of people in your neighbourhood that have adapted a measure'''
        neighbours = self.distance_to_neighbours < self.radius_neighbourhood # neighbours are people withing range
        adopted = np.zeros((self.n, self.n-1))

        for i in range(self.n):
            if measure == 0:
                measure_new = np.delete(self.adapt_measure_0[:], i) # delete agent itself
            if measure == 1:
                measure_new = np.delete(self.adapt_measure_1[:], i)
            if measure == 2:
                measure_new = np.delete(self.adapt_measure_2[:], i) 
            if measure == 3:
                measure_new = np.delete(self.adapt_measure_3[:], i)

            adopted[i,:] = measure_new * neighbours[i,:] # whether someone has adopted an adaptation measure (true/false)
        
        share_neighbours_adopted = np.nanmax(np.sum(adopted, axis = 1) / np.sum(neighbours, axis= 1), 0)

        return  share_neighbours_adopted

    def Neighbours_adopted_attributes(self, measure, attribute, relative):
        '''Calculate a the average of the values of agents' attribute (e.g. crop production) within neighbourhood. 
        Make it relative to the characteristics of the other agents (e.g. per hectare, or number of livestock)'''
        
        neighbours = self.distance_to_neighbours < self.radius_neighbourhood # neighbours are people withing range
        average = np.zeros(self.n)

        for i in range(self.n):
            if measure == 0:
                measure_new = np.delete(self.adapt_measure_0[:], i) # delete agent itself
            if measure == 1:
                measure_new = np.delete(self.adapt_measure_1[:], i)
            if measure == 2:
                measure_new = np.delete(self.adapt_measure_2[:], i) 
            if measure == 3:
                measure_new = np.delete(self.adapt_measure_3[:], i) 
            attribute_new = np.delete(attribute/relative[i], i)
            adopted_attributes = measure_new * attribute_new * neighbours[i,:]
            if relative[i] == 0.0:
                average[i] = 0.0
            else:
                average[i]= adopted_attributes[adopted_attributes > 0].mean()
             
        average[np.isnan(average)] = 0
        average_neighbours_adopted = average
        assert (average_neighbours_adopted != np.nan).all()
        assert (average_neighbours_adopted >= 0).all()

        return  average_neighbours_adopted

    
#------------------------------------------------
#                 STEP FUNCTION
#------------------------------------------------

    def load_timestep_data(self):
        '''Update data of timestep'''
        
        self.spei = self.model.data.spei.sample_coords(self.lonlat, self.model.current_time)
        self.latest_spei[:, 1:] = self.latest_spei[:, 0:-1]
        self.latest_spei[:, 0] = self.spei

        self.soil_moisture = self.model.swb.tht_dt
        self.discharge = self.model.ro.dis_dt
        self.groundwater = self.model.gw.wte_dt

    def step(self):
        '''Take a step in the decision module'''
        self.load_timestep_data()
        if self.model.config['general']['BAU'] == False: # run when people adapt (not bussiness as usual)
            self.Decision_to_Adapt()
        if self.current_day_of_year <365 and self.model.config['general']['BAU'] == True: # If business as usual, only run first year for initialisation. 
            self.Decision_to_Adapt()

    def add_agents(self):
        raise NotImplementedError

class Agents(AgentBaseClass):
    def __init__(self, model):
        self.model = model
        self.agent_types = []
        self.farmers = Farmers(model, self) 

    def step(self):
        self.farmers.step()
