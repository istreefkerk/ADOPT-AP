import numpy as np
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import coords_to_pixels
import pandas as pd
import os
import pyproj
import stats


class Farmers(AgentBaseClass):
    '''
    Class for the farmers in the model. The farmers are the agents that are
    simulated in the model. They have a number of attributes and methods that
    define their behavior and interactions with other agents and the environment.
    '''

#------------------------------------------------
#             INITIATION PROCEDURE
#------------------------------------------------

    def __init__(self, model, agents):
        """
        Initialize the Farmers agent.

        Args:
            model: The main model object containing configuration and data.
            agents: Reference to the Agents container class.
        """
        self.model = model
        self.agents = agents
        self._initiate_agents()
        
    def _initiate_agents(self):
        """
        Run all agent initialization steps:
        - Locations
        - Attributes
        - Socio-hydrology
        - Storing variables
        """
        self.model.data._initiate_locations()
        self.model.data._initiate_attributes()
        self.model.data._initiate_socio_hydrology()
        self.model.data._initiate_storing_variables()

    def _load_initial_state(self):
        """
        Load the initial state for the agent at the start of the simulation.
        """
        self.load_timestep_data()

    @property
    def current_day_of_year(self):
        """
        Calculate the current day of the year.

        Returns:
            int: Day of the year (1-366).
        """

        start_day_per_month = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
        return start_day_per_month[self.model.current_time.month - 1] + self.model.current_time.day


#----------------------------------------------------
#  DECISION TO ADAPT (PROTECTION MOTIVATION THEORY)
#----------------------------------------------------

    def SelfEfficacy(self, measure):
        '''
        Calculate Self Efficacy for every agent [0-1]. 

        Args:
            measure: what adaptation measure to calculate self efficacy for
            edu: number of years education
            Knowledge(): Share of people in social network adopting the measure
            forecast_information: whether the agent receives forecast information
            crop_livelihood: share of crop livelihood   
            livestock_livelihood: share of livestock livelihood
            receive_aid: whether the agent receives aid

        Returns:
            Self Efficacy for every agent, based on above characteristics
        
        '''
        
        if measure == 0: # migration
            self.selfefficacy[:,measure] = (0.85 + ( 0.07 * self.edu  + 0.42 * self.Knowledge(measure) + 0.33 * self.forecast_information + 0.36 * self.crop_livelihood + 0.97 * self.livestock_livelihood))
        if measure == 1: # livestock types
            self.selfefficacy[:,measure] = (1.29 + ( 0.11 * self.edu  + 0.33 * self.Knowledge(measure) + 0.30 * self.receive_aid + 0.73 * self.crop_livelihood + 0.89 * self.livestock_livelihood))
        if measure == 2: # crop types
            self.selfefficacy[:,measure] = (1.47 + ( 0.25 * self.Knowledge(measure) + 0.18 * self.receive_aid + 0.39 * self.forecast_information + 1.16 * self.crop_livelihood))
        if measure == 3: # irrigation
            self.selfefficacy[:,measure] = (1.06 + ( 0.08 * self.edu + 0.44 * self.Knowledge(measure) + 0.35 * self.forecast_information  + 1.21 * self.crop_livelihood))
        if measure == 4: # water harvesting
            self.selfefficacy[:,measure] = (1.47 + ( 0.10 * self.edu + 0.63 * self.crop_livelihood + 0.31 * self.livestock_livelihood)) 
        if measure == 5: # soil conservation
            self.selfefficacy[:,measure] = (1.32 + ( 0.42 * self.Knowledge(measure) + 0.95 * self.crop_livelihood + 0.45 * self.forecast_information + 0.29 * self.receive_aid )) 

        return self.selfefficacy[:,measure] / 5

    def Knowledge(self, measure):
        """
        Share of people in social network adopting the measure.

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Knowledge values for all agents.
        """

        self.knowledge = np.minimum(np.maximum(self.Neighbours_adopted(measure), 0),1)

        return self.knowledge

    def RiskAppraisal_1_livestock(self, damage):
        """
        Calculate Risk Appraisal for crops for every agent.

        Args:
            damage (np.ndarray): Damage values.

        Returns:
            np.ndarray: Risk perception values.
        """

        self.risk_perception = np.maximum((1-np.exp(-damage)), 0) # minimum risk perception of zero
        
        return self.risk_perception

    def RiskAppraisal_1_crops(self, damage):
        'Calculate Risk Appraisal for crops for every agent'

        self.risk_perception = np.maximum((1-np.exp(-damage)), 0) # minimum risk perception of zero
        
        return self.risk_perception

    def Adaptationcosts(self, measure):
        """
        Calculate relative Adaptation Costs for every agent [0-1].

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Cost perception values for all agents.
        """

        if measure == 0: # migration

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.159, 0.052, 0.198, 0.237, 0.354]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.179,0.156,0.217,0.142,0.306]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.159, 0.052, 0.198, 0.237, 0.354])))
            
        if measure == 1: # livestock types

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.019, 0.015, 0.060, 0.360, 0.546]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0, 0.040, 0.169, 0.329, 0.462]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.019, 0.015, 0.060, 0.360, 0.546])))

        if measure == 2: # crop types

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.052, 0.082, 0.232, 0.408, 0.226]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.022, 0.062, 0.093, 0.332, 0.491]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.052, 0.082, 0.232, 0.408, 0.226])))

        if measure == 3: # irrigation

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.016, 0.023, 0.074, 0.222, 0.665]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.040, 0.035, 0.015, 0.143, 0.767]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.016, 0.023, 0.074, 0.222, 0.665])))
            
        if measure == 4: # water harvesting

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.027, 0.073, 0.203, 0.303, 0.394]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.009, 0.049, 0.112, 0.268, 0.562]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.027, 0.073, 0.203, 0.303, 0.394])))

        if measure == 5: # soil conservation

            self.costperception[:,measure] = np.where((self.climate_zone == 53), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.124, 0.086, 0.244, 0.321, 0.225]), np.where(((self.climate_zone == 62) | (self.climate_zone == 71)), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.020, 0.052, 0.076, 0.314, 0.538]), stats.fixed_choice(self.seeds, [5,4,3,2,1], size=(self.n), p = [0.124, 0.086, 0.244, 0.321, 0.225])))
            

        return self.costperception[:,measure] / 5 # scale to 0-1

    def AdaptationEfficacy_crops(self, measure):
        """
        Calculate relative Adaptation Efficacy for crops for every agent [0-1].

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Adaptation efficacy values.
        """

        # change crop types
        if measure == 2:
            pot_harvest_neighbours_crops =  np.where(np.nanmean(self.latest_harvest_crops, axis = 1) > 0, np.maximum(np.minimum(self.Pot_Harvest_Neighbours_crops(2, self.land_size) / np.nanmean(self.latest_harvest_crops, axis =1), 1), 0), 0) # absolute yield that can be gained extra
            pot_harv_crops = np.minimum(pot_harvest_neighbours_crops + 0.418/5, 1)

            self.adaptation_eff = np.where((self.receive_extension == 0),  
                pot_harvest_neighbours_crops, pot_harv_crops)

        # irrigation
        if measure == 3:
            pot_harvest_neighbours_irrigation = np.where(np.nanmean(self.latest_harvest_crops, axis = 1) > 0, np.maximum(np.minimum(self.Pot_Harvest_Neighbours_crops(3, self.land_size) / np.nanmean(self.latest_harvest_crops, axis =1), 1), 0), 0)
            pot_harv_irrigation = np.minimum(pot_harvest_neighbours_irrigation + 0.75/5, 1) 

            self.adaptation_eff = np.where((self.receive_extension == 0), 
                pot_harvest_neighbours_irrigation, pot_harv_irrigation)

        assert (self.adaptation_eff >= 0).all()
        assert (self.adaptation_eff <= 1).all()
        assert (self.adaptation_eff != np.nan).all()

        return self.adaptation_eff

    def AdaptationEfficacy_livestock(self, measure):
        """
        Calculate relative Adaptation Efficacy for livestock for every agent [0-1].

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Adaptation efficacy values.
        """

        # migration
        if measure == 0:
            pot_harvest_neighbours_migration = np.where((np.nanmean(self.latest_harvest_livestock, axis = 1)) > 0, np.maximum(np.minimum(self.Pot_Harvest_Neighbours_livestock(0, np.sum(self.nr_livestock, axis = 1))/ np.nanmean(self.latest_harvest_livestock, axis = 1), 1),0), 0)  # absolute number of livestock that can be gained
            pot_harvest_migration= np.minimum(pot_harvest_neighbours_migration + 0.041/5, 1)
            self.adaptation_eff = np.where((self.receive_extension == 0), 
                pot_harvest_neighbours_migration , pot_harvest_migration)

        # change livestock types
        if measure == 1:
            pot_harvest_neighbours_livestock = np.where((np.nanmean(self.latest_harvest_livestock, axis = 1)) > 0, np.maximum(np.minimum(self.Pot_Harvest_Neighbours_livestock(1, np.sum(self.nr_livestock, axis = 1)) / np.nanmean(self.latest_harvest_livestock, axis =1), 1), 0), 0)
            pot_harvest_livestock = np.minimum(pot_harvest_neighbours_livestock + 0.759/5, 1) 
            self.adaptation_eff = np.where((self.receive_extension == 0), 
                pot_harvest_neighbours_livestock, pot_harvest_livestock)

        assert (self.adaptation_eff >= 0).all()
        assert (self.adaptation_eff <= 1).all()
        assert (self.adaptation_eff != np.nan).all()

        return self.adaptation_eff

    def AdaptationEfficacy(self, measure):
        """
        Calculate relative Adaptation Efficacy for every agent [0-1].

        Args:
            measure (int): Adaptation measure index.

        Returns:
            np.ndarray: Adaptation efficacy values.
        """
        if measure == 4:

            pot_harvest_neighbours_water_harvesting = np.where(np.nanmean(self.latest_harvest_livestock, axis = 1) > 0, np.minimum(np.maximum(self.livestock_livelihood * np.minimum(np.maximum(self.Pot_Harvest_Neighbours_livestock(4, np.sum(self.nr_livestock, axis = 1))/ np.nanmean(self.latest_harvest_livestock, axis = 1), 0), 1), 0) + np.where(np.nanmean(self.latest_harvest_crops, axis = 1) > 0, self.crop_livelihood * np.minimum(np.maximum(self.Pot_Harvest_Neighbours_crops(4, self.land_size)/ np.nanmean(self.latest_harvest_crops, axis = 1), 0), 1) , 0), 1), 0)
            pot_harvest_water_harvesting = np.minimum(pot_harvest_neighbours_water_harvesting + 0.2934/5, 1) 

            self.adaptation_eff = np.where(self.receive_extension == 0, 
                                pot_harvest_neighbours_water_harvesting, pot_harvest_water_harvesting)

        if measure == 5:

            pot_harvest_neighbours_soil_conservation = np.where(np.nanmean(self.latest_harvest_livestock, axis = 1) > 0, np.minimum(np.maximum(self.livestock_livelihood * np.minimum(np.maximum(self.Pot_Harvest_Neighbours_livestock(5, np.sum(self.nr_livestock, axis = 1))/ np.nanmean(self.latest_harvest_livestock, axis = 1), 0), 1), 0) + np.where(np.nanmean(self.latest_harvest_crops, axis = 1) > 0, self.crop_livelihood * np.minimum(np.maximum(self.Pot_Harvest_Neighbours_crops(5, self.land_size)/ np.nanmean(self.latest_harvest_crops, axis =1), 0), 1) , 0), 1), 0)
            pot_harvest_soil_conservation = np.minimum(pot_harvest_neighbours_soil_conservation + 0.3572/5, 1) 

            self.adaptation_eff = np.where(self.receive_extension == 0, 
                                           pot_harvest_neighbours_soil_conservation, pot_harvest_soil_conservation)

        assert (self.adaptation_eff >= 0).all()
        assert (self.adaptation_eff <= 1).all()
        assert (self.adaptation_eff != np.nan).all()

        return self.adaptation_eff

    def Pot_Harvest_Neighbours_crops(self, measure, relative):
        """
        Estimate potential crop harvest gain from neighbors who adopted a measure.

        Args:
            measure (int): Adaptation measure index.
            relative (np.ndarray): Land size for normalization.

        Returns:
            np.ndarray: Potential harvest gain.
        """

        yields = np.nanmean(self.latest_harvest_crops, axis = 1)

        yield_neighbours = self.Neighbours_adopted_attributes(measure, yields, relative) * self.land_size

        pot_harvest = np.maximum(yield_neighbours - yields, 0) # minimum zero
        return pot_harvest

    def Pot_Harvest_Neighbours_livestock(self, measure, relative):
        """
        Estimate potential livestock production gain from neighbors who adopted a measure.

        Args:
            measure (int): Adaptation measure index.
            relative (np.ndarray): Livestock count for normalization.

        Returns:
            np.ndarray: Potential livestock gain.
        """
        
        yields = np.nanmean(self.latest_harvest_livestock, axis = 1)
        
        relative_yield = self.Neighbours_adopted_attributes(measure, yields, relative)

        yield_neighbours =  relative_yield* np.sum(self.nr_livestock, axis = 1) # multiply again to get to actual production (relative to farmer)
        
        pot_harvest = np.maximum(yield_neighbours - yields , 0) # minimum zero
        return pot_harvest


    def Harvest_crops(self):
        """
        Return production of crops in [kg].

        Returns:
            np.ndarray: Crop production for each agent.
        """

        crop_map = self.crop_map # flat
        type_of_crop = self.adapt_measure_2[:] # either 0 (maize) or 1 (casava) 
        self.crop_production = self.model.abc.Crop_production(crop_map, type_of_crop, self.coordinates, self.n) # land_size [hectare] # yield = [*1000 kg / hectare] 

        self.yield_crops = self.model.abc.yield_crops[0] *1000
        self.crop_produce = self.crop_production * self.land_size # [kg] 
        self.latest_harvest_crops[:, 1:] = self.latest_harvest_crops[:, 0:-1]
        self.latest_harvest_crops[:, 0] = self.crop_produce


        assert (self.crop_produce != np.nan).all()
        assert (self.crop_produce >= 0).all()
        
        return self.crop_produce

    def Grid_sum_livestock(self): 
        """
        Calculate the sum of livestock within every grid cell.

        Returns:
            np.ndarray: 2D grid of livestock numbers.
        """
        
        self.grid_livestock_nrs = np.zeros((self.height, self.width))
        
        for i in range(self.n):
            self.grid_livestock_nrs[self.livestock_coords[i, 1],self.livestock_coords[i, 0]] += np.sum(self.nr_livestock[i])
        return self.grid_livestock_nrs

    def Harvest_livestock(self):
        """
        Return production of livestock in number of livestock.

        Returns:
            tuple: (cows, goats) production arrays.
        """
        self.yield_grass = self.model.abc.grass_yield_.flatten()

        grass_yield_household = self.yield_grass.reshape(self.height,self.width)[self.livestock_coords[:, 1],self.livestock_coords[:, 0]]
        
        sum_livestock_grid_cell = np.zeros(self.n)
        
        self.Grid_sum_livestock()
        
        sum_livestock_grid_cell = self.grid_livestock_nrs[self.livestock_coords[:, 1],self.livestock_coords[:, 0]]

        # livestock_production [# of livestock]
        cows = self.model.abc.Livestock_production(grass_yield_household, sum_livestock_grid_cell, self.nr_livestock[:,0], self.feed_required_cattle, self.feed_residue_cattle, self.net_birth_rate_cattle, self.weight_gain_rate_cattle)
        goats = self.model.abc.Livestock_production(grass_yield_household, sum_livestock_grid_cell, self.nr_livestock[:,1], self.feed_required_goats, self.feed_residue_goats, self.net_birth_rate_goats, self.weight_gain_rate_goats)
        self.livestock_produce =  cows + goats

        self.yearly_harvest_livestock[:, 1] = self.yearly_harvest_livestock[:, 0].copy() # put previous day to column 1
        self.yearly_harvest_livestock[:, 0] = (self.yearly_harvest_livestock[:, 1] * 364 + self.livestock_produce) / 365 # average over the year
        
        if self.current_day_of_year == self.start_dry_season: # save average harvest of last year
            self.latest_harvest_livestock[:, 1:] = self.latest_harvest_livestock[:, 0:-1]
            self.latest_harvest_livestock[:, 0] = self.yearly_harvest_livestock[:, 0]

        assert (self.livestock_produce != np.nan).all()
        assert (self.livestock_produce >= 0).all()

        return cows, goats

    def Income_Crops(self):
        """
        Calculate income based on crop production.

        Returns:
            np.ndarray: Earnings from crops for each agent.
        """
        earnings_crops = np.maximum((self.Harvest_crops())- ( self.food_consumption * self.HH_size),0)  * self.crop_market() # harvest[*1000 kg] maize_price[USD/kg]
        earnings = earnings_crops

        return earnings

    def Damage_livestock(self):
        """
        Determine damage by comparing livestock production to average of last 10 years.

        Returns:
            np.ndarray: Damage values for livestock.
        """
        mean = np.nanmean(self.latest_harvest_livestock, axis=1)
        self.damage = np.where(mean > 0, ((mean - self.livestock_produce) / mean), 0) 
        return self.damage

    def Damage_crops(self):
        """
        Determine damage by comparing crop production to average of last 10 years.

        Returns:
            np.ndarray: Damage values for crops.
        """

        mean = np.nanmean(self.latest_harvest_crops, axis=1)
        damage =  np.where(mean > 0, ((mean - self.crop_produce) / mean), 0) 
        return damage

    def distance_livestock_migration(self, coordinates, livestock_coords):
        """
        Calculate the distance an agent has travelled with their livestock [km].

        Args:
            coordinates (np.ndarray): Home coordinates.
            livestock_coords (np.ndarray): Current livestock coordinates.

        Returns:
            np.ndarray: Distances for each agent.
        """
        distance = np.sqrt(abs((coordinates[:, 0] - livestock_coords[:,0])**2 + (coordinates[:,1] - livestock_coords[:,1])**2) )
        return distance

    def Protection_Motivation_Theory(self, measure, damage, risk_appraisal):
        """
        Calculate Intention to Adapt for every measure based on Protection Motivation Theory.

        Args:
            measure (int): Adaptation measure index.
            damage (np.ndarray): Damage values.
            risk_appraisal (np.ndarray): Risk appraisal memory.

        Returns:
            tuple: Intention to adapt for livestock, crops, and combined.
        """
        self_efficacy = self.SelfEfficacy(measure)
        adaptation_costs = self.Adaptationcosts(measure) # from 0 to 1
        drought_damage = np.maximum((1-np.exp(-damage)), 0)

        if measure == 0 or measure == 1:
            self.adaptation_efficacy = self.AdaptationEfficacy_livestock(measure)
            self.RiskAppraisal_Livestock = np.minimum(np.maximum(risk_appraisal[:, 1] + drought_damage + np.where(damage ==0 ,- 0.125 * risk_appraisal[:, 1], 0), 0),1) 
            self.CopingAppraisal_Livestock = self.gamma * self_efficacy + self.delta * self.adaptation_efficacy + self.epsilon * (1 - adaptation_costs)
            self.IntentionToAdapt_Livestock = self.alpha * self.RiskAppraisal_Livestock + self.beta * self.CopingAppraisal_Livestock

        if measure == 2 or measure == 3:
            self.adaptation_efficacy = self.AdaptationEfficacy_crops(measure)
            self.RiskAppraisal_Crops = np.minimum(np.maximum(risk_appraisal[:, 1] + drought_damage + np.where(damage ==0 ,- 0.125 * risk_appraisal[:, 1], 0), 0),1) 
            self.CopingAppraisal_Crops = self.gamma * self_efficacy + self.delta * self.adaptation_efficacy + self.epsilon * (1 - adaptation_costs)
            self.IntentionToAdapt_Crops = self.alpha * self.RiskAppraisal_Crops + self.beta * self.CopingAppraisal_Crops

        if measure == 4 or measure == 5: # water harvesting or soil moisture conservation
            self.adaptation_efficacy = self.AdaptationEfficacy(measure)
            self.RiskAppraisal = np.minimum(np.maximum(risk_appraisal[:, 0] + drought_damage + np.where(damage ==0 ,- 0.125 * risk_appraisal[:, 1], 0), 0),1) 
            self.CopingAppraisal = self.gamma * self_efficacy + self.delta * self.adaptation_efficacy + self.epsilon * (1 - adaptation_costs)
            self.IntentionToAdapt = self.alpha * self.RiskAppraisal + self.beta * self.CopingAppraisal

        assert (self.IntentionToAdapt_Livestock >= 0).all()
        assert (self.IntentionToAdapt_Livestock != np.nan).all()
        assert (self.IntentionToAdapt_Crops >= 0).all()
        assert (self.IntentionToAdapt_Crops != np.nan).all()
        assert (self.IntentionToAdapt >= 0).all()
        assert (self.IntentionToAdapt != np.nan).all()
        return self.IntentionToAdapt_Livestock, self.IntentionToAdapt_Crops, self.IntentionToAdapt

    def Decision_to_Adapt(self):
        """
        Determines whether people adapt.

        If agents can pay and intention to adapt is greater than a threshold,
        update assets, location, and adaptation measure.
        """
 
        # DECISION MADE BEGINNING DRY PERIOD
        if (self.current_day_of_year == self.start_dry_season):

            Random_threshold = np.full(self.n, self.intention_to_behavior)

            damage = self.Damage_livestock()
            damage_livelihood = (self.livestock_livelihood) * self.Damage_livestock() + (self.crop_livelihood * self.Damage_crops())
            
            self.risk_appraisal[:, 1:] = self.risk_appraisal[:, 0:-1] # previous timestep
            self.risk_appraisal[:, 0] = self.RiskAppraisal_1_livestock(damage) # current timestep
            PMT_0 = self.Protection_Motivation_Theory(0,damage, self.risk_appraisal)[0]
            PMT_1 = self.Protection_Motivation_Theory(1,damage, self.risk_appraisal)[0]
            PMT_4 = self.Protection_Motivation_Theory(4, damage_livelihood, self.risk_appraisal)[2]

            assert (PMT_0 != np.nan).all()
            assert (PMT_0 >= 0).all()

            assert (PMT_1 != np.nan).all()
            assert (PMT_1 >= 0).all()

            assert (PMT_4 != np.nan).all()
            assert (PMT_4 >= 0).all()

            costs_migration = self.adapt_costs[0] * (self.nr_livestock[:,0] + self.nr_livestock[:,1]) #migration
            costs_livestock = self.adapt_costs[1] * (self.nr_livestock[:,0] + self.nr_livestock[:,1]) #livestock change
            costs_water_harvesting = np.maximum(self.adapt_costs[4], self.adapt_costs[4]) #water harvesting 

            likelihood_to_adapt_0 = ( 1 -  ( ( 1 - PMT_0 ) ** ( self.lifetime_measures[0] ) ) ) 
            likelihood_to_adapt_1 = ( 1 -  ( ( 1 - PMT_1 ) ** ( self.lifetime_measures[1] ) ) )
            likelihood_to_adapt_4 = ( 1 -  ( ( 1 - PMT_4 ) ** ( self.lifetime_measures[4] ) ) )

                
            #[0] = MIGRATION

            self.adapt_measure_0[:] = 0 # after one year the measure expires
            r = list(range(self.n))
            stats.fixed_shuffle(self.seeds, r)

            # create copy of grass_yield
            self.grass_yield_copy = self.model.abc.grass_yield_.copy()

            self.livestock_coords = self.coordinates.copy() # start at home
            
            for i in r:

                if costs_migration[i] < self.assets[i] and likelihood_to_adapt_0[i] > Random_threshold[i]:

                    #find max coordinates
                    self.coordinates_max_grass[i] = self.max_neighbourhood_individual(self.range_lands[i], self.values_neighbourhood(self.range_lands, self.grass_yield_copy)[i]) #[self.range_lands[i]]

                    #update grass yield by consuming
                    self.grass_yield_copy[self.coordinates_max_grass[i][0],self.coordinates_max_grass[i][1]] -= (np.minimum(self.nr_livestock[:,0][i] * self.feed_required_cattle* (1- self.feed_residue_cattle), self.grass_yield_copy[self.coordinates_max_grass[i][0],self.coordinates_max_grass[i][1]] *1000*self.nr_livestock[:,0][i]) - np.minimum(self.nr_livestock[:,1][i] * self.feed_required_goats* (1- self.feed_residue_goats), self.grass_yield_copy[self.coordinates_max_grass[i][0],self.coordinates_max_grass[i][1]] *1000*self.nr_livestock[:,1][i])) #self.livestock_coords[i, 1],self.livestock_coords[i, 0]] * needs
                
                    self.adapt_measure_0[i] = 1
                    self.livestock_coords[i, 0] = self.coordinates_max_grass[i][1]
                    self.livestock_coords[i, 1] = self.coordinates_max_grass[i][0]
                    self.assets[i] = self.assets[i] - costs_migration[i]

                else:
                    self.grass_yield_copy[self.livestock_coords[i, 1],self.livestock_coords[i, 0]] -= (np.minimum(self.nr_livestock[:,0][i] * self.feed_required_cattle* (1- self.feed_residue_cattle), self.grass_yield_copy[self.livestock_coords[i, 1],self.livestock_coords[i, 0]] *1000*self.nr_livestock[:,0][i]) - np.minimum(self.nr_livestock[:,1][i] * self.feed_required_goats* (1- self.feed_residue_goats), self.grass_yield_copy[self.livestock_coords[i, 1],self.livestock_coords[i, 0]] *1000*self.nr_livestock[:,1][i])) #self.livestock_coords[i, 1],self.livestock_coords[i, 0]] * needs

            #[1] = LIVESTOCK CHANGE
            self.adapt_measure_1 = np.where((costs_livestock < self.assets) & (likelihood_to_adapt_1 > Random_threshold), 1, self.adapt_measure_1)
            self.assets = np.where((costs_livestock < self.assets) & (likelihood_to_adapt_1 > Random_threshold), self.assets - costs_livestock, self.assets)
            self.nr_livestock[:, 1] = np.where((costs_livestock < self.assets) & (likelihood_to_adapt_1 > Random_threshold), self.nr_livestock[:, 1] + self.nr_livestock[:, 0], self.nr_livestock[:, 1])
            self.nr_livestock[:, 0] = np.where((costs_livestock < self.assets) & (likelihood_to_adapt_1 > Random_threshold), 1, self.nr_livestock[:, 0])
            self.distance_migration = self.distance_livestock_migration(self.coordinates, self.livestock_coords)

            #[4] = WATER HARVESTING
            self.adapt_measure_4 = np.where((costs_water_harvesting < self.assets) & (likelihood_to_adapt_4 > Random_threshold), 1, self.adapt_measure_4)
            self.assets = np.where(likelihood_to_adapt_4 > Random_threshold, self.assets - costs_water_harvesting, self.assets)   
            
        # DECISIONS MADE JUST BEFORE (SHORT) RAINY SEASON
        if (self.current_day_of_year == (self.planting_date - 1)):
            Random_threshold = np.full(self.n, self.intention_to_behavior)
            
            damage = self.Damage_crops()
            damage_livelihood = (self.livestock_livelihood) * self.Damage_livestock() + (self.crop_livelihood * self.Damage_crops())

            self.risk_appraisal[:, 1:] = self.risk_appraisal[:, 0:-1] # previous timestep
            self.risk_appraisal[:, 0] = self.RiskAppraisal_1_crops(damage) # curent timestep
            PMT_2 = self.Protection_Motivation_Theory(2, damage, self.risk_appraisal)[1]
            PMT_3 = self.Protection_Motivation_Theory(3, damage, self.risk_appraisal)[1]
            PMT_5 = self.Protection_Motivation_Theory(5, damage_livelihood, self.risk_appraisal)[2]

            assert (PMT_2 != np.nan).all()
            assert (PMT_2 >= 0).all()

            assert (PMT_3 != np.nan).all()
            assert (PMT_3 >= 0).all()

            assert (PMT_5 != np.nan).all()
            assert (PMT_5 >= 0).all()
            
            costs_crops = self.adapt_costs[2] * self.land_acres #, crop types (30/acre)
            costs_irrigation = self.adapt_costs[3] * self.land_size # irrigation (1000*land size [ha])
            costs_soil_cons = self.adapt_costs[5] * self.land_acres # agroforesty (415/acre)

            likelihood_to_adapt_2 = ( 1 -  (( 1 - PMT_2 ) ** ( 1 / self.lifetime_measures[2] )) )
            likelihood_to_adapt_3 = ( 1 -  (( 1 - PMT_3 ) ** ( 1 / self.lifetime_measures[3] )) )
            likelihood_to_adapt_5 = ( 1 -  (( 1 - PMT_5 ) ** ( 1 / self.lifetime_measures[5] )) )

            self.adapt_measure_2[:] = 0 # after one year the measure expires

            crop_change_map = self.crop_map.reshape(self.height,self.width)

            #[2] = CROP CHANGE
            self.adapt_measure_2 = np.where((costs_crops < self.assets) & (likelihood_to_adapt_2 > Random_threshold), 1, 0)
            
            self.assets = np.where((costs_crops < self.assets) & (likelihood_to_adapt_2 > Random_threshold), self.assets - costs_crops, self.assets)
            self.land_use[self.coordinates[:,1],self.coordinates[:,0]] = np.where((costs_crops < self.assets) & (likelihood_to_adapt_2 > Random_threshold), 40, self.land_use[self.coordinates[:,1],self.coordinates[:,0]]) # representing certain
            crop_change_map[self.coordinates[:,1],self.coordinates[:,0]] = np.where((costs_crops < self.assets) & (likelihood_to_adapt_2 > Random_threshold), 10, crop_change_map[self.coordinates[:,1],self.coordinates[:,0]]) # set cropmap to casava 
            self.crop_map = crop_change_map.flatten() # communicated to ABM-CONNECTOR

            #[3] = IRRIGATION CHANGE
            self.assets = np.where(likelihood_to_adapt_3 > Random_threshold, self.assets - costs_irrigation, self.assets)
            self.adapt_measure_3 = np.where( likelihood_to_adapt_3 > Random_threshold, 1 , self.adapt_measure_3)
                
            #[5] = SOIL MOISTURE CONSERVATION
            self.adapt_measure_5 = np.where((costs_soil_cons < self.assets) & (likelihood_to_adapt_5 > Random_threshold), 1, self.adapt_measure_5)
            self.assets = np.where(likelihood_to_adapt_5 > Random_threshold, self.assets - costs_soil_cons, self.assets)

            for i in range(self.n):
                    if likelihood_to_adapt_5[i] > Random_threshold[i]:
                        
                        self.adapt_measure_5_grid[self.coordinates[i,1],self.coordinates[i,0]] = 1

            
    def Daily_needs(self):
        """
        Calculate daily needs of household and livestock.

        If needs are not met, agents may buy water, sell livestock, or livestock may die.
        Updates assets and livestock numbers accordingly.
        """

        # #END OF DRY SEASON SEASON: pay livestock

        if self.current_day_of_year == (self.end_dry_season + 1): 
            # in expenses of livestock and crops are calculated
            self.assets -= np.maximum((self.expenditures_livestock), 0)
            self.livestock_coords = self.coordinates.copy() # go back home
            
        # #END OF RAINY SEASON: collect income crops
        if self.current_day_of_year == (self.harvest_date + 1): 
            # in Income() harvest crops are calculated, and assest are updated
            income_crops = self.Income_Crops()
            self.assets +=  np.maximum(income_crops + self.off_farm_income - self.expenditures_crops - self.other_expenditures, 0) # update assets because of income/loss of harvest
        
        #CHECK IF DAILY NEEDS ARE MET -> water for livestock and household, otherwise buy. Livestock dies if not have been watered for X amount of days
        self.Harvest_livestock() # update daily livestock production
        #if drought: 10ksh/jerrycan household to get water
        self.assets = np.where((self.assets >= 2) & (self.model.abc.water_available == 0), self.assets - (2 + self.nr_livestock[:,0] * 0.05 + self.nr_livestock[:,1] * 0.01 - self.HH_size * self.Water_costs(self.current_day_of_year)), self.assets)
        
        self.memory_water_available[:, 1:] = self.memory_water_available[:, 0:-1]
        self.memory_water_available[:, 0] = self.model.abc.water_available + self.model.abc.livestock_from_storage[self.coordinates[:,1],self.coordinates[:,0]]

        # if goats have no water for 4 days, they die -> 20% remains
        self.nr_livestock[: , 0] = np.where(np.sum(self.memory_water_available[:, 0:4], axis = 1) == 0, self.nr_livestock[: , 0] / 5, self.nr_livestock[: , 0])

        # if goats have no water for 8 days, they die -> 20% remains
        self.nr_livestock[: , 1] = np.where(np.sum(self.memory_water_available, axis = 1) == 0, self.nr_livestock[: , 1] / 5, self.nr_livestock[: , 1] )

        #if you dont have money, sell goat (no drought)
        self.assets = np.where((self.assets <= 2) & (self.nr_livestock[:, 1] > 2), self.assets + self.goat_price * self.livestock_market(), self.assets) # livestock market??
        self.nr_livestock[:, 1] = np.where((self.assets <= 2) & (self.nr_livestock[:, 1] > 2), self.nr_livestock[:, 1] - 1, self.nr_livestock[:, 1])

        # if you have no goats sell cows... (no drought)
        self.assets = np.where((self.assets <= 2) & (self.nr_livestock[:, 1] < 2), self.assets + self.cow_price * self.livestock_market(), self.assets)
        self.nr_livestock[:, 0] = np.where((self.assets <= 2) & (self.nr_livestock[:, 0] > 2), self.nr_livestock[:, 0] - 1, self.nr_livestock[:, 0])
        

        self.milk_production_cows = self.nr_livestock[:, 0] * self.max_milk_production_cows  * self.model.abc.milk_yield_ratio.reshape(self.height, self.width)[self.livestock_coords[:, 1],self.livestock_coords[:, 0]] 
        self.milk_production_goats =  self.nr_livestock[:, 1] * self.max_milk_production_goats * self.model.abc.milk_yield_ratio.reshape(self.height, self.width)[self.livestock_coords[:, 1],self.livestock_coords[:, 0]] 
        
        # INCOME THROUGHOUT THE YEAR FOR LIVESTOCK
        self.milk_production = self.milk_production_cows + self.milk_production_goats
        self.assets += np.maximum(((self.milk_production_cows - (self.cow_milk_consumption * self.HH_size)) * self.milk_price_cows), 0) + np.maximum(((self.milk_production_goats- (self.goat_milk_consumption * self.HH_size)) * self.milk_price_goats) , 0)


#------------------------------------------------
#                MARKET & SEASONS
#------------------------------------------------

    def Water_costs(self, day): 
        """
        Calculate the costs of water per day.

        Args:
            day (int): Day of year.

        Returns:
            int: Water cost in Ksh
        """	
        if day < self.lenght_short_dry_season:  # short dry season
            water_costs = self.water_cost_dry_season
        if day > self.start_dry_season and day < self.end_dry_season: # long dry season
            water_costs = self.water_cost_dry_season
        else:
            water_costs = self.water_cost_wet_season
        return water_costs

    def crop_market(self):
        """
        Calculate the price of maize based on the average harvest of the last 10 years.

        Returns:
            float: Adjusted maize price.
        """
        total_harvest_year = np.mean(self.crop_produce)
        average_harvest = np.nanmean(self.latest_harvest_crops) # average over 10 years
        crop_price = self.maize_price * (average_harvest/total_harvest_year)
        return crop_price

    def livestock_market(self):
        """
        Calculate the price of livestock based on the average harvest of the last 10 years.

        Returns:
            float: Livestock price factor.
        """
        total_livestock_year = np.mean(self.livestock_produce)
        average_livestock = np.nanmean(self.latest_harvest_livestock)
        livestock_price_factor = (average_livestock/total_livestock_year)
        return livestock_price_factor

#------------------------------------------------
#                 STEP FUNCTION
#------------------------------------------------

    def load_timestep_data(self):
        """
        Update data for the current timestep.

        Loads environmental and hydrological variables from the model.
        """
        self.distance_hh_water = self.model.abc.distance_hh_water


    def step(self):
        """
        Take a step in the decision module.

        Updates agent state for the current timestep, including adaptation
        decisions and daily needs.
        """
        self.load_timestep_data()
        
        if self.model.config['general']['BAU'] == False:
            
            self.Decision_to_Adapt()

        self.Daily_needs()

class Agents(AgentBaseClass):
    """
    Container class to represent all agents in the model.

    Holds references to different agent types (currently only Farmers).
    """
    def __init__(self, model):
        """
        Initialize the agents container.

        Args:
            model: The main model object.
        """
        self.model = model
        self.agent_types = []
        self.farmers = Farmers(model, self) 

    def step(self):
        """
        Step function for the agents.

        Advances all agent types by one timestep.
        """

        if self.model.current_time >= self.model.config['general']['start_time']:
            self.farmers.step()