from datetime import date
import numpy as np
from dateutil.relativedelta import relativedelta

# from honeybees.model import Model as ABM_Model
# from honeybees.reporter import Reporter
from polygene.model import Model as ABM_Model
from polygene.reporter import Reporter
from agents import Agents
from data import Data
# from honeybees.area import Area
from polygene.area import Area
from artists import Artists
from DRYP.run_DRYP import DRYP_Model

from datetime import timedelta

class D2EModel(ABM_Model,DRYP_Model): #,DRYP_Model
    # initialize all model settings
    def __init__(self, config_path, study_area, filename_input, export_folder):
        self.__init_ABM__(config_path, study_area, export_folder)
        self.__init_hydromodel__(filename_input)
        self.reporter = Reporter(self, export_folder)
    
    # initialize ABM module
    def __init_ABM__(self, config_path, study_area, export_folder):
        ABM_Model.__init__(self, config_path) # current_time, timestep_length,  , args = None, n_timesteps = n_timesteps
        self.current_time = self.config['general']['start_time'] # date(2005, 1, 1) 
        self.timestep_length = timedelta(days=1)
        self.end_time = self.config['general']['end_time'] #date(2015, 12, 31) 
        self.n_timesteps = (self.end_time - self.current_time) / self.timestep_length
        self.n_timesteps = int(self.n_timesteps)
        #assert self.n_timesteps.is_integer()
        self.export_folder = self.config['general']['export_folder']

        self.artists = Artists(self)
        self.area = Area(self, study_area)
        self.data = Data(self)
        self.agents = Agents(self)
        self.reporter = Reporter(self, export_folder)
        
    #initialize hydrological module
    def __init_hydromodel__(self, filename_input):
       DRYP_Model.__init__(self, filename_input)
        
    # take a step    
    def step(self, step_size=1):
        if isinstance(step_size, str):
            n = self.parse_step_str(step_size)
        else:
            n = step_size
        for _ in range(n):
            if self.config['general']['ABM'] == True: # alleen runnen als ABM module aanstaat
                ABM_Model.step(self, 1, report=True) # false
            DRYP_Model.step(self)
            self.reporter.step()

    def run(self):
        
        for i in range(self.n_timesteps):
            print('number in loop', i, 'out of', self.n_timesteps)
            self.step()
        self.reporter.report()
        DRYP_Model.save_output(self)