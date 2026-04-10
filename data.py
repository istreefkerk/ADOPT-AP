import os
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from honeybees.library.mapIO import NetCDFReader, ArrayReader
import rasterio
import matplotlib.pyplot as plt
import xarray as xr


class Data():
    def __init__(self, model):
        self.model = model
        self.data_folder = 'DataDrive'
        
        #Define data sources

        self.elevation = ArrayReader( 
            fp='DataDrive/Ewaso/EW_1k_dem_m.tif', 
            bounds=self.model.bounds
        )

        self.river_network = ArrayReader(
            fp='DataDrive/Ewaso/EW_1k_riv.asc', 
            bounds=self.model.bounds
        )

        self.land_cover = ArrayReader(
            fp='DataDrive/Ewaso/EW_land_cover.asc', 
            bounds=self.model.bounds
        )

        self.density = ArrayReader(
            fp='DataDrive/Ewaso/EW_density.asc', 
            bounds=self.model.bounds
        )

        self.farm_agents = ArrayReader(
            fp='DataDrive/Ewaso/EW_farm_agents.asc', 
            bounds=self.model.bounds
        )

        self.greenhouses = ArrayReader(
            fp='DataDrive/Ewaso/EW_greenhouses.asc', 
            bounds=self.model.bounds
        )

        self.mask = ArrayReader(
            fp='DataDrive/Ewaso/EW_1k_mask_m.asc',
            bounds=self.model.bounds
        )

        self.admin = ArrayReader(
            fp='DataDrive/Ewaso/EW_admin_2.asc', 
            bounds=self.model.bounds
        )

        self.climate_zone = ArrayReader(
            fp='DataDrive/Ewaso/EW_climate_zone.asc', 
            bounds=self.model.bounds
        )

        self.sub_catchment = ArrayReader(
            fp='DataDrive/Ewaso/Sub_catchments.asc', 
            bounds=self.model.bounds
        )

    def step(self):
        self.spei.step()