import os
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from honeybees.library.mapIO import NetCDFReader, ArrayReader
from honeybees.library.raster import write_to_array
import rasterio
import matplotlib.pyplot as plt


class Data():
    def __init__(self, model):
        self.model = model
        self.data_folder = 'DataDrive'
        
        self.elevation = ArrayReader( 
            fp='DataDrive/Sub-Ewaso/dem_square.tif', #test/test_dem.tif'
            bounds=self.model.bounds
        )

        self.river_network = ArrayReader(
            fp='DataDrive/Sub-Ewaso/riv_square.tif', #'DataDrive/Sub-Ewaso/riv_square.tif', test/test_riv.tif
            bounds=self.model.bounds
        )

        self.land_cover = ArrayReader(
            fp='DataDrive/Sub-Ewaso/1k_land_cover.tif', #Sub-Ewaso/1k_land_cover.tif DataDrive/, test/test_land_cover.tif
            bounds=self.model.bounds
        )

        self.density = ArrayReader(
            fp='DataDrive/Sub-Ewaso/Sub_density.tif', #Sub-Ewaso/1k_land_cover.tif DataDrive/, test/test_land_cover.tif
            bounds=self.model.bounds
        )

        self.mask = ArrayReader(
            fp='DataDrive/Sub-Ewaso/Sub_mask.tif', #Sub-Ewaso/1k_land_cover.tif DataDrive/, test/test_land_cover.tif
            bounds=self.model.bounds
        )
        # Not used in test version
        #self.spei = NetCDFReader(
        #    fp=os.path.join('DataDrive/Sub-Ewaso/Final_SPEI_catchment_squared_2.nc'),  # 'DataDrive/Sub-Ewaso/Final_SPEI_catchment_squared.nc'
        #    varname='spei', bounds = self.model.bounds, latname = 'lat', lonname ='lon', timename ='time'
        #)

    def step(self):
        pass
        #self.spei.step()
