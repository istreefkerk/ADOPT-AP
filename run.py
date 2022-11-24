import json
import argparse
import os
import numpy as np
import pandas as pd
import rasterio

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas

from model import D2EModel

def get_study_area():
    with rasterio.open('DataDrive/Sub-Ewaso/dem_square.tif') as src: #'DataDrive/Sub-Ewaso/dem_square.tif', test/test_dem.tif'
        bounds = src.bounds
        print(bounds)
    
    return {
        'name': 'isiolo',
        'xmin': bounds.left,
        'xmax': bounds.right,
        'ymin': bounds.bottom,
        'ymax': bounds.top,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.set_defaults(headless=True) # was False
    parser.add_argument('--no-browser', dest='browser', action='store_false')
    parser.set_defaults(browser=True)
    parser.add_argument('--port', dest='port', type=int, default=8521)
    #parser.add_argument('--export_folder', dest='export_folder', type=str, default=None)
    args = parser.parse_args()

    study_area = get_study_area()

    CONFIG_PATH = 'down2earth_true.yml' #'down2earth_false.yml' 

    MODEL_NAME = 'DOWN2EARTH'
    
    filename_input = 'DataDrive/Sub-Ewaso/input.dmp' #test/input_test.dmp

    export_folder = 'report'
    
    series_to_plot = []

    model_params = {
        "config_path": CONFIG_PATH,
        "study_area": study_area,
        "filename_input": filename_input,
        "report_folder": export_folder
    }

    if args.headless:
        model = D2EModel(**model_params)
        model.run()
        report = model.report()
    else:
        server_elements = [
            Canvas(study_area['xmin'], study_area['xmax'], study_area['ymin'], study_area['ymax'], max_canvas_height=800, max_canvas_width=1200, unit='meters')
        ] + [ChartModule(series) for series in series_to_plot]

        DISPLAY_TIMESTEPS = [
            'day',
            'week',
            'month',
            'year'
        ]

        server = ModularServer(MODEL_NAME, D2EModel, server_elements, DISPLAY_TIMESTEPS, model_params=model_params, port=None)
        server.launch(port=args.port, browser=args.browser)