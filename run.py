import json
import argparse
import os
import numpy as np
import pandas as pd
import rasterio
import random
from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas

from model import D2EModel

def get_study_area():
    with rasterio.open('DataDrive/Ewaso/EW_1k_dem_m.asc') as src: # set model bounds
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

    # settings 
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.set_defaults(headless=True) 
    parser.add_argument('--no-browser', dest='browser', action='store_false')
    parser.set_defaults(browser=True)
    parser.add_argument('--port', dest='port', type=int, default=8521)
    parser.add_argument('--config', dest='config', type=str, default='down2earth_true.yml')
    parser.add_argument('--export_folder', dest='export_folder', type=str, default=None)
    args = parser.parse_args()

    study_area = get_study_area()

    CONFIG_PATH = parser.parse_args().config 

    MODEL_NAME = 'DOWN2EARTH' # model name
    
    filename_input = 'DataDrive/Ewaso/input.dmp' # insert input file

    export_folder = parser.parse_args().export_folder
    
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