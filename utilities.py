import numpy as np
import os
import pandas as pd
import torch
import yaml
import metrics

def load_yaml_params(yaml_file = 'params.yaml', verbose = 1):
    with open(yaml_file, 'r') as file:
        try:
            params = yaml.safe_load(file)
            
            # Initialize other parameters
            params['checkpoint_path'] = os.path.join(params['checkpoint_path'], params['machine'])
            params['checkpoint_name'] = f"{params['checkpoint_name']}_seed{params['seed']}.pth"
            params['checkpoint_filepath'] = os.path.join(params['checkpoint_path'], params['checkpoint_name'])

            # Get cpu, gpu or mps device for training.
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

            params['device'] = device
            
            params['criterion'] = eval(f"metrics.{params['criterion']}")

            if verbose ==1:
                print(f"Using {device} device")
                for k,v in params.items():
                    print(k, ':', v)

            return params
        
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            return None

def get_sensors_enabled(params):

    return [sensor for sensor, value in params['sensors_enabled'].items() if value==True]

if __name__=='__main__':
    params=load_yaml_params()
    print(params)