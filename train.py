# libraries
import argparse
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# custom
from models import Autoencoder, IMADSModelManager
import utilities
from datasets.dataset_IMADS import IMADSDatasetTrain, IMADSBaseDataset
from preprocessing.prepr_pipelines import PreprocessingPipeline

if __name__ == '__main__':
    with mlflow.start_run():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder", type=str, help="path to train and test data root folder")
        parser.add_argument("--checkpoint_path", type=str, help="path to checkpoints folder")
        parser.add_argument("--results_folder" , type=str, help="path to results folder")
        parser.add_argument("--model_path", type=str, help="path to mlflow mdoel tracking uri")
        args = parser.parse_args()

        params = utilities.load_yaml_params()

        params['data_folder'] = args.data_folder if args.data_folder else params['data_folder']
        params['checkpoint_path'] = args.checkpoint_path if args.checkpoint_path else params['checkpoint_path']
        params['results_folder'] = args.results_folder if args.results_folder else params['results_folder']
        params['ml_tracking']['model_path'] = args.model_path if args.model_path else params['ml_tracking']['model_path']
        
        # Set the seed for general torch operations
        torch.manual_seed(params['seed'])

        # Set the seed for MPS torch operations (ones that happen on the MPS Apple GPU)

        if params['device'] == 'mps':
            torch.mps.manual_seed(params['seed'])
        elif params['device'] == 'cuda':
            torch.cuda.manual_seed(params['seed'])
        elif params['device'] == 'cpu':
            torch.manual_seed(params['seed'])
        else:
            raise ValueError(f"Wrong device value: {params['device']}")

        sensors_enabled = utilities.get_sensors_enabled(params)
        print(sensors_enabled)

        train_dataset = IMADSDatasetTrain(
            data_folder=Path(params['data_folder']),
            sensors_enabled=sensors_enabled,
            machine=params['machine'], 
            window_size_ms=params['window_size_ms'], 
            device=params['device'],
            transform_pipeline=None,
            valid_size=params['valid_size'],
            seed=params['seed']
            )
        
        pipeline_params = params['preprocess_pipeline']
        preproc_pipeline = PreprocessingPipeline(
            norm=params['normalization'],
            device=params['device'],
            **pipeline_params,  # Correctly unpack the pipeline parameters
        )
        
        train_dataset.set_transform_pipeline(preproc_pipeline.pipeline)
        train_dataset.apply_preprocess_pipeline()

        X_valid, y_valid = train_dataset.get_valid_dataset()
        valid_dataset = IMADSBaseDataset(X_valid, 
                                        y_valid['anomaly_label'].to_list(), 
                                        sensors_enabled=sensors_enabled,
                                        device=params['device'], 
                                        transform_pipeline=preproc_pipeline.pipeline)
        
        train_data_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Extract the number of channels and window lengths for each sensor\n",
        num_channels = [x.shape[1] for x in train_dataset.X]
        window_lengths = [x.shape[2] for x in train_dataset.X]
        sensors = train_dataset.sensor_dict

        model = Autoencoder(window_lengths, num_channels, params['layer_dims'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])    

        model_manager = IMADSModelManager(
            model, 
            optimizer, 
            params['criterion'], 
            window_lengths, 
            num_channels, 
            sensors,
            save_after_n_epochs=params['save_after_n_epochs'],
            params=params)
        
        model_manager.train(train_data_loader, valid_data_loader, retrain = params['retrain'])

        # Log the final model
        mlflow.pytorch.log_model(model, "model")