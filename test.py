import argparse
import mlflow
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# custom
from models import Autoencoder, IMADSModelManager
import utilities
from datasets.dataset_IMADS import IMADSDatasetTest
from preprocessing.prepr_pipelines import PreprocessingPipeline

if __name__ == '__main__':
    with mlflow.start_run():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder", type=str, help="path to train and test data root folder")
        parser.add_argument("--checkpoint_path", type=str, help="path to checkpoints folder")
        parser.add_argument("--results_folder" , type=str, help="path to results folder")
        parser.add_argument("--model_path", type=str, help="path to mlflow mdoel tracking uri")
        args = parser.parse_args()

        print('Resolved model path:', args.model_path)

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

        pipeline_params = params['preprocess_pipeline']
        preproc_pipeline = PreprocessingPipeline(
            norm=params['normalization'],
            device=params['device'],
            **pipeline_params,  # Correctly unpack the pipeline parameters
        )        

        test_dataset = IMADSDatasetTest(
            data_folder=Path(params['data_folder']),
            sensors_enabled=sensors_enabled,
            machine=params['machine'], 
            window_size_ms=params['window_size_ms'],
            device=params['device'],
            transform_pipeline=None,
        )
        test_dataset.set_transform_pipeline(preproc_pipeline.pipeline)
        test_dataset.apply_preprocess_pipeline()

        test_data_loader = DataLoader(
            test_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Extract the number of channels and window lengths for each sensor\n",
        num_channels = [x.shape[1] for x in test_dataset.X]
        window_lengths = [x.shape[2] for x in test_dataset.X]
        sensors = test_dataset.sensor_dict

        # model = Autoencoder(window_lengths, num_channels, params['layer_dims'])
        # checkpoint = torch.load(params['checkpoint_filepath'], map_location=torch.device(params['device']), weights_only=False)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        model = mlflow.pytorch.load_model(model_uri=params['ml_tracking']['model_path'], map_location=params['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        model_manager = IMADSModelManager(model, optimizer, params['criterion'], window_lengths, num_channels, sensors, params=params)

        AUC_scores = model_manager.test(test_data_loader, 'median')

        #show results as in paper
        metrics = AUC_scores.copy()
        metrics.columns = ['ST', 'Source', 'Target']
        new_order = [
            'total_loss',
            'f_ism330dhcx_acc',
            's_ism330dhcx_acc',
            'f_ism330dhcx_gyro',
            's_ism330dhcx_gyro',
            'f_imp23absu_mic',
            's_imp23absu_mic']
        metrics = metrics.reindex(new_order)
        metrics.index = [
            'Overall',
            'F-acc',
            'S-acc',
            'F-gyr',
            'S-gyr',
            'F-mic',
            'S-mic']
        metrics = metrics * 100
        metrics = metrics.round(2)
        print(metrics)
        
        results_path = Path(params['results_folder']) / params['machine']
        results_path.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(results_path / 'AUC_scores.csv')

        # Log metrics to MLflow
        for index, row in metrics.iterrows():
            for column in metrics.columns:
                metric_name = f"{index}_{column.replace(' ', '_')}"
                mlflow.log_metric(metric_name, row[column])
        
        # Log artifacts (CSV file)
        mlflow.log_artifacts(results_path)




