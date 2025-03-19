import hydra
import mlflow
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# custom
import metrics
from models import IMADSModelManager
import utilities
from datasets.dataset_IMADS import IMADSDatasetTest
from preprocessing.prepr_pipelines import PreprocessingPipeline


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    print(cfg)
    with mlflow.start_run():
        
        device = utilities.get_device(verbose=1)
        utilities.set_torch_seed(cfg.seed, device, verbose=1)

        sensors_enabled = utilities.get_sensors_enabled(cfg)
        print(f'Sensors enabled: {sensors_enabled}')

        pipeline_cfg = cfg.preprocess_pipeline
        preproc_pipeline = PreprocessingPipeline(
            norm=cfg.normalization,
            device=device,
            **pipeline_cfg,  # Correctly unpack the pipeline parameters
        )

        test_dataset = IMADSDatasetTest(
            data_folder=cfg.data_folder,
            sensors_enabled=sensors_enabled,
            machine = cfg.machine,
            window_size_ms = cfg.window_size_ms,
            transform_pipeline = None,
            device=device,
        )
        test_dataset.set_transform_pipeline(preproc_pipeline.pipeline)
        test_dataset.apply_preprocess_pipeline()

        test_data_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        # Extract the number of channels and window lengths for each sensor\n",
        num_channels = [x.shape[1] for x in test_dataset.X]
        window_lengths = [x.shape[2] for x in test_dataset.X]
        sensors = test_dataset.sensor_dict

        # model = Autoencoder(window_lengths, num_channels, cfg['layer_dims'])
        # checkpoint = torch.load(cfg['checkpoint_filepath'], map_location=torch.device(cfg['device']), weights_only=False)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

        model = mlflow.pytorch.load_model(
            model_uri= f'{cfg.ml_tracking.path}/{cfg.ml_tracking.model_path}', 
            map_location=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        model_manager = IMADSModelManager(model=model, 
                                          optimizer=optimizer, 
                                          criterion=eval(f"metrics.{cfg.criterion}"), 
                                          window_lengths=window_lengths, 
                                          num_channels=num_channels, 
                                          sensors=sensors)

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
        
        results_path = Path(cfg.results_folder) / cfg.machine
        results_path.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(results_path / 'AUC_scores.csv')

        # Log metrics to MLflow
        for index, row in metrics.iterrows():
            for column in metrics.columns:
                metric_name = f"{index}_{column.replace(' ', '_')}"
                mlflow.log_metric(metric_name, row[column])
        
        # Log artifacts (CSV file)
        mlflow.log_artifacts(local_dir=str(results_path),
                            artifact_path='results',
                            run_id=mlflow.active_run().info.run_id)

if __name__ == '__main__':
    main()