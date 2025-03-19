# libraries
# import argparse
import hydra
from hydra.utils import  get_original_cwd, to_absolute_path
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# custom
import metrics
from models import Autoencoder, IMADSModelManager
import utilities
from datasets.dataset_IMADS import IMADSDatasetTrain, IMADSBaseDataset
from preprocessing.prepr_pipelines import PreprocessingPipeline


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    print(cfg)
    with mlflow.start_run():

        device = utilities.get_device(verbose=1)
        utilities.set_torch_seed(cfg.seed, device, verbose=1)

        sensors_enabled = utilities.get_sensors_enabled(cfg)
        print(f'Sensors enabled: {sensors_enabled}')

        train_dataset = IMADSDatasetTrain(
            data_folder = cfg.data_folder,
            sensors_enabled = sensors_enabled,
            machine = cfg.machine,
            window_size_ms = cfg.window_size_ms,
            transform_pipeline = None,
            device=device,
            valid_size = cfg.valid_size,
            seed = cfg.seed
        )

        pipeline_cfg = cfg.preprocess_pipeline
        preproc_pipeline = PreprocessingPipeline(
            norm=cfg.normalization,
            device=device,
            **pipeline_cfg,  # Correctly unpack the pipeline parameters
        )

        train_dataset.set_transform_pipeline(preproc_pipeline.pipeline)
        train_dataset.apply_preprocess_pipeline()

        X_valid, y_valid = train_dataset.get_valid_dataset()
        valid_dataset = IMADSBaseDataset(X_valid,
                                        y_valid['anomaly_label'].to_list(),
                                        sensors_enabled=sensors_enabled,
                                        device=device,
                                        transform_pipeline=preproc_pipeline.pipeline)

        train_data_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=cfg.batch_size, shuffle=False)

        # Extract the number of channels and window lengths for each sensor\n",
        num_channels = [x.shape[1] for x in train_dataset.X]
        window_lengths = [x.shape[2] for x in train_dataset.X]
        sensors = train_dataset.sensor_dict

        model = Autoencoder(window_lengths, num_channels, cfg.layer_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        model_manager = IMADSModelManager(
            model,
            optimizer,
            eval(f"metrics.{cfg.criterion}"),
            window_lengths,
            num_channels,
            sensors,
            save_after_n_epochs=cfg.save_after_n_epochs,
            checkpoint_path=cfg.checkpoint_path,
            checkpoint_filename=cfg.checkpoint_name,
            ml_tracking_path=cfg.ml_tracking.path,
            ml_tracking_model_path=cfg.ml_tracking.model_path,
            )

        model_manager.train(train_data_loader, valid_data_loader,
                            retrain=cfg.retrain, epochs=cfg.num_epochs)

        # Log the final model
        mlflow.pytorch.log_model(model, "model")


if __name__ == '__main__':
    main()