# libraries
import mlflow
import mlflow.pytorch
import numpy as np
import os
import pandas as pd
import shutil
import torch
import torch.nn as nn
from pathlib import Path

# custom libraries
import utilities
from metrics.perf_metrics import sensor_specific_loss, overall_loss, get_individual_losses, calculate_single_auc, group_by_segment_id


class IMADSModelManager:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 window_lengths: list[int],
                 num_channels: list[int],
                 sensors: dict,
                 preprocess_pipeline: callable = None,
                 postprocess_pipeline: callable = None,
                 save_after_n_epochs: int = 10,
                 checkpoint_path: str = 'checkpoints',
                 checkpoint_filename: str = None,
                 ml_tracking_path: str = 'mlruns',
                 ml_tracking_model_path: str = 'model',
                 ):

        self.model = model
        self.best_model = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.window_lengths = window_lengths
        self.num_channels = num_channels
        self.sensors = sensors
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_filename = Path(checkpoint_filename) if checkpoint_filename else Path('')
        self.checkpoint_filepath = self.checkpoint_path / self.checkpoint_filename
        self.ml_tracking_path = Path(ml_tracking_path)
        self.ml_tracking_model_path = Path(ml_tracking_model_path)
        self.preprocess_pipeline = preprocess_pipeline
        self.postprocess_pipeline = postprocess_pipeline
        self.save_after_n_epochs = save_after_n_epochs

        # Get cpu, gpu or mps device for training.
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"IMADSModelManager: Using {device} device")
        self.device = device
        self.best_model_checkpoint = None

    def load_checkpoint(self,
                         name: str # include .pth extension
                         ):

        checkpoint = None
        checkpoint_filepath = self.checkpoint_path / name
        if checkpoint_filepath.exists():
            try:
                checkpoint = torch.load(
                    checkpoint_filepath,
                    map_location=torch.device(self.device)
                )
            except:
                print(f'Error loading checkpoint at {checkpoint_filepath}')
        return checkpoint

    def set_best_model(self):
        if self.best_model_checkpoint:
            checkpoint = self.best_model_checkpoint
            print(
                'Found best model checkpoint in model_manager args, setting best model ...')
        else:
            print(
                'best model checkpoint not found in model_manager args, searching best model checkpoint ...')
            try:
                checkpoint = self.load_checkpoint(
                    f'best_{self.checkpoint_filename}.pth')
                print(
                    'Found best model checkpoint saved in memory, setting best model ...')
            except:
                raise (ValueError(
                    "No best model checkpoint found, please train the model first"))

        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_weights(self, epoch):
        checkpoint_filename = f'model_epoch_{epoch + 1}.pth'
        checkpoint_filepath = self.checkpoint_path / checkpoint_filename

        # Log checkpoint to MLflow
        mlflow.log_artifact(checkpoint_filepath, artifact_path="checkpoints")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model

    def format_checkpoint(self, epoch, model, optimizer, training_losses, training_losses_sensor, valid_losses, valid_losses_sensor):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': training_losses,
            'training_losses_sensor': training_losses_sensor,
            'valid_losses': valid_losses,
            'valid_losses_sensor': valid_losses_sensor
        }
        return checkpoint

    def save_checkpoint(self, checkpoint, name: str ='model.pth', verbose=0):

        checkpoint_filepath = self.checkpoint_path / name
        checkpoint_filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            checkpoint,
            checkpoint_filepath
        )

        # Log checkpoint to MLflow
        mlflow.log_artifact(checkpoint_filepath, artifact_path="checkpoints")

        if verbose > 0:
            print(
                f'Checkpoint saved at epoch {checkpoint["epoch"]}, checkpoint_filepath: {checkpoint_filepath}')

    def remove_checkpoint(self, 
                          name: str #include .pth
                          ):
        checkpoint_filepath = self.checkpoint_path / name
        if checkpoint_filepath.exists():
            if checkpoint_filepath.is_file():
                checkpoint_filepath.unlink(missing_ok=True)  # Removed `missing_ok` for compatibility
            else:
                shutil.rmtree(str(checkpoint_filepath))

    def train(self, train_data_loader, valid_data_loader, retrain=False, epochs=100):

        # Move model to the specified device
        # Fix: move the model to the specified device
        self.model.to(self.device)
        # Calculate the total number of batches in the training data
        num_batches = len(train_data_loader)

        # Initialize lists to store loss metrics for training and validation
        training_losses = [0 for _ in range(epochs)]
        training_losses_sensor = [0 for _ in range(epochs)]
        valid_losses = [0 for _ in range(epochs)]
        valid_losses_sensor = [0 for _ in range(epochs)]

        # Initialize the best validation loss to infinity and other training
        # controls
        self.best_valid_loss = float('inf')
        self.best_model_checkpoint = None  # To store the best model state if improved

        start_epoch = 0
        checkpoint = None
        if not retrain:
            checkpoint = self.load_checkpoint(f'{self.checkpoint_filename.name}.pth')
            if checkpoint:
                print('Loaded checkpoint')
                # Fix: use self.model.load_state_dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                training_losses = checkpoint['training_losses']
                training_losses_sensor = checkpoint['training_losses_sensor']
                valid_losses = checkpoint['valid_losses']
                valid_losses_sensor = checkpoint['valid_losses_sensor']
            else:
                print('No checkpoint available, training from scratch')
        else:
            self.remove_checkpoint(name=self.checkpoint_filename)
            print('removed checkpoint, training from scratch')

        # Main training loop over specified number of epochs
        for epoch in range(start_epoch, epochs):
            self.model.train()  # Set the model to training mode
            training_loss_epoch = 0
            multisensor = len(self.window_lengths) > 1
            if multisensor:
                training_loss_epoch_sensor = np.zeros(
                    len(self.model.window_lengths))

            # Loop over each batch from the data loader
            for batch_idx, (x_batch, _) in enumerate(train_data_loader):
                # x_batch: list (total_num_channels * window_lenght)

                if self.preprocess_pipeline:
                    x_batch = self.preprocess_pipeline(x_batch)

                x_batch = torch.concat(
                    # Flatten and concatenate batch data
                    [x.flatten(1) for x in x_batch], axis=1)

                self.optimizer.zero_grad()  # Zero the gradients to prepare for backward pass
                _, x_batch_estimate = self.model(x_batch)  # Forward pass

                # Calculate loss for each sensor without affecting gradients
                with torch.no_grad():
                    if multisensor:
                        training_loss_batch_sensor = sensor_specific_loss(
                            self.criterion,
                            x_batch,
                            x_batch_estimate,
                            self.window_lengths,
                            self.num_channels)
                        training_loss_batch_sensor = [
                            torch.mean(single_sensor_vec) for single_sensor_vec in training_loss_batch_sensor]

                if self.postprocess_pipeline:
                    x_batch_estimate = self.postprocess_pipeline(
                        x_batch_estimate)

                # Calculate overall loss from the batch
                loss = torch.mean(overall_loss(
                    self.criterion, x_batch, x_batch_estimate))

                loss.backward()  # Backpropagate the loss
                self.optimizer.step()  # Update model parameters

                # Convert sensor losses to list and track the batch loss
                if multisensor:
                    training_loss_batch_sensor = [
                        l.item() for l in training_loss_batch_sensor]
                training_loss_batch = loss.item()

                # Accumulate total loss for the epoch
                training_loss_epoch += training_loss_batch
                if multisensor:
                    training_loss_epoch_sensor += training_loss_batch_sensor

                # Calculate progress and average losses
                percent_complete = 100 * (batch_idx + 1) / num_batches
                avg_batch_loss = training_loss_epoch / (batch_idx + 1)
                if multisensor:
                    avg_batch_sensor_loss = training_loss_epoch_sensor / \
                        (batch_idx + 1)

                # Print training progress
                print(
                    f'Train Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{num_batches}] | '
                    f'{percent_complete:.2f}% Complete | Avg Batch Loss: {avg_batch_loss:.4f}', end='\r')

            # Append average losses after each epoch
            training_losses[epoch] = avg_batch_loss
            # Log training metrics to MLflow
            mlflow.log_metric("training_loss", avg_batch_loss, step=epoch)

            if multisensor:
                training_losses_sensor[epoch] = avg_batch_sensor_loss
                for i, sensor_loss in enumerate(avg_batch_sensor_loss):
                    mlflow.log_metric(
                        f"training_loss_sensor_{i}", sensor_loss, step=epoch)

            # Evaluate model on validation data and track losses
            avg_batch_loss, avg_batch_sensor_loss = self.evaluate(
                valid_data_loader)
            valid_losses[epoch] = avg_batch_loss

            # Print validation results
            print(
                f'\nValid Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(valid_data_loader)}] | '
                f'{percent_complete:.2f}% Complete | Avg Batch Loss: {avg_batch_loss:.4f}')
            # Log validation metrics to MLflow
            mlflow.log_metric("validation_loss", avg_batch_loss, step=epoch)

            if multisensor:
                valid_losses_sensor[epoch] = avg_batch_sensor_loss
                print(f'sensor losses {avg_batch_sensor_loss}')
                print('\n')
                for i, sensor_loss in enumerate(avg_batch_sensor_loss):
                    mlflow.log_metric(
                        f"validation_loss_sensor_{i}", sensor_loss, step=epoch)

            # track model weights every n epochs
            if (epoch + 1) % self.save_after_n_epochs == 0:
                checkpoint = self.format_checkpoint(
                    epoch, self.model, self.optimizer, training_losses, training_losses_sensor, valid_losses, valid_losses_sensor)
                self.save_checkpoint(
                    checkpoint=checkpoint, name=f'epoch{epoch}' + self.checkpoint_filename, verbose=1)

            if avg_batch_loss < self.best_valid_loss:
                print(f'model improved valid loss = {avg_batch_loss}')
                self.best_valid_loss = avg_batch_loss
                self.best_model = self.model
                self.best_model_checkpoint = self.format_checkpoint(
                    epoch, self.model, self.optimizer, training_losses, training_losses_sensor, valid_losses, valid_losses_sensor)

        # save best model only at the end of training
        best_model_name = f'best_{self.checkpoint_filename.name}.pth'
        self.remove_checkpoint(name=best_model_name)
        self.save_checkpoint(checkpoint=self.best_model_checkpoint,
                             name=best_model_name, verbose=1)
        
        # Log best model to MLflow
        best_model_path = self.ml_tracking_path / self.ml_tracking_model_path
        print(f"saving best model in path: {best_model_path}")

        if best_model_path.exists():
            shutil.rmtree(str(best_model_path))
        mlflow.pytorch.save_model(
            path = best_model_path,
            pytorch_model=self.best_model
            )

        # Convert lists to numpy arrays for further processing if needed
        self.valid_losses_sensor = pd.DataFrame(valid_losses_sensor).values
        self.training_losses_sensor = pd.DataFrame(
            training_losses_sensor).values

        # Load the best model state if one was saved
        try:
            # Fix: load the best model checkpoint
            checkpoint = self.load_checkpoint(
                f'best_{self.checkpoint_filename.name}.pth')
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
        except:
            pass

    def evaluate(self, data_loader):
        # Set the model to evaluation mode, which disables dropout and batch
        # normalization
        self.model.eval()
        total_loss = 0.0  # Initialize the total loss for the validation process

        # Disable gradient calculations for efficiency and safety during
        # evaluation
        with torch.no_grad():
            valid_loss_epoch = 0  # Total loss for the epoch
            # Array to hold sensor-specific losses
            valid_loss_epoch_sensor = np.zeros(len(self.window_lengths))

            # Iterate over each batch in the provided data loader
            for batch_idx, (x_batch, _) in enumerate(data_loader):
                # Flatten and concatenate batch data for processing
                x_batch = torch.concat(
                    [x.flatten(1) for x in x_batch], axis=1)
                # Compute model predictions
                _, x_batch_estimate = self.model(x_batch)

                # Compute sensor-specific losses without affecting gradients
                valid_loss_batch_sensor = sensor_specific_loss(
                    self.criterion,
                    x_batch,
                    x_batch_estimate,
                    self.window_lengths,
                    self.num_channels)
                valid_loss_batch_sensor = [
                    torch.mean(single_sensor_vec) for single_sensor_vec in valid_loss_batch_sensor]

                # Calculate the mean loss for the batch (scalar)
                valid_loss_batch = torch.mean(overall_loss(
                    self.criterion, x_batch, x_batch_estimate))
                valid_loss_batch = valid_loss_batch.item()  # Get Python scalar from tensor

                # Convert list of tensor losses to numpy array for
                # sensor-specific losses
                valid_loss_batch_sensor = np.array(
                    [l.item() for l in valid_loss_batch_sensor])

                # Accumulate losses for the entire epoch
                valid_loss_epoch += valid_loss_batch
                valid_loss_epoch_sensor += valid_loss_batch_sensor

                # Compute average loss across all batches processed so far
                avg_batch_loss = valid_loss_epoch / (batch_idx + 1)
                avg_batch_sensor_loss = valid_loss_epoch_sensor / \
                    (batch_idx + 1)

        # Return average losses for overall and sensor-specific evaluations
        return avg_batch_loss, avg_batch_sensor_loss

    def test(self, test_data_loader, aggregation_type):
        # Initialize lists to store various metrics
        sensor_losses_fusing = []
        sensor_losses_individual = []
        total_loss = []
        flattened_inputs = []
        predictions = []
        embeddings = []
        y = []
        # Set model to evaluation mode
        self.model.eval()
        for batch_idx, (x_batch, y_batch) in enumerate(test_data_loader):
            # Flatten and concatenate input data for processing
            x_batch = torch.concat([x.flatten(1) for x in x_batch], axis=1)

            # Get model outputs including embeddings and predictions
            embedding, x_batch_estimate = self.model(x_batch)

            # Compute sensor-specific losses and convert them to NumPy for
            # easier manipulation
            sensor_loss_batch = torch.stack(
                sensor_specific_loss(
                    self.criterion,
                    x_batch,
                    x_batch_estimate,
                    self.window_lengths,
                    self.num_channels)).detach().cpu().numpy()
            sensor_losses_fusing.append(sensor_loss_batch.T)

            # Get individual sensor losses using a utility function
            sensor_loss_batch_individual = get_individual_losses(
                self.model, self.sensors, self.window_lengths, self.num_channels, x_batch, self.criterion)
            sensor_losses_individual.append(sensor_loss_batch_individual.T)

            # Compute total loss for the batch and append to the list
            total_loss.append(
                self.criterion(x_batch, x_batch_estimate).detach().cpu().numpy())
            flattened_inputs.append(x_batch.detach().cpu().numpy())
            predictions.append(x_batch_estimate.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())

            y.append(y_batch)

        # Concatenate arrays for the whole test dataset
        flattened_inputs = np.concatenate(flattened_inputs, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)

        y = pd.concat([pd.DataFrame(yi)
                      for yi in y], axis=0)  # Flatten the list y
        y['label'] = y['anomaly_label'].apply(
            lambda x: 'normal' if x == 'normal' else 'anomaly')

        # Create DataFrame with sensor fusion anomaly scores and individual
        # sensor scores
        anomaly_scores_df = pd.DataFrame(
            data=np.concatenate(sensor_losses_fusing), columns=[
                f'f_{sensor}' for sensor in self.sensors])
        anomaly_scores_df[[f's_{sensor}' for sensor in self.sensors]] = np.concatenate(
            sensor_losses_individual)

        # Add total loss to the DataFrame
        anomaly_scores_df['total_loss'] = pd.Series(
            np.concatenate(total_loss, axis=0))

        # Combine anomaly scores DataFrame with Y_test for analysis
        Y_test = pd.concat([anomaly_scores_df.reset_index(
            drop=True), y.reset_index(drop=True)], axis=1)

        Y_test_grouped = group_by_segment_id(
            Y_test, anomaly_scores_df.columns, aggregation_type, verbose=0)

        # Calculate AUC for each anomaly score column
        results = {}
        for column in anomaly_scores_df.columns:
            results[column] = calculate_single_auc(
                Y_test_grouped, anomaly_score_column=column)

        # Assemble results into DataFrame and adjust index
        results = pd.concat(results)  # Concatenate results for each column
        results.index.names = ['column_names', 'duplicate']
        results.index = results.index.droplevel(
            'duplicate')  # Simplify index for clarity

        return results

    def get_anomaly_scores(self, data_loader, criterion):
        # Initialize lists to store various metrics
        sensor_losses_fusing = []
        sensor_losses_individual = []
        total_loss = []
        flattened_inputs = []
        predictions = []
        embeddings = []
        labels = []

        # Set model to evaluation mode
        self.model.eval()
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            # Flatten and concatenate input data for processing
            x_batch = torch.concat([x.flatten(1) for x in x_batch], axis=1).to(
                self.device)
            labels.append(pd.DataFrame(y_batch))

            # Get model outputs including embeddings and predictions
            embedding, x_batch_estimate = self.model(x_batch)

            # Compute sensor-specific losses and convert them to NumPy for
            # easier manipulation
            sensor_loss_batch = torch.stack(
                sensor_specific_loss(
                    self.criterion,
                    x_batch,
                    x_batch_estimate,
                    self.window_lengths,
                    self.num_channels)).detach().cpu().numpy()
            sensor_losses_fusing.append(sensor_loss_batch.T)

            # Get individual sensor losses using a utility function
            sensor_loss_batch_individual = get_individual_losses(
                self.model, self.sensors, self.window_lengths, self.num_channels, x_batch, self.criterion)
            sensor_losses_individual.append(sensor_loss_batch_individual.T)

            # Compute total loss for the batch and append to the list
            total_loss.append(
                self.criterion(x_batch, x_batch_estimate).detach().cpu().numpy())
            flattened_inputs.append(x_batch.detach().cpu().numpy())
            predictions.append(x_batch_estimate.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())

        # Concatenate arrays for the whole test dataset
        flattened_inputs = np.concatenate(flattened_inputs, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)
        labels_df = pd.concat(labels, axis=0)

        # Create DataFrame with sensor fusion anomaly scores and individual
        # sensor scores
        anomaly_scores_df = pd.DataFrame(
            data=np.concatenate(sensor_losses_fusing), columns=[
                f'f_{sensor}' for sensor in self.sensors])
        anomaly_scores_df[[f's_{sensor}' for sensor in self.sensors]] = np.concatenate(
            sensor_losses_individual)

        # Add total loss to the DataFrame
        anomaly_scores_df['total_loss'] = pd.Series(
            np.concatenate(total_loss, axis=0))

        # add labels
        anomaly_scores_df = pd.concat(
            [anomaly_scores_df, labels_df.reset_index(drop=True)], axis=1)

        return anomaly_scores_df, flattened_inputs, predictions, embeddings
