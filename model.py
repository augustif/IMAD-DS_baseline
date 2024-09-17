# libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# custom libraries
import utilities


class AutoencoderFC(nn.Module):
    def __init__(self, window_lengths, num_channels, params, sensors):
        super(AutoencoderFC, self).__init__()
        self.window_lengths = window_lengths
        self.sensors = sensors
        self.num_channels = num_channels
        self.input_dim = sum(window_length * num_channel for window_length,
                             num_channel in zip(window_lengths, num_channels))
        self.params = params
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_layers = []
        current_dim = self.input_dim

        for dim in self.params['layer_dims'][:-1]:
            encoder_layers.append(nn.Linear(current_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())

            current_dim = dim

        dim = self.params['layer_dims'][-1]
        encoder_layers.append(nn.Linear(current_dim, dim))
        current_dim = dim
        return nn.Sequential(*encoder_layers)

    def build_decoder(self):
        decoder_layers = []
        current_dim = self.params['layer_dims'][-1]

        for dim in reversed(self.params['layer_dims'][:-1]):
            decoder_layers.append(nn.Linear(current_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            current_dim = dim
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def fit(self, train_data_loader, valid_data_loader, optimizer):
        # Move model to the specified device
        self.to(self.params['device'])
        # Calculate the total number of batches in the training data
        num_batches = len(train_data_loader)

        # Initialize lists to store loss metrics for training and validation
        training_losses = []
        training_losses_sensor = []
        valid_losses = []
        valid_losses_sensor = []

        # Initialize the best validation loss to infinity and other training
        # controls
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # To store the best model state if improved

        # Main training loop over specified number of epochs
        for epoch in range(self.params['num_epochs']):
            self.train()  # Set the model to training mode
            training_loss_epoch = 0
            training_loss_epoch_sensor = np.zeros(len(self.window_lengths))

            # Loop over each batch from the data loader
            for batch_idx, (x_batch, _) in enumerate(train_data_loader):
                x_batch = torch.concat(
                    # Flatten and concatenate batch data
                    [x.flatten(1) for x in x_batch], axis=1)
                optimizer.zero_grad()  # Zero the gradients to prepare for backward pass
                _, x_batch_estimate = self(x_batch)  # Forward pass

                # Calculate loss for each sensor without affecting gradients
                with torch.no_grad():
                    training_loss_batch_sensor = utilities.sensor_specific_loss(
                        self.params['criterion'],
                        x_batch,
                        x_batch_estimate,
                        self.window_lengths,
                        self.num_channels)
                    training_loss_batch_sensor = [
                        torch.mean(single_sensor_vec) for single_sensor_vec in training_loss_batch_sensor]

                # Calculate overall loss from the batch
                loss = torch.mean(utilities.overall_loss(
                    self.params['criterion'], x_batch, x_batch_estimate))

                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update model parameters

                # Convert sensor losses to list and track the batch loss
                training_loss_batch_sensor = [
                    l.item() for l in training_loss_batch_sensor]
                training_loss_batch = loss.item()

                # Accumulate total loss for the epoch
                training_loss_epoch += training_loss_batch
                training_loss_epoch_sensor += training_loss_batch_sensor

                # Calculate progress and average losses
                percent_complete = 100 * (batch_idx + 1) / num_batches
                avg_batch_loss = training_loss_epoch / (batch_idx + 1)
                avg_batch_sensor_loss = training_loss_epoch_sensor / \
                    (batch_idx + 1)

                # Print training progress
                print(
                    f'Train Epoch [{epoch+1}/{self.params["num_epochs"]}] | Batch [{batch_idx+1}/{num_batches}] | '
                    f'{percent_complete:.2f}% Complete | Avg Batch Loss: {avg_batch_loss:.4f}', end='\r')

            # Append average losses after each epoch
            training_losses.append(avg_batch_loss)
            training_losses_sensor.append(avg_batch_sensor_loss)

            # Evaluate model on validation data and track losses
            avg_batch_loss, avg_batch_sensor_loss = self.evaluate(
                valid_data_loader)
            valid_losses.append(avg_batch_loss)
            valid_losses_sensor.append(avg_batch_sensor_loss)

            # Print validation results
            print(
                f'\nValid Epoch [{epoch+1}/{self.params["num_epochs"]}] | Batch [{batch_idx+1}/{len(valid_data_loader)}] | '
                f'{percent_complete:.2f}% Complete | Avg Batch Loss: {avg_batch_loss:.4f}')
            print(f'sensor losses {avg_batch_sensor_loss}')
            print('\n')

            # Check for improvement and update patience or terminate training
            # if needed
            patience_counter += 1
            if avg_batch_loss < best_valid_loss:
                print(f'model improved valid loss = {avg_batch_loss}')
                best_model_state = self.state_dict()  # Save the best model state
                best_valid_loss = avg_batch_loss
                patience_counter = 0
            if patience_counter == self.params['patience']:
                print(
                    f'for {self.params["patience"]} epochs model has not improved, training stopped')
                break

        # Convert lists to numpy arrays for further processing if needed
        valid_losses_sensor = pd.DataFrame(valid_losses_sensor).values
        training_losses_sensor = pd.DataFrame(training_losses_sensor).values

        # Load the best model state if one was saved
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def evaluate(self, data_loader):
        # Set the model to evaluation mode, which disables dropout and batch
        # normalization
        self.eval()
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
                _, x_batch_estimate = self(x_batch)

                # Compute sensor-specific losses without affecting gradients
                valid_loss_batch_sensor = utilities.sensor_specific_loss(
                    self.params['criterion'],
                    x_batch,
                    x_batch_estimate,
                    self.window_lengths,
                    self.num_channels)
                valid_loss_batch_sensor = [
                    torch.mean(single_sensor_vec) for single_sensor_vec in valid_loss_batch_sensor]

                # Calculate the mean loss for the batch (scalar)
                valid_loss_batch = torch.mean(utilities.overall_loss(
                    self.params['criterion'], x_batch, x_batch_estimate))
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

    def test(self, test_data_loader, Y_test, criterion, aggregation_type):
        # Initialize lists to store various metrics
        sensor_losses_fusing = []
        sensor_losses_individual = []
        total_loss = []
        flattened_inputs = []
        predictions = []
        embeddings = []

        # Set model to evaluation mode
        self.eval()
        for batch_idx, (x_batch, _) in enumerate(test_data_loader):
            # Flatten and concatenate input data for processing
            x_batch = torch.concat([x.flatten(1) for x in x_batch], axis=1)

            # Get model outputs including embeddings and predictions
            embedding, x_batch_estimate = self(x_batch)

            # Compute sensor-specific losses and convert them to NumPy for
            # easier manipulation
            sensor_loss_batch = torch.stack(
                utilities.sensor_specific_loss(
                    criterion,
                    x_batch,
                    x_batch_estimate,
                    self.window_lengths,
                    self.num_channels)).detach().cpu().numpy()
            sensor_losses_fusing.append(sensor_loss_batch.T)

            # Get individual sensor losses using a utility function
            sensor_loss_batch_individual = utilities.get_individual_losses(
                self, self.sensors, self.window_lengths, self.num_channels, x_batch, criterion)
            sensor_losses_individual.append(sensor_loss_batch_individual.T)

            # Compute total loss for the batch and append to the list
            total_loss.append(
                criterion(x_batch, x_batch_estimate).detach().cpu().numpy())
            flattened_inputs.append(x_batch.detach().cpu().numpy())
            predictions.append(x_batch_estimate.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())

        # Concatenate arrays for the whole test dataset
        flattened_inputs = np.concatenate(flattened_inputs, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)

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
        Y_test_combined = pd.concat([pd.Series(Y_test), anomaly_scores_df], axis=1)

        # Group by segment_id and aggregate as specified
        Y_test_grouped = utilities.group_by_segment_id(
            Y_test_combined, anomaly_scores_df.columns, aggregation_type)

        # Calculate AUC for each anomaly score column
        results = {}
        for column in anomaly_scores_df.columns:
            results[column] = utilities.calculate_single_auc(
                Y_test_grouped, anomaly_score_column=column)

        # Assemble results into DataFrame and adjust index
        results = pd.concat(results)  # Concatenate results for each column
        results.index.names = ['column_names', 'duplicate']
        results.index = results.index.droplevel(
            'duplicate')  # Simplify index for clarity

        return results
