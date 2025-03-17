import numpy as np
import pandas as pd
import torch

def get_masked_batch(x_batch_concat, start, end):
    """
    Create a masked batch for a specific segment of the concatenated input batch.

    Parameters:
    x_batch_concat (torch.Tensor): Concatenated input batch for all sensors.
    start (int): Start index of the segment to be masked.
    end (int): End index of the segment to be masked.

    Returns:
    torch.Tensor: Masked input batch.
    """
    # Initialize a mask of zeros with the same shape as the input batch
    mask = torch.zeros_like(x_batch_concat)
    # Set the mask to 1 for the specified segment
    mask[:, start:end] = 1
    # Apply the mask to the input batch
    x_batch_concat_masked = x_batch_concat * mask
    return x_batch_concat_masked


def group_by_segment_id(df, anomaly_score_columns, aggregation_type='mean', verbose = 1):
    """
    Group a DataFrame by 'segment_id' and aggregate the specified columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing data to be grouped and aggregated.
    anomaly_score_columns (list): List of column names containing anomaly scores to be aggregated.
    aggregation_type (str): Type of aggregation to apply to anomaly score columns (default is 'mean').

    Returns:
    pd.DataFrame: DataFrame grouped by 'segment_id' with aggregated values.
    """
    # Define the aggregation operations for each column
    grouping_dict = {k:'first' for k in df.columns if k is not 'segment_id'}

    # Add the specified aggregation type for each anomaly score column
    for column in anomaly_score_columns:
        grouping_dict[column] = aggregation_type
    
    if verbose:
        print('grouping_dict', ':', grouping_dict)
        
    # Group the DataFrame by 'segment_id' and apply the aggregation
    grouped_df = df.groupby('segment_id').agg(grouping_dict).reset_index()

    return grouped_df

def sensor_specific_loss(criterion, x_batch_concat, x_batch_estimate, WINDOW_LENGTHS, NUM_CHANNELS):
    """
    Calculate the loss for each sensor separately.

    Parameters:
    criterion (function): Loss function to compute the loss.
    x_batch_concat (torch.Tensor): Concatenated input batch for all sensors.
    x_batch_estimate (torch.Tensor): Concatenated estimated output batch for all sensors.
    WINDOW_LENGTHS (list): List of window lengths for each sensor.
    NUM_CHANNELS (list): List of number of channels for each sensor.

    Returns:
    list: List of loss values for each sensor.
    """
    sensor_loss = []
    start = 0
    # Iterate over each sensor
    for i in range(len(WINDOW_LENGTHS)):
        # Calculate the length of the concatenated data for the current sensor
        single_sensor_concat_length = WINDOW_LENGTHS[i] * NUM_CHANNELS[i]
        # Extract the corresponding segment from the concatenated input and estimate
        single_sensor_concat = x_batch_concat[:,
                                              start:start+single_sensor_concat_length]
        single_sensor_estimate = x_batch_estimate[:,
                                                  start:start+single_sensor_concat_length]
        # Compute the loss for the current sensor and append to the list
        sensor_loss.append(
            criterion(single_sensor_concat, single_sensor_estimate))
        # Update the start index for the next sensor
        start += single_sensor_concat_length
    return sensor_loss


def overall_loss(criterion, x_batch_concat, x_batch_estimate):
    """
    Calculate the overall loss between the concatenated input batch and the estimated batch.

    Parameters:
    criterion (function): Loss function to compute the loss (e.g., MSE, MAE, MAPE).
    x_batch_concat (torch.Tensor): Concatenated input batch for all sensors.
    x_batch_estimate (torch.Tensor): Concatenated estimated output batch for all sensors.

    Returns:
    torch.Tensor: A scalar tensor representing the overall mean loss.
    """
    return torch.mean(criterion(x_batch_concat, x_batch_estimate))


def get_individual_losses(best_model, SENSORS, WINDOW_LENGTHS, NUM_CHANNELS, x_batch_concat, criterion):
    """
    Compute individual losses for each sensor using the best model.

    Parameters:
    best_model (torch.nn.Module): Trained model to estimate the outputs.
    SENSORS (list): List of sensor names.
    WINDOW_LENGTHS (list): List of window lengths for each sensor.
    NUM_CHANNELS (list): List of number of channels for each sensor.
    x_batch_concat (torch.Tensor): Concatenated input batch for all sensors.
    criterion (function): Loss function to compute the loss.

    Returns:
    numpy.ndarray: Array of individual losses for each sensor in the batch.
    """
    start = 0
    sensor_loss_batch_individual = []
    # Iterate over each sensor
    for i, sensor in enumerate(SENSORS):
        l = WINDOW_LENGTHS[i] * NUM_CHANNELS[i]
        end = start + l
        # Generate masked input batch for the current sensor
        masked_batch = get_masked_batch(x_batch_concat, start, end)
        # Get the model's output for the masked batch
        _, masked_estimate = best_model(masked_batch)
        # Compute the loss for the current sensor and append to the list
        individual_loss = sensor_specific_loss(
            criterion, masked_batch, masked_estimate, WINDOW_LENGTHS, NUM_CHANNELS)[i]
        sensor_loss_batch_individual.append(individual_loss)
        # Update the start index for the next sensor
        start += l
    # Stack the individual losses into a tensor, detach, and convert to numpy array
    sensor_loss_batch_individual = torch.stack(
        sensor_loss_batch_individual).detach().cpu().numpy()
    return sensor_loss_batch_individual


def calculate_single_auc(df, anomaly_score_column):
    """
    Calculate the Area Under the Curve (AUC) for anomaly detection scores.

    Parameters:
    df (pd.DataFrame): DataFrame containing anomaly scores and labels.
    anomaly_score_column (str): Column name of the anomaly scores in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the AUC scores for different domains.
    """
    # Create masks to filter the DataFrame based on split and anomaly labels
    source_mask = df['split_label'].isin(
        ['Normal_Source_Test', 'Anomaly_Source_Test'])
    target_mask = df['split_label'].isin(
        ['Normal_Target_Test', 'Anomaly_Target_Test'])
    normal_samples_mask = df['anomaly_label'] == 'normal'
    anormal_samples_mask = df['anomaly_label'] != 'normal'

    # Filter the DataFrame into normal and anomalous samples
    all_normal = df[normal_samples_mask]
    all_anormal = df[anormal_samples_mask]
    source_normal = df[source_mask & normal_samples_mask]
    source_anormal = df[source_mask & anormal_samples_mask]
    target_normal = df[target_mask & normal_samples_mask]
    target_anormal = df[target_mask & anormal_samples_mask]

    # Group domains for AUC calculation
    domain_names = ['all', 'source', 'target']
    domains = [
        [all_normal, all_anormal],
        [source_normal, source_anormal],
        [target_normal, target_anormal]
    ]

    AUC = {}
    # Calculate AUC for each domain
    for k in range(3):
        normal, anormal = domains[k]
        # Initialize an empty AUC matrix
        AUC_mtx = np.zeros((normal.shape[0], anormal.shape[0]))
        # Populate the AUC matrix
        for i, normal_sample in enumerate(normal[anomaly_score_column]):
            for j, anormal_sample in enumerate(anormal[anomaly_score_column]):
                # Compare normal and anomalous samples to compute AUC
                AUC_mtx[i, j] = int(normal_sample < anormal_sample)
        # Calculate the AUC score for the current domain
        AUC_score = AUC_mtx.sum() / (AUC_mtx.shape[0] * AUC_mtx.shape[1])
        AUC[domain_names[k]] = AUC_score

    # Convert AUC dictionary to DataFrame
    AUC = pd.DataFrame(AUC, index=[anomaly_score_column])

    return AUC