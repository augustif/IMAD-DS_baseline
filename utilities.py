import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

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


def group_by_segment_id(df, anomaly_score_columns, aggregation_type='mean'):
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
    grouping_dict = {
        'combined_label': 'first',  # Use the first value of 'combined_label' in each group
        'split_label': 'first',     # Use the first value of 'split_label' in each group
        'anomaly_label': 'first',   # Use the first value of 'anomaly_label' in each group
        'domain_shift_op': 'first',  # Use the first value of 'domain_shift_op' in each group
        'domain_shift_env': 'first'  # Use the first value of 'domain_shift_env' in each group
    }

    # Add the specified aggregation type for each anomaly score column
    for column in anomaly_score_columns:
        grouping_dict[column] = aggregation_type

    # Group the DataFrame by 'segment_id' and apply the aggregation
    grouped_df = df.groupby('segment_id').agg(grouping_dict).reset_index()

    return grouped_df


def load_dataset(train_path, test_path, label_names, sensors):
    """
    Load training and testing datasets from HDF5 files.

    Parameters:
    train_path (str): Path to the training dataset HDF5 file.
    test_path (str): Path to the testing dataset HDF5 file.
    label_names (list): List of label names to extract from the HDF5 files.
    sensors (list): List of sensor names to extract from the HDF5 files.

    Returns:
    tuple: A tuple containing the following elements:
        - X_train_raw (list): List of numpy arrays containing raw training data for each sensor.
        - Y_train_raw (pd.DataFrame): DataFrame containing training labels.
        - X_test (list): List of numpy arrays containing raw testing data for each sensor.
        - Y_test (pd.DataFrame): DataFrame containing testing labels.
    """
    with h5py.File(train_path, 'r') as f_train, h5py.File(test_path, 'r') as f_test:
        # Extract raw training data for each sensor
        X_train_raw = [f_train[sensor][:] for sensor in sensors]
        # Extract and decode training labels
        Y_train_raw = pd.DataFrame([[s.decode(
            'utf-8') for s in f_train[label_name][:].flatten()] for label_name in label_names]).T
        Y_train_raw.columns = label_names

        # Extract raw testing data for each sensor
        X_test = [f_test[sensor][:] for sensor in sensors]
        # Extract and decode testing labels
        Y_test = pd.DataFrame([[s.decode(
            'utf-8') for s in f_test[label_name][:].flatten()] for label_name in label_names]).T
        Y_test.columns = label_names

    return X_train_raw, Y_train_raw, X_test, Y_test


class CustomDataset(Dataset):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes:
    X (list): List of numpy arrays, where each array contains data from a different sensor.
    """

    def __init__(self, X):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        """
        self.X = X

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        int: Length of the dataset, which is the length of the first sensor's data.
        """
        return len(self.X[0])

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the specified index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        list: A list of samples from each sensor at the specified index.
        """
        return [x[idx] for x in self.X]


def standardize_window(data):
    """
    Standardize the data within each window for each channel.

    Parameters:
    data (numpy.ndarray): Input data of shape (N, C, L), where N is the number of samples,
                          C is the number of channels, and L is the window length.

    Returns:
    numpy.ndarray: Standardized data.
    """
    N, C, L = data.shape
    # Calculate mean and standard deviation for each window
    mean_ = data.mean(2).reshape(N, C, 1)
    std_ = data.std(2).reshape(N, C, 1)
    # Standardize data
    data -= mean_
    data /= std_ + 1e-5  # Adding a small value to avoid division by zero
    return data


def standardize(data, mean, std):
    """
    Standardize the data using provided mean and standard deviation.

    Parameters:
    data (numpy.ndarray): Input data to be standardized.
    mean (numpy.ndarray): Mean value for standardization.
    std (numpy.ndarray): Standard deviation value for standardization.

    Returns:
    numpy.ndarray: Standardized data.
    """
    return (data - mean) / std + 1e-5  # Adding a small value to avoid division by zero


def min_max_scale_window(data):
    """
    Apply min-max scaling to the data within each window for each channel.

    Parameters:
    data (numpy.ndarray): Input data of shape (N, C, L), where N is the number of samples,
                          C is the number of channels, and L is the window length.

    Returns:
    numpy.ndarray: Min-max scaled data.
    """
    N, C, L = data.shape
    # Calculate min and max for each window
    max_ = data.max(2).reshape(N, C, 1)
    min_ = data.min(2).reshape(N, C, 1)
    # Apply min-max scaling
    data -= min_
    # Adding a small value to avoid division by zero
    data /= (max_ - min_) + 1e-5
    return data


def min_max_scale(data, min_val, max_val):
    """
    Apply min-max scaling to the data using provided min and max values.

    Parameters:
    data (numpy.ndarray): Input data to be scaled.
    min_val (numpy.ndarray): Minimum value for scaling.
    max_val (numpy.ndarray): Maximum value for scaling.

    Returns:
    numpy.ndarray: Min-max scaled data.
    """
    return (data - min_val) / (max_val - min_val)


def normalize_data(X_train, X_valid, X_test, normalisation):
    """
    Normalize the training, validation, and test datasets using the specified normalization method.

    Parameters:
    X_train (list): List of numpy arrays for training data, one per sensor.
    X_valid (list): List of numpy arrays for validation data, one per sensor.
    X_test (list): List of numpy arrays for test data, one per sensor.
    normalisation (str): Normalization method ('std', 'min-max', 'std_window', or 'min-max_window').

    Returns:
    tuple: Normalized training, validation, and test datasets.
    """
    sensor_count = len(X_train)
    channel_counts = [X_train[i].shape[1] for i in range(sensor_count)]

    if normalisation == 'std':
        # Calculate means and standard deviations across all samples and windows for each sensor
        means_ = [X_train[i].mean(2).mean(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]
        stds_ = [X_train[i].std(2).mean(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]

        # Standardize each dataset using the calculated means and standard deviations
        X_train = [standardize(X_train[i], means_[i], stds_[i])
                   for i in range(sensor_count)]
        X_valid = [standardize(X_valid[i], means_[i], stds_[i])
                   for i in range(sensor_count)]
        X_test = [standardize(X_test[i], means_[i], stds_[i])
                  for i in range(sensor_count)]

    elif normalisation == 'min-max':
        # Calculate min and max values across all samples and windows for each sensor
        mins_ = [X_train[i].min(2).min(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]
        maxs_ = [X_train[i].max(2).max(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]

        # Apply min-max scaling to each dataset using the calculated min and max values
        X_train = [min_max_scale(X_train[i], mins_[i], maxs_[i])
                   for i in range(sensor_count)]
        X_valid = [min_max_scale(X_valid[i], mins_[i], maxs_[i])
                   for i in range(sensor_count)]
        X_test = [min_max_scale(X_test[i], mins_[i], maxs_[i])
                  for i in range(sensor_count)]

    elif normalisation == 'std_window':
        # Apply standardization within each window for each sensor
        X_train = [standardize_window(X_train[i]) for i in range(sensor_count)]
        X_valid = [standardize_window(X_valid[i]) for i in range(sensor_count)]
        X_test = [standardize_window(X_test[i]) for i in range(sensor_count)]

    elif normalisation == 'min-max_window':
        # Apply min-max scaling within each window for each sensor
        X_train = [min_max_scale_window(X_train[i])
                   for i in range(sensor_count)]
        X_valid = [min_max_scale_window(X_valid[i])
                   for i in range(sensor_count)]
        X_test = [min_max_scale_window(X_test[i]) for i in range(sensor_count)]

    return X_train, X_valid, X_test


def MSE(tensor1, tensor2):
    """
    Calculate the Mean Squared Error (MSE) between two tensors.

    Parameters:
    tensor1 (torch.Tensor): The first input tensor.
    tensor2 (torch.Tensor): The second input tensor.

    Returns:
    torch.Tensor: A tensor containing the MSE for each sample in the batch.
    """
    return torch.mean((tensor1 - tensor2) ** 2, dim=1)


def MAE(tensor1, tensor2):
    """
    Calculate the Mean Absolute Error (MAE) between two tensors.

    Parameters:
    tensor1 (torch.Tensor): The first input tensor.
    tensor2 (torch.Tensor): The second input tensor.

    Returns:
    torch.Tensor: A tensor containing the MAE for each sample in the batch.
    """
    return torch.mean(torch.abs(tensor1 - tensor2), dim=1)


def MAPE(tensor1, tensor2, epsilon=1e-10):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two tensors.

    Parameters:
    tensor1 (torch.Tensor): The first input tensor.
    tensor2 (torch.Tensor): The second input tensor.
    epsilon (float): A small value added to the denominator to avoid division by zero (default is 1e-10).

    Returns:
    torch.Tensor: A tensor containing the MAPE for each sample in the batch.
    """
    return torch.mean(torch.abs((tensor1 - tensor2) / (tensor1 + epsilon)) * 100, dim=1)
