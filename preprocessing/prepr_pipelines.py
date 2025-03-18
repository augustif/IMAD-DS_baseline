import numpy as np
import torch

class Normalize:

    def __init__(self, norm='std'):
        self.norm = norm

    def transform(self, X):
        X = normalize_data(X, self.norm)
        return X

class NormaAndAddPosEnc:

    def __init__(self, norm='std', device='cpu', pos_enc_type='classic'):
        self.norm = norm
        self.pos_enc_type = pos_enc_type
        self.device = device
        self.pos_enc = None
        
        
    def transform(self, X):
        if self.pos_enc is None:
            self.pos_enc = calc_sensors_positional_encodings(X)
            self.pos_enc = [pe.to(self.device) for pe in self.pos_enc]  # Ensure pos_enc is moved to the specified device
        X = normalize_data(X, self.norm)
        X = apply_positional_encodings_to_sensors(X, self.pos_enc)
        return X

class PreprocessingPipeline:

    Pipelines = {
        'Normalize': Normalize,
        'NormalizeAndAddPositionalEncodings': NormaAndAddPosEnc
    }

    def __init__(self, pipeline_name, **kwargs):
        pipeline_class = self.Pipelines[pipeline_name]
        pipeline_params = {k: v for k, v in kwargs.items() if k in pipeline_class.__init__.__code__.co_varnames}
        self.pipeline = pipeline_class(**pipeline_params)

def normalize_data(X, normalization):
    """
    Normalize the training, validation, and test datasets using the specified normalization method.

    Parameters:
    X (list): list of n sensors numpy arrays, with shape (num_windows, num_channels, window_lenght) .
    normalization (str): Normalization method ('std', 'min-max', 'std_window', or 'min-max_window').

    Returns:
    tuple: Normalized datasets.
    """

    sensor_count = len(X)
    channel_counts = [X[i].shape[1] for i in range(sensor_count)]

    if normalization == 'std':
        # Calculate means and standard deviations across all samples and windows for each sensor
        means_ = [X[i].mean(2).mean(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]
        stds_ = [X[i].std(2).mean(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]

        # Standardize each dataset using the calculated means and standard deviations
        X = [standardize(X[i], means_[i], stds_[i])
                for i in range(sensor_count)]

    elif normalization == 'min-max':
        # Calculate min and max values across all samples and windows for each sensor
        mins_ = [X[i].min(2).min(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]
        maxs_ = [X[i].max(2).max(0).reshape(
            1, channel_counts[i], 1) for i in range(sensor_count)]

        # Apply min-max scaling to each dataset using the calculated min and max values
        X = [min_max_scale(X[i], mins_[i], maxs_[i])
                for i in range(sensor_count)]

    elif normalization == 'std_window':
        # Apply standardization within each window for each sensor
        X = [standardize_window(X[i]) for i in range(sensor_count)]

    elif normalization == 'min-max_window':
        # Apply min-max scaling within each window for each sensor
        X = [min_max_scale_window(X[i])
                for i in range(sensor_count)]
    
    return X
  
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

def apply_positional_encodings_to_sensors(X, pos_enc):
    """
    """
    S = len(X)

    # Add the positional encoding to the data
    X = [X[i] + pos_enc[i] for i in range(S)]
    return X

def calc_sensors_positional_encodings(X, type = 'classic'):
    """
    Add positional encoding to the input data.
    X (list): list of n sensors numpy arrays, with shape (num_windows, num_channels, window_lenght) .
    """
    S = len(X) #number of sensors
    C = [X[i].shape[1] for i in range(S)] # number of channels (n_sensors, n_channels)
    L = [X[i].shape[2] for i in range(S)] # window lengths (n_sensors, window_lengths)

    # Create a positional encoding matrix for each sensor
    pos_enc = [torch.zeros((C[i], L[i])) for i in range(S)]
    
    # for i in range(S):
    #     for k in range(L[i]):
    #         if k % 2 == 0:
    #             pos_enc[i][k] = np.sin(k)
    #         else:
    #             pos_enc[i][k] = np.cos(k)
    for i in range(S):
        pos_enc[i] = calc_positional_encodings(X[i], type = type)
    
    return pos_enc

def calc_positional_encodings(x, type = 'classic'):
    """
    Calculate positional encodings for the input data.
    x (numpy.ndarray): Input data of shape (N, C, L), where N is the number of samples,
                       C is the number of channels, and L is the window length.
    
    Returns:
    numpy.ndarray: Positional encodings for the input data.
    """
    N, C, L = x.shape
    # Create a positional encoding matrix
    pos_enc = torch.zeros((C, L))

    if type == 'classic':
        max = torch.pi/2
        for k in range(L):
            if k % 2 == 0:
                pos_enc[:, k] = torch.sin(torch.tensor(k)/L*max)
            else:
                pos_enc[:, k] = torch.cos(torch.tensor(k)/L*max)
    elif type == 'linear':
        pos_enc[:, k] = torch.linspace(0, 1, L)
            
    return pos_enc

