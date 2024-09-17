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