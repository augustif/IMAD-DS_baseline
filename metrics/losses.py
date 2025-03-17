import torch

def ME(tensor1, tensor2):
    return torch.mean(tensor1 - tensor2, dim=1)

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
