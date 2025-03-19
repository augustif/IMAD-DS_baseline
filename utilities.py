import torch
def get_device(verbose: int = 0):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if verbose>0:
        print(f"Using device: {device}")
    return device

def set_torch_seed(seed: int, device: str = 'cpu', verbose: int = 0):
    torch.manual_seed(seed)
    
    if device == 'mps':
        torch.mps.manual_seed(seed)
    elif device == 'cuda':
        torch.cuda.manual_seed(seed)
    elif device == 'cpu':
        torch.manual_seed(seed)
    else:
        raise ValueError(f"Wrong device value: {device}")
    
    if verbose>0:
        print(f"Setting torch seed to {seed}")
    
def get_sensors_enabled(cfg):
    return [sensor for sensor, value in cfg.sensors_enabled.items() if value==True]