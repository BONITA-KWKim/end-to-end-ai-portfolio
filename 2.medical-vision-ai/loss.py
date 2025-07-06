import torch.nn as nn

def create_loss_fn(name="cross_entropy"):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
