import torch.optim as optim

def create_optimizer(model, name="adam", lr=0.001, momentum=0.9):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
