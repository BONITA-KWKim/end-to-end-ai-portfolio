from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts
)

def get_scheduler(optimizer, name="cosine_warm_restart", **kwargs):
    """
    주어진 이름에 해당하는 학습률 스케줄러를 반환합니다.
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저 객체
        name (str): 사용할 스케줄러 이름
        kwargs: 각 스케줄러에 맞는 하이퍼파라미터

    Returns:
        torch.optim.lr_scheduler._LRScheduler
    """
    name = name.lower()

    if name == "step":
        return StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1))
    elif name == "exp":
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("t_max", 50), eta_min=kwargs.get("eta_min", 0))
    elif name == "cosine_warm_restart":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("t_0", 10),
            T_mult=kwargs.get("t_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-6)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
