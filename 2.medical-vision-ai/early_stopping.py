import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before stopping when no improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints messages when improvement occurs.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Args:
            score
            epoch (int): Current epoch.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"Initial score set at epoch {epoch}: {score:.4f}")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"No improvement at epoch {epoch}: {score:.4f} (Best: {self.best_score:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Improvement at epoch {epoch}: {score:.4f}")
