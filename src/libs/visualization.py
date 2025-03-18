"""Visualization functions"""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use("science")


def plot_losses(train_losses: np.ndarray, valid_losses: np.ndarray) -> None:
    """
    Plot the evolution of train and valid losses.

    Args:
        train_losses (np.ndarray): Train losses.
        valid_losses (np.ndarray): Validation losses.
    """
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Training and Validation Losses")
    plt.grid(visible=True, which="major", axis="y")
    plt.xlabel("Epoch")
    plt.xticks([x for x in range(0, train_losses.shape[0] - 1) if x % 2 == 0])
    plt.ylabel("Loss")
    plt.ylim((4, 10))
    plt.legend(loc="upper right")
    plt.show()
