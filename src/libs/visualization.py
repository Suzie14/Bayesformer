"""Visualization functions"""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from IPython import display

plt.style.use("science")


def plot_losses(
    train_losses: np.ndarray, valid_losses: np.ndarray, ylog: bool = False
) -> None:
    """
    Plot the evolution of train and valid losses.

    Args:
        train_losses (np.ndarray): Train losses.
        valid_losses (np.ndarray): Validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Training and Validation Losses")
    plt.grid(visible=True, which="major", axis="y")
    plt.xlabel("Epoch")
    plt.xticks([x for x in range(0, train_losses.shape[0] - 1) if x % 2 == 0])
    plt.ylabel("Loss")
    if ylog:
        plt.yscale("log")
    plt.legend(loc="upper right")
    plt.show()


def plot_comparison_losses(
    train_losses: list[np.ndarray],
    valid_losses: list[np.ndarray],
    names: list[str],
    ylog: bool = False,
) -> None:
    """
    Compare the evolution of train and valid losses for different models.

    Args:
        train_losses (list[np.ndarray]): List of train losses for different models.
        valid_losses (list[np.ndarray]): List of validation losses for different models.
        names (list[str]): List of names for different models.
        ylog (bool, optional): If True, use logarithmic scale for y-axis. Defaults to False.
    """
    assert (
        len(train_losses) == len(valid_losses) == len(names)
    ), "All lists must have the same length"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(train_losses)):
        axes[0].plot(train_losses[i], label=names[i])
        axes[1].plot(valid_losses[i], label=names[i])
    axes[0].set_title("Training Losses")
    axes[1].set_title("Validation Losses")
    axes[0].grid(visible=True, which="major", axis="y")
    axes[1].grid(visible=True, which="major", axis="y")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    if ylog:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    plt.show()


def plot_decision_boundary(
    model: torch.nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    model_type: str = "classic",
    nsamples: int = 100,
    nbh: int = 2,
    cmap: str = "RdBu",
):
    """
    Plot decision boundary for a given model.

    Args:
        model (torch.nn.Module): Model to plot decision boundary for.
        X (np.ndarray): Input data.
        Y (np.ndarray): Target data.
        model_type (str, optional): Type of model. Defaults to "classic".
        nsamples (int, optional): Number of samples. Defaults to 100.
        nbh (int, optional): Number of neighbors. Defaults to 2.
        cmap (str, optional): Colormap. Defaults to "RdBu".
    """
    h = 0.02 * nbh
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(
        np.arange(x_min * 2, x_max * 2, h), np.arange(y_min * 2, y_max * 2, h)
    )

    test_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(
        torch.FloatTensor
    )
    model.eval()
    with torch.no_grad():
        if model_type == "classic":
            pred = torch.sigmoid(model(test_tensor))
        else:
            outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
            for i in range(nsamples):
                outputs[i] = torch.sigmoid(model(test_tensor))
            pred = outputs.mean(0).squeeze()

    Z = pred.reshape(xx.shape).detach().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.cla()
    ax.set_title("Classification Analysis")
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z, colors="k", linestyles=":", linewidths=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="Paired_r", edgecolors="k")
    display.display(plt.gcf())
    display.clear_output(wait=True)
