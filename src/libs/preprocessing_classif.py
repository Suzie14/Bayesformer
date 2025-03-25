"""Functions for preprocessing (Bayesian classification illustration)."""

from sklearn.datasets import make_moons
import torch
import torch.utils.data as data

from src.configs import constants


def get_data(
    nb_samples: int = 1000, noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for Bayesian classification illustration.

    Args:
        nb_samples (int, optional): The number of samples to generate. Defaults to 1000.
        noise (float, optional): The noise level to use for generating the data. Defaults to 0.1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The generated data (X, y).
    """
    X, y = make_moons(
        n_samples=nb_samples, noise=noise, random_state=constants.RANDOM_SEED
    )
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    X, y = X.type(torch.float), y.type(torch.float)
    return X, y


def get_dataloader(
    X: torch.Tensor, y: torch.Tensor, batch_size: int = 32
) -> data.DataLoader:
    """
    Create a DataLoader for the given data.

    Args:
        X (torch.Tensor): The input features.
        y (torch.Tensor): The target labels.
        batch_size (int, optional): The batch size to use for the DataLoader. Defaults to 32.

    Returns:
        data.DataLoader: The created DataLoader.
    """
    torch_train_dataset = data.TensorDataset(X, y)  # create your dataset
    return data.DataLoader(torch_train_dataset, batch_size=batch_size)
