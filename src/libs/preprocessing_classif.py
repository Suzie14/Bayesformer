"""Functions for preprocessing (Bayesian classification illustration)."""

from sklearn.datasets import make_moons
import torch
import torch.utils.data as data

from src.configs import constants


def get_data(
    nb_samples: int = 1000, noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = make_moons(
        n_samples=nb_samples, noise=noise, random_state=constants.RANDOM_SEED
    )
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    X, y = X.type(torch.float), y.type(torch.float)
    return X, y


def get_dataloader(
    X: torch.Tensor, y: torch.Tensor, batch_size: int = 32
) -> data.DataLoader:
    torch_train_dataset = data.TensorDataset(X, y)  # create your datset
    return data.DataLoader(torch_train_dataset, batch_size=batch_size)
