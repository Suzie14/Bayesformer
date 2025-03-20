"""Classes and functions to make Bayesian classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


from torch.utils.data import DataLoader


_LinearVariational = typing.TypeVar(
    name="_LinearVariational", bound="LinearVariational"
)


class LinearVariational(nn.Module):
    def __init__(
        self: _LinearVariational,
        input_dimension: int,
        output_dimension: int,
        prior_var: float,
    ) -> None:
        super(LinearVariational, self).__init__()
        self.prior_var = prior_var
        self.w_mu = nn.Parameter(torch.zeros(input_dimension, output_dimension))
        self.w_rho = nn.Parameter(torch.zeros(input_dimension, output_dimension))
        self.b_mu = nn.Parameter(torch.zeros(output_dimension))

    def sampling(
        self: _LinearVariational, mu: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.randn_like(mu)
        std = std = torch.log1p(torch.exp(rho))
        return mu + eps * std

    def kl_divergence(self: _LinearVariational) -> torch.Tensor:
        prior_std = np.sqrt(self.prior_var)
        rho_theta = self.w_rho
        mu_theta = self.w_mu
        sigma_theta = torch.sqrt(torch.log(1 + torch.exp(rho_theta)))
        sigmap = torch.zeros(mu_theta.shape)
        sigmap.fill_(prior_std)
        divs = (
            torch.log(sigmap / sigma_theta)
            + (sigma_theta**2 + mu_theta**2) / (2 * sigmap**2)
            - 0.5
        )
        return torch.mean(divs)

    def forward(self: _LinearVariational, x: torch.Tensor) -> torch.Tensor:
        W = self.sampling(self.w_mu, self.w_rho)
        b = self.b_mu
        out = torch.matmul(x, W) + b
        return out


_VariationalMLP = typing.TypeVar(name="_VariationalMLP", bound="VariationalMLP")


class VariationalMLP(nn.Module):
    def __init__(
        self: _VariationalMLP,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        prior_var: float,
    ) -> None:
        super(VariationalMLP, self).__init__()
        self.hidden_layer = LinearVariational(
            input_dimension, hidden_dimension, prior_var
        )
        self.output_layer = LinearVariational(
            hidden_dimension, output_dimension, prior_var
        )

    def kl_divergence(self: _VariationalMLP) -> torch.Tensor:
        kl_hidden = self.hidden_layer.kl_divergence()
        kl_output = self.output_layer.kl_divergence()
        return kl_hidden + kl_output

    def forward(self: _VariationalMLP, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def train_model(
        self: _VariationalMLP,
        train_loader: DataLoader,
        nb_epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(nb_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self(x).squeeze()
                elbo = self.kl_divergence() + criterion(output, y)
                elbo.backward()
                optimizer.step()
        print(f"Training completed after {nb_epochs} epochs.")


_MLP = typing.TypeVar(name="_MLP", bound="MLP")


class MLP(nn.Module):
    def __init__(
        self: _MLP,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: bool = False,
        dropout_rate: float | None = None,
    ) -> None:
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, output_dimension)
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def forward(self: _MLP, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden_layer(x))
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.output_layer(x)
        return x

    def train_model(
        self: _MLP,
        train_loader: DataLoader,
        nb_epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(nb_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self(x).squeeze()
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        print(f"Training completed after {nb_epochs} epochs.")
