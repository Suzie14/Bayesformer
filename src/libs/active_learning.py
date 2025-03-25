"""Functions for active learning"""

import torch

from src.model.transformer import MyTransformer
from src.libs.tokenizer import Tokenizer
from src.libs import preprocessing


def compute_uncertainty(probs: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """
    Compute the uncertainty of the model's predictions.

    Args:
        probs (torch.Tensor): The predicted probabilities of the model.
        mode (str, optional): The mode to compute the uncertainty. Available modes are 'max', 'margin' and 'entropy'.
            Defaults to "max".

    Returns:
        torch.Tensor: The computed uncertainty.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode == "max":
        return torch.max(probs, dim=-1)[0]
    elif mode == "margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, 0] - sorted_probs[:, 1]
    elif mode == "entropy":
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Available modes are 'max', 'margin' and 'entropy'."
        )


def greedy_uncertainty(
    model: MyTransformer,
    prompts: torch.Tensor,
    new_tokens: int,
    device: str,
    mode: str = "max",
) -> torch.Tensor:
    """
    Compute the uncertainty of the model's predictions using a greedy approach.

    Args:
        model (MyTransformer): The model to compute the uncertainty for.
        prompts (torch.Tensor): The input prompts.
        new_tokens (int): The number of new tokens to generate.
        device (str): The device to run the computation on.
        mode (str, optional): The mode to compute the uncertainty. Defaults to "max".

    Returns:
        torch.Tensor: The computed uncertainty.
    """
    input_tensor = prompts.to(device)
    uncertainties = []
    for _ in range(new_tokens):
        output = model(input_tensor)  # (prompt_length, batch_size, ntokens)
        logits = output[-1, :, :]  # (batch_size, ntokens)
        tokens = torch.argmax(logits, -1).view((1, -1))  # (1, batch_size)
        input_tensor = torch.cat(
            (input_tensor, tokens), 0
        )  # (prompt_length + 1, batch_size)
        probs = torch.softmax(logits, dim=-1)  # (batch_size, ntokens)
        uncertainty = compute_uncertainty(probs, mode).unsqueeze(0)  # (batch_size,)
        uncertainties.append(uncertainty)
    return torch.cat(uncertainties, dim=0)


def sampling_uncertainty(
    model: MyTransformer,
    prompts: torch.Tensor,
    new_tokens: int,
    device: str,
    mode: str = "max",
    temperature: float = 0.8,
    nb_samples: int = 20,
) -> torch.Tensor:
    """
    Compute the uncertainty of the model's predictions using sampling.

    Args:
        model (MyTransformer): The model to compute the uncertainty for.
        prompts (torch.Tensor): The input prompts.
        new_tokens (int): The number of new tokens to generate.
        device (str): The device to run the computation on.
        mode (str, optional): The mode to compute the uncertainty. Defaults to "max".
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.8.
        nb_samples (int, optional): The number of samples to use for sampling. Defaults to 20.

    Returns:
        torch.Tensor: The computed uncertainty.
    """
    uncertainties = torch.zeros(new_tokens, prompts.shape[1]).to(device)
    for _ in range(nb_samples):
        input_tensor = prompts  # (prompt_length, batch_size)
        curr_uncertainties = []
        for _ in range(new_tokens):
            output = model(input_tensor)  # (prompt_length, batch_size, ntokens)
            logits = output[-1, :, :]  # (batch_size, ntokens)
            logits /= temperature  # (batch_size * nb_samples, ntokens)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, ntokens)
            tokens = torch.multinomial(probs, num_samples=1).view(
                (1, -1)
            )  # (1, batch_size)
            input_tensor = torch.cat(
                (input_tensor, tokens), dim=0
            )  # (prompt_length + 1, batch_size)
            uncertainty = compute_uncertainty(probs, mode).unsqueeze(0)  # (batch_size,)
            curr_uncertainties.append(uncertainty)
        uncertainties += torch.cat(curr_uncertainties, dim=0)
    return uncertainties / nb_samples


def dropout_uncertainty(
    model: MyTransformer,
    prompts: torch.Tensor,
    new_tokens: int,
    device: str,
    mode: str = "max",
    nb_samples: int = 20,
) -> torch.Tensor:
    """
    Compute the uncertainty of the model's predictions using dropout.

    Args:
        model (MyTransformer): The model to compute the uncertainty for.
        prompts (torch.Tensor): The input prompts.
        new_tokens (int): The number of new tokens to generate.
        device (str): The device to run the computation on.
        mode (str, optional): The mode to compute the uncertainty. Defaults to "max".
        nb_samples (int, optional): The number of samples to use for dropout. Defaults to 20.

    Returns:
        torch.Tensor: The computed uncertainty.
    """
    uncertainties = torch.zeros(new_tokens, prompts.shape[1]).to(device)
    for _ in range(nb_samples):
        input_tensor = prompts  # (prompt_length, batch_size)
        curr_uncertainties = []
        for _ in range(new_tokens):
            output = model(input_tensor)  # (prompt_length, batch_size, ntokens)
            logits = output[-1, :, :]  # (batch_size, ntokens)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, ntokens)
            tokens = torch.argmax(logits, -1).view((1, -1))  # (1, batch_size)
            input_tensor = torch.cat(
                (input_tensor, tokens), dim=0
            )  # (prompt_length + 1, batch_size)
            uncertainty = compute_uncertainty(probs, mode).unsqueeze(0)  # (batch_size,)
            curr_uncertainties.append(uncertainty)
        uncertainties += torch.cat(curr_uncertainties, dim=0)
    return uncertainties / nb_samples


def select_samples(
    model: MyTransformer,
    tokenizer: Tokenizer,
    pool_dataset: list[tuple[str, str]],
    device: str,
    nb_samples: int = 200,
    compute: str = "greedy",
    mode: str = "max",
) -> list[int]:
    """
    Select samples from the pool dataset to add to the training set.

    Args:
        model (MyTransformer): The model to use for selecting samples.
        tokenizer (Tokenizer): The tokenizer to use for preprocessing the dataset.
        pool_dataset (list[tuple[str, str]]): The pool dataset to select samples from.
        device (str): The device to run the computation on.
        nb_samples (int, optional): The number of samples to select. Defaults to 200.
        compute (str, optional): The method to compute the uncertainty. Available methods are 'greedy', 'sampling' and 'dropout'.
            Defaults to "greedy".
        mode (str, optional): The mode to compute the uncertainty. Defaults to "max".

    Returns:
        list[int]: The indices of the selected samples.

    Raises:
        ValueError: If the compute mode is unknown.
    """
    model.eval()
    with torch.no_grad():
        prompts, target_answers, prompt_length, answers_length, _, _ = (
            preprocessing.get_batch(pool_dataset, tokenizer, 0, len(pool_dataset))
        )
        prompts = prompts.to(device)  # (prompt_length, batch_size)
        target_answers = target_answers.to(device)  # (answers_length + 1, batch_size)
        if compute == "greedy":
            uncertainties = greedy_uncertainty(
                model, prompts, answers_length + 1, device, mode
            )
        elif compute == "sampling":
            uncertainties = sampling_uncertainty(
                model, prompts, answers_length + 1, device, mode
            )
        elif compute == "dropout":
            uncertainties = dropout_uncertainty(
                model, prompts, answers_length + 1, device, mode
            )
        else:
            raise ValueError(
                f"Unknown compute mode: {compute}. For Transformer, use 'greedy' or 'sampling'. For BayesFormer, use 'dropout'."
            )
        uncertainties = torch.mean(uncertainties, dim=0)  # (batch_size,)
        _, indices = torch.topk(uncertainties, nb_samples)  # (nb_samples,)
        return indices.tolist()


def create_new_train_set(
    train_dataset: list[tuple[str, str]],
    pool_dataset: list[tuple[str, str]],
    indices: list[int],
) -> list[tuple[str, str]]:
    """
    Create a new training set by adding selected samples from the pool dataset.

    Args:
        train_dataset (list[tuple[str, str]]): The current training dataset.
        pool_dataset (list[tuple[str, str]]): The pool dataset to select samples from.
        indices (list[int]): The indices of the selected samples.

    Returns:
        list[tuple[str, str]]: The new training dataset.
    """
    new_train_dataset = train_dataset.copy()
    for index in indices:
        new_train_dataset.append(pool_dataset[index])
    return new_train_dataset
