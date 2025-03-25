"""Utility functions."""

import numpy as np
import torch
import torch.nn as nn

from src.libs import preprocessing
from src.model.transformer import MyTransformer
from src.libs.tokenizer import Tokenizer


def train(
    model: MyTransformer,
    train_dataset: list[tuple[str, str]],
    valid_dataset: list[tuple[str, str]],
    nb_epochs: int,
    batch_size: int,
    lr: float,
    vocab_size: int,
    tokenizer: Tokenizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the model.

    Args:
        model (MyTransformer): The model to train.
        train_dataset (list[tuple[str, str]]): The training dataset.
        valid_dataset (list[tuple[str, str]]): The validation dataset.
        nb_epochs (int): The number of epochs to train for.
        batch_size (int): The batch size to use for training.
        lr (float): The learning rate to use for training.
        vocab_size (int): The size of the vocabulary.
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        device (str): The device to run the training on.

    Returns:
        tuple[np.ndarray, np.ndarray]: The training and validation losses.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    valid_losses = []
    best_valid_loss = float("inf")

    for epoch in range(1, nb_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # Training
        model.train()
        for batch_train, i in enumerate(range(0, len(train_dataset) - 1, batch_size)):
            prompts, target_answers, prompt_length, answers_length, _, _ = (
                preprocessing.get_batch(
                    dataset=train_dataset,
                    tokenizer=tokenizer,
                    i=i,
                    batch_size=batch_size,
                )
            )
            prompts = prompts.to(device)  # (prompt_length, batch_size)
            target_answers = target_answers.to(
                device
            )  # (answers_length + 1, batch_size)
            input_tensor = torch.cat(
                (prompts, target_answers), 0
            )  # (prompt_length + answers_length + 1, batch_size)
            model.zero_grad()
            output = model(
                input_tensor
            )  # (prompt_length + answers_length + 1, batch_size, vocab_size)
            output_answers = output[prompt_length - 1 : -1, :, :].reshape(
                -1, vocab_size
            )  # ((answers_length + 1) * batch_size, vocab_size)
            target_answers = target_answers.view(-1)
            loss = criterion(output_answers, target_answers)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch_valid, i in enumerate(
                range(0, len(valid_dataset) - 1, batch_size)
            ):
                prompts, target_answers, prompt_length, answers_length, _, _ = (
                    preprocessing.get_batch(
                        dataset=valid_dataset,
                        tokenizer=tokenizer,
                        i=i,
                        batch_size=batch_size,
                    )
                )
                prompts = prompts.to(device)  # (prompt_length, batch_size)
                target_answers = target_answers.to(
                    device
                )  # (answers_length + 1, batch_size)
                input_tensor = torch.cat(
                    (prompts, target_answers), 0
                )  # (prompt_length + answers_length + 1, batch_size)
                output = model(input_tensor)
                output_answers = output[prompt_length - 1 : -1, :, :].reshape(
                    -1, vocab_size
                )
                target_answers = target_answers.view(-1)
                loss = criterion(output_answers, target_answers)
                valid_loss += loss.item()

        train_loss = train_loss / len(train_dataset)
        valid_loss = valid_loss / len(valid_dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if nb_epochs < 10 or epoch % (nb_epochs // 10) == 0:
            print(
                f"EPOCH [{epoch} / {nb_epochs}] ----------- TRAIN LOSS : {train_loss:.4f}, VALID LOSS : {valid_loss:.4f}"
            )

        # Save model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
    print(f"Best valid loss : {best_valid_loss:.4f}")
    return np.array(train_losses), np.array(valid_losses)


def generate(
    model: MyTransformer, prompts: torch.Tensor, new_tokens: int, device: str
) -> torch.Tensor:
    """
    Generate new tokens using the model.

    Args:
        model (MyTransformer): The model to use for generation.
        prompts (torch.Tensor): The input prompts.
        new_tokens (int): The number of new tokens to generate.
        device (str): The device to run the generation on.

    Returns:
        torch.Tensor: The generated tokens.
    """
    input_tensor = prompts.to(device)  # (prompt_length, batch_size)
    for _ in range(new_tokens):
        output = model(input_tensor)  # (prompt_length, batch_size, ntokens)
        logits = output[-1, :, :]  # (batch_size, ntokens)
        tokens = torch.argmax(logits, -1).view((1, -1))  # (1, batch_size)
        input_tensor = torch.cat((input_tensor, tokens), 0)
    return input_tensor


def evaluate(
    model: MyTransformer,
    dataset: list[tuple[str, str]],
    tokenizer: Tokenizer,
    batch_size: int,
    device: str,
) -> float:
    """
    Evaluate the model on the given dataset.

    Args:
        model (MyTransformer): The model to evaluate.
        dataset (list[tuple[str, str]]): The dataset to evaluate on.
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        batch_size (int): The batch size to use for evaluation.
        device (str): The device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(dataset) - 1, batch_size)):
            prompts, target_answers, prompt_length, answers_length, _, _ = (
                preprocessing.get_batch(dataset, tokenizer, i, batch_size)
            )
            prompts = prompts.to(device)  # (prompt_length, batch_size)
            target_answers = target_answers.to(
                device
            )  # (answers_length + 1, batch_size)
            output = generate(
                model, prompts, answers_length + 1, device
            )  # (prompt_length + answers_length + 1, batch_size)
            answers_tokens = output[
                prompt_length:, :
            ]  # (answers_length + 1, batch_size), contains tokens
            equality_test = (
                answers_tokens == target_answers
            )  # (answers_length + 1, batch_size), contains boolean values
            correct += torch.all(equality_test, axis=0).float().sum()
    accuracy = correct / len(dataset)
    return accuracy.item()
