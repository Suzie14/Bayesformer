"""Functions for preprocessing (learn to do additions)."""

import random
import torch
from src.configs import constants
from src.libs.tokenizer import CharacterLevelTokenizer


def sample_datapoint(num_digits: int = 3) -> tuple[str, str]:
    """
    Generate a single datapoint for addition.

    Args:
        num_digits (int, optional): The number of digits for each number. Defaults to 3.

    Returns:
        tuple[str, str]: The generated datapoint in the format (prompt, answer).
    """
    a_list = [random.randint(0, 9) for _ in range(num_digits)]
    b_list = [random.randint(0, 9) for _ in range(num_digits)]
    a_int = int("".join([str(x) for x in a_list]))
    b_int = int("".join([str(x) for x in b_list]))
    a_str = "".join([str(x) for x in a_list])
    b_str = "".join([str(x) for x in b_list])
    sum_int = a_int + b_int
    return (a_str + "+" + b_str + "=", str(sum_int))


def create_dataset(nb_samples: int, num_digits: int = 3) -> list[tuple[str, str]]:
    """
    Create a dataset of addition problems.

    Args:
        nb_samples (int): The number of samples to generate.
        num_digits (int, optional): The number of digits for each number. Defaults to 3.

    Returns:
        list[tuple[str, str]]: The generated dataset.
    """
    return [sample_datapoint(num_digits) for _ in range(nb_samples)]


def pad(
    tokenizer: CharacterLevelTokenizer,
    token_list: list[int],
    type_list: str = "prompts",
) -> tuple[list[int], int]:
    """
    Pad the token list to the same length.

    Args:
        tokenizer (CharacterLevelTokenizer): The tokenizer to use for padding.
        token_list (list[int]): The list of tokens to pad.
        type_list (str, optional): The type of list to pad. Defaults to "prompts".

    Returns:
        tuple[list[int], int]: The padded list of tokens and the maximum length.
    """
    max_length = max([len(x) for x in token_list])
    out = []
    for x in token_list:
        if type_list == "prompts":
            out.append(
                [tokenizer.token_to_id[constants.PAD_TOKEN]] * (max_length - len(x)) + x
            )
        if type_list == "answers":
            out.append(
                x
                + [tokenizer.token_to_id[constants.EOS_TOKEN]]
                + [tokenizer.token_to_id[constants.PAD_TOKEN]] * (max_length - len(x))
            )
    return out, max_length


def get_batch(
    dataset: list[tuple[str, str]],
    tokenizer: CharacterLevelTokenizer,
    i: int,
    batch_size: int,
) -> tuple[list[int], list[int], int, int, str, str]:
    """
    Get a batch of data from the dataset.

    Args:
        dataset (list[tuple[str, str]]): The dataset to get the batch from.
        tokenizer (CharacterLevelTokenizer): The tokenizer to use for encoding.
        i (int): The starting index of the batch.
        batch_size (int): The batch size.

    Returns:
        tuple[list[int], list[int], int, int, str, str]: The encoded prompts, encoded answers, prompt length, answer length, prompts, and answers.
    """
    prompts = [dataset[i][0] for i in range(i, i + batch_size)]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    padded_prompts, prompt_length = pad(tokenizer, encoded_prompts, "prompts")
    answers = [dataset[i][1] for i in range(i, i + batch_size)]
    encoded_answers = [tokenizer.encode(answer) for answer in answers]
    padded_answers, answers_length = pad(tokenizer, encoded_answers, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, prompt_length, answers_length, prompts, answers
