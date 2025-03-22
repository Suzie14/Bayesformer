"""Functions for preprocessing (learn to do calculus)."""

import random
import torch

from src.configs import constants
from src.libs.tokenizer import CharacterLevelTokenizer


def sample_datapoint(num_digits: int = 3) -> tuple[str, str]:
    a_list = [random.randint(0, 9) for _ in range(num_digits)]
    b_list = [random.randint(0, 9) for _ in range(num_digits)]
    a_int = int("".join([str(x) for x in a_list]))
    b_int = int("".join([str(x) for x in b_list]))
    a_str = "".join([str(x) for x in a_list])
    b_str = "".join([str(x) for x in b_list])
    sum_int = a_int + b_int
    return (a_str + "+" + b_str + "=", str(sum_int))


def create_dataset(
    nb_samples: int, num_digits: int = 3
) -> list[tuple[str, str]]:
    return [sample_datapoint(num_digits) for _ in range(nb_samples)]


def pad(
    tokenizer: CharacterLevelTokenizer,
    token_list: list[int],
    type_list: str = "prompts",
) -> tuple[list[int], int]:
    max_length = max([len(x) for x in token_list])
    out = []
    for x in token_list:
        if type_list == "prompts":
            out.append(
                [tokenizer.token_to_id[constants.PAD_TOKEN]]
                * (max_length - len(x))
                + x
            )
        if type_list == "answers":
            out.append(
                x
                + [tokenizer.token_to_id[constants.EOS_TOKEN]]
                + [tokenizer.token_to_id[constants.PAD_TOKEN]]
                * (max_length - len(x))
            )
    return out, max_length


def get_batch(
    dataset: list[tuple[str, str]],
    tokenizer: CharacterLevelTokenizer,
    i: int,
    batch_size: int,
) -> tuple[list[int], list[int], int, int, str, str]:
    prompts = [dataset[i][0] for i in range(i, i + batch_size)]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    padded_prompts, prompt_length = pad(tokenizer, encoded_prompts, "prompts")

    answers = [dataset[i][1] for i in range(i, i + batch_size)]
    encoded_answers = [tokenizer.encode(answer) for answer in answers]
    padded_answers, answers_length = pad(tokenizer, encoded_answers, "answers")

    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, prompt_length, answers_length, prompts, answers

    data = data_train if split == "train" else data_test
    prompts = [tokenizer.encode(data[i][0]) for i in range(i, i + batch_size)]
    padded_prompts, length_prompts = pad(prompts, "prompts")
    answers = [tokenizer.encode(data[i][1]) for i in range(i, i + batch_size)]
    padded_answers, length_answers = pad(answers, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers
