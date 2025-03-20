"""Class and functions for the tokenizers."""

import regex as re
import typing

from src.configs import constants

_CharacterLevelTokenizer = typing.TypeVar(
    "_CharacterLevelTokenizer", bound="CharacterLevelTokenizer"
)


class CharacterLevelTokenizer:
    """
    character-level
    """

    def __init__(self: _CharacterLevelTokenizer):
        self.vocab = (
            [str(x) for x in range(10)]
            + ["+", "="]
            + [constants.PAD_TOKEN, constants.EOS_TOKEN]
        )
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

    def clean(self: _CharacterLevelTokenizer, text: str) -> str:
        """
        removes all characters not in the vocabulary
        """
        out = re.sub(self.pattern, "", text)
        return out

    def pre_tokenization(self: _CharacterLevelTokenizer, text: str) -> list[str]:
        """
        character-level
        """
        return [c for c in text]

    def encode(self: _CharacterLevelTokenizer, text: str) -> list[int]:
        text_list = self.pre_tokenization(self.clean(text))
        return [self.token_to_id[c] for c in text_list]

    def decode(self: _CharacterLevelTokenizer, token_list: list[int]) -> str:
        return "".join([self.id_to_token[x] for x in token_list])
