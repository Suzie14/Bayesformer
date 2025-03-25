"""Class and functions for the tokenizers."""

import abc
import regex as re

from src.configs import constants


class Tokenizer(abc.ABC):
    """Abstract class for tokenizers."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, token_list: list[int]) -> str:
        raise NotImplementedError


###############################################################
#                                                             #
#                  CHARACTER-LEVEL TOKENIZER                  #
#                                                             #
###############################################################


class CharacterLevelTokenizer(Tokenizer):
    """
    character-level
    """

    def __init__(self):
        self.vocab = (
            [str(x) for x in range(10)]
            + ["+", "="]
            + [constants.PAD_TOKEN, constants.EOS_TOKEN]
        )
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

    def clean(self, text: str) -> str:
        """
        removes all characters not in the vocabulary
        """
        out = re.sub(self.pattern, "", text)
        return out

    def pre_tokenization(self, text: str) -> list[str]:
        """
        character-level
        """
        return [c for c in text]

    def encode(self, text: str) -> list[int]:
        text_list = self.pre_tokenization(self.clean(text))
        return [self.token_to_id[c] for c in text_list]

    def decode(self, token_list: list[int]) -> str:
        return "".join([self.id_to_token[x] for x in token_list])


###############################################################
#                                                             #
#                 REVERSED-PAIRING TOKENIZER                  #
#                                                             #
###############################################################


def remove_leading_zeros(s: str) -> str:
    """
    Remove the leading 0 of a string

    Args:
        s (str): string with leading 0

    Returns:
        str: string with removed leading 0
    """
    return s.lstrip("0") or "0"


def pad_with_0(s: str, m: int) -> str:
    """
    Given a string s and a integer m with m>=len(s),
    add m-len(s) "0" add the begining of s

    Args:
        s (str): string to be filled
        m (int): target length

    Returns:
        str: stirng filled with 0s.
    """
    n = max(0, m - len(s))
    first_shaped = "0" * n + s
    return first_shaped


def pad_equally_0(s1: str, s2: str) -> tuple[str]:
    """
    Given two strings s1 and s2, add 0 to the left
    of the smallest one two make them equally sized

    Args:
        s1 (str): input string
        s2 (str): input string

    Returns:
        tuple[str]: equally sized padded with 0 strings
    """
    s1_without_0 = remove_leading_zeros(s1)
    s2_without_0 = remove_leading_zeros(s2)
    m = max(len(s1_without_0), len(s2_without_0))
    s1_padded = pad_with_0(s1_without_0, m)
    s2_padded = pad_with_0(s2_without_0, m)
    return s1_padded, s2_padded


class Vocabulary:
    """
    Vocabulary class. Used for:
        - convert token to id
        - convert id to token
    """

    def __init__(self, method: str = "tens_pairing"):
        self.vocab = self.get_vocab(method=method)
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"
        self.ntokens = len(self.vocab)

    def get_vocab(self, method: str) -> list[str]:
        """
        Build a vocabulary. I have only implemented the
        vocabulary corresponding to the "gathering" technique, i.e.,
        the vocabulary is only composed of:
            - all the combinations of two digits ("00", "01",..."99"),
            - the one-length digit ("0",...,"9") to encode answer to the addition,
            - PAD and EOS tokens
        Args:
            method (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            list[str]: _description_
        """
        if method == "tens_pairing":
            return (
                [pad_with_0(str(x), 2) for x in range(100)]
                + [str(x) for x in range(10)]
                + ["+", "="]
                + [constants.PAD_TOKEN, constants.EOS_TOKEN]
            )
        else:
            raise NotImplementedError

    def clean(self, text: str) -> str:
        """
        Clean a string. To be use before encoding.

        Args:
            text (str): text to be cleaned

        Returns:
            str: cleaned text
        """
        out = re.sub(self.pattern, "", text)
        return out


class Encoder:
    """
    Encoder of the tokenizer. Only used for its "encode" method.
    """

    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        """
        Encode a given text. Designed to encode inputs of the form:
            - "10+23=33" (statement)
            - "10+23=" (addition)
            - "33" (result)

        Args:
            text (str): one of the mentionned inputs.

        Returns:
            list[int]: encoded list
        """
        text_cleaned = self.vocabulary.clean(text)
        # statment or addition input
        if "=" in text_cleaned:
            addition, result = text_cleaned.split("=")
            addition_encoded = self._encode_addition(addition)
            result_encoded = self._encode_number(result)
            # n.b.: result_encoded can be []
            return addition_encoded + result_encoded
        # result input
        else:
            return self._encode_number(text_cleaned)

    def _encode_addition(self, addition_str: str) -> list[int]:
        """
        Encode an addition (strin of the form "10+2=").

        Args:
            addition_str (str): addition to be encoded

        Returns:
            list[int]: list of encoded tokens
        """
        first, second = addition_str.split("+")
        # pad with 0 to get equally sized number ("21";"352" -> "021";"352")
        first_pad_0, second_pad_0 = pad_equally_0(first, second)
        # concatenante the corresponding tens ("021";"352"-> "03";"25";"12")
        list_tokens = [
            "".join([tok1, tok2]) for tok1, tok2 in zip(first_pad_0, second_pad_0)
        ]
        # encode the addition and reverse to bgin by the units
        encoded_addition = [
            self.vocabulary.token_to_id[token] for token in list_tokens
        ][::-1]
        # add the encoded sign equal at th end of the addition
        encoded_addition.append(self.vocabulary.token_to_id["="])
        return encoded_addition

    def _encode_number(self, number_str: str) -> list[int]:
        """
        Encode a number by encoding each of its digits.
        (Return [] if number_str was an empty string (used to encode addition).)

        Args:
            number_str (str): string number to be encoded

        Returns:
            list[int]: list of encoded tokens
        """

        return [self.vocabulary.token_to_id[digit] for digit in number_str[::-1]]


class Decoder:
    """Decoder of the tokenizer. Only used for its "decode" method."""

    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def _decode_end(self, tokens_txt: str) -> str:
        """
        Decodes the end of a statement, e.g.,
            "432[EOS][PAD][PAD]" -> "234[EOS][PAD][PAD]".

        Args:
            tokens_txt (str): _description_

        Returns:
            str: _description_
        """
        # input can be an empty list -> return empty string
        if not tokens_txt:
            return ""

        # if the input is a padded answer
        elif constants.EOS_TOKEN in tokens_txt:
            result, format_tokens = tokens_txt.split(constants.EOS_TOKEN)
            return "".join(
                [
                    remove_leading_zeros(result[::-1]),
                    constants.EOS_TOKEN,
                    format_tokens,
                ]
            )

        # if it is a simple number (not used in the training and eval procedure)
        else:
            return remove_leading_zeros(tokens_txt[::-1])

    def _decode_addition(self, tokens_txt: str) -> str:
        """
        Return the decoded string given a "gathered-format" string.
        For example, "[PAD][PAD]01023584=" -> "[PAD][PAD]38+1254=".

        Args:
            tokens_txt (str): "gathered-format" string

        Returns:
            str: decoded string
        """
        padded = constants.PAD_TOKEN in tokens_txt
        if padded:
            split_pad = tokens_txt.split(constants.PAD_TOKEN)
            n_pad = len(split_pad) - 1
            without_pad_token = split_pad[-1]

        else:
            without_pad_token = tokens_txt
            n_pad = 0

        first = without_pad_token[::2][::-1]
        second = without_pad_token[1::2][::-1]

        addition = "".join(
            [
                remove_leading_zeros(first),
                "+",
                remove_leading_zeros(second),
                "=",
            ]
        )
        decoded_addition = n_pad * constants.PAD_TOKEN + addition
        return decoded_addition

    def decode(self, ids_list: list[int]) -> str:
        """
        Returns the decoded string given a list of ids.

        Args:
            ids_list (list[int]): list of ids

        Returns:
            str: Decoded string
        """
        token_str = "".join([self.vocabulary.id_to_token[id_] for id_ in ids_list])
        # input is an addition ("1+1=") or a statement ("1+1=2")
        if "=" in token_str:
            # print(token_str)
            addition, result = token_str.split("=")
            return "".join([self._decode_addition(addition), self._decode_end(result)])

        # input is a simple number ("123") (not used in the procedure but might be useful)
        else:
            return self._decode_end(token_str)


class ReversedPairingTokenizer(Tokenizer):
    """Tokenizer class of the assignment"""

    def __init__(self):
        # build a vocabulary class
        self.vocabulary = Vocabulary()
        # set up the encoder and decoder attributes
        self.tokenizer_encoder = Encoder(self.vocabulary)
        self.tokenizer_decoder = Decoder(self.vocabulary)
        # argument needed by future cells in the notebook
        # not as "good practice" as I wanted but the less
        # worse
        self.token_to_id = self.vocabulary.token_to_id
        self.ntokens = self.vocabulary.ntokens

    def encode(self, text: str) -> list[int]:
        """
        Encode a given text according to the technique.

        Args:
            text (str): input text

        Returns:
            list[int]: encoded input text
        """
        return self.tokenizer_encoder.encode(text)

    def decode(self, encoded_tokens: list[int]) -> str:
        """
        Decode an list of id.

        Args:
            encoded_tokens (list[int]): list of ids

        Returns:
            str: decoded list of ids
        """
        return self.tokenizer_decoder.decode(encoded_tokens)
