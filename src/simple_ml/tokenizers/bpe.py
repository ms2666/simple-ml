from collections import Counter
from tqdm.auto import tqdm

# Add constants at the top
SINGLE_BYTE_TOKENS = 256
SPECIAL_TOKENS = {
    "<PAD>": 0,  # Padding token
    "<BOS>": 1,  # Beginning of sequence
    "<EOS>": 2,  # End of sequence
    "<UNK>": 3,  # Unknown token
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)


def get_stats(ids: list[int]) -> Counter[tuple[int, int]]:
    """
    Count frequency of consecutive pairs in the list of ids.

    Args:
        ids (list of int): List of token IDs.

    Returns:
        Counter: A Counter object mapping pairs to their frequencies.
    """
    return Counter(zip(ids[:-1], ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Merge a specified pair in the list of ids into a new id.

    Args:
        ids (list of int): Current list of token IDs.
        pair (tuple of int): The pair of IDs to merge.
        idx (int): The new ID to replace the merged pair.

    Returns:
        list of int: The updated list of token IDs after merging.
    """
    new_ids: list[int] = []
    i = 0

    while i < len(ids):
        # Check if the current and next IDs form the target pair
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(idx)
            i += 2  # Skip the next ID as it's part of the merged pair
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BasicTokenizer:
    def __init__(self, vocab_size: int):
        if vocab_size < NUM_SPECIAL_TOKENS + SINGLE_BYTE_TOKENS:
            raise ValueError(
                f"vocab_size must be at least {NUM_SPECIAL_TOKENS + SINGLE_BYTE_TOKENS} "
                f"({NUM_SPECIAL_TOKENS} special tokens + {SINGLE_BYTE_TOKENS} byte tokens)"
            )
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens = SPECIAL_TOKENS

        # Initialize special tokens first
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

        # Add reverse mapping for special tokens
        self.special_tokens_decoder = {
            token: idx for token, idx in self.special_tokens.items()
        }

    def split_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        """Split text into (is_special, text) pairs"""
        spans = []
        current = 0
        while current < len(text):
            found_special = False
            for token in self.special_tokens_decoder:
                if text[current:].startswith(token):
                    if current < len(text):
                        spans.append((True, token))
                    current += len(token)
                    found_special = True
                    break
            if not found_special:
                start = current
                while current < len(text) and not any(
                    text[current:].startswith(token)
                    for token in self.special_tokens_decoder
                ):
                    current += 1
                spans.append((False, text[start:current]))
        return spans

    def train(self, text: str) -> None:
        """
        Train the tokenizer on the provided text to build the vocabulary.

        Args:
            text (str): The input text to train on.
        """
        # Initialize vocabulary with single-byte tokens after special tokens
        self.vocab.update(
            {i + NUM_SPECIAL_TOKENS: bytes([i]) for i in range(SINGLE_BYTE_TOKENS)}
        )

        ids = list(text.encode("utf-8"))
        start_idx = NUM_SPECIAL_TOKENS + SINGLE_BYTE_TOKENS

        for i in tqdm(
            range(self.vocab_size - start_idx), desc="Training BPE", leave=False
        ):
            stats = get_stats(ids)
            if not stats:
                break
            pair, _ = stats.most_common(1)[0]
            idx = start_idx + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> tuple[list[int], list[int]]:
        """
        Encode a string into a list of token IDs based on the trained merges.

        Args:
            text (str): The input text to encode.

        Returns:
            list of int: The list of token IDs.
            list of int: The list of attention mask values.
        """
        ids = []
        for is_special, span in self.split_special_tokens(text):
            if is_special:
                ids.append(self.special_tokens_decoder[span])
            else:
                span_ids = [(b + NUM_SPECIAL_TOKENS) for b in span.encode("utf-8")]

                while len(span_ids) >= 2:
                    stats = get_stats(span_ids)
                    if not stats:
                        break
                    pair = min(
                        stats.keys(), key=lambda p: self.merges.get(p, float("inf"))
                    )
                    if pair not in self.merges:
                        break
                    span_ids = merge(span_ids, pair, self.merges[pair])
                ids.extend(span_ids)

        if add_special_tokens:
            ids = [self.special_tokens["<BOS>"]] + ids + [self.special_tokens["<EOS>"]]

        return ids, [1] * len(ids)

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding_style: str | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Encode a list of strings into a list of token IDs.

        Args:
            texts (list of str): List of input texts to encode.
            add_special_tokens (bool): Whether to add special tokens to each encoded sequence.
            padding_style (str): Padding style to use for sequences.

        Returns:
            list of list of int: List of token ID lists.
            list of list of int: List of attention mask lists.
        """
        if padding_style is None:
            tmp = [self.encode(text, add_special_tokens) for text in texts]
            return [ids for ids, _ in tmp], [mask for _, mask in tmp]

        if padding_style == "max_length":

            tmp = [self.encode(text, add_special_tokens) for text in texts]

            max_length = max(len(ids) for ids, _ in tmp)

            padded_token_ids = [self.pad_sequence(ids, max_length) for ids, _ in tmp]
            attention_masks = [mask + [0] * (max_length - len(mask)) for _, mask in tmp]

            return padded_token_ids, attention_masks

        raise ValueError(f"Invalid padding style: {padding_style}")

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of IDs back into a string.

        Args:
            ids (list of int): List of token IDs.
            skip_special_tokens (bool): Whether to skip special tokens in the output.

        Returns:
            str: The decoded string.
        """
        if skip_special_tokens:
            ids = [id for id in ids if id >= NUM_SPECIAL_TOKENS]

        tokens = []
        for idx in ids:
            if idx in self.special_tokens.values():
                token = next(k for k, v in self.special_tokens.items() if v == idx)
                tokens.append(token.encode("utf-8"))
            else:
                tokens.append(self.vocab.get(idx, b"?"))

        return b"".join(tokens).decode("utf-8", errors="replace")

    def decode_batch(self, sequences: list[list[int]], **kwargs) -> list[str]:
        """
        Decode a list of token ID sequences back into a list of strings.

        Args:
            sequences (list of list of int): List of token ID lists.

        Returns:
            list of str: List of decoded strings.
        """
        return [self.decode(ids, **kwargs) for ids in sequences if ids]

    def pad_sequence(self, ids: list[int], max_length: int) -> list[int]:
        """Pad or truncate sequence to max_length"""
        if len(ids) > max_length:
            return ids[:max_length]
        return ids + [self.special_tokens["<PAD>"]] * (max_length - len(ids))

    def __repr__(self) -> str:
        return f"BasicTokenizer(vocab_size={self.vocab_size})"

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens["<UNK>"]
