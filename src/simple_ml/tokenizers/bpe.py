from collections import Counter
from tqdm.auto import tqdm


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


SINGLE_BYTE_TOKENS = 256


class BasicTokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train the tokenizer on the provided text to build the vocabulary.

        Args:
            text (str): The input text to train on.
            vocab_size (int): The desired size of the vocabulary.
        """
        # Initialize vocabulary with single-byte tokens
        ids = list(text.encode("utf-8"))
        # using 2^8 = 256 because a byte is 8 bits. So 255 is the max value a single byte can take
        self.vocab = {}
        self.vocab.update({i: bytes([i]) for i in range(SINGLE_BYTE_TOKENS)})
        self.merges = {}

        for i in tqdm(
            range(vocab_size - SINGLE_BYTE_TOKENS), desc="Training BPE", leave=False
        ):
            stats = get_stats(ids)
            if not stats:
                break
            # Select the most frequent pair
            pair, _ = stats.most_common(1)[0]
            idx = SINGLE_BYTE_TOKENS + i
            # Merge the selected pair
            ids = merge(ids, pair, idx)
            # Update merges and vocabulary
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of IDs back into a string.

        Args:
            ids (list of int): List of token IDs.

        Returns:
            str: The decoded string.
        """
        ids_to_bytes = [bytes(self.vocab.get(idx, b"?")) for idx in ids]
        return b"".join(ids_to_bytes).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs based on the trained merges.

        Args:
            text (str): The input text to encode.

        Returns:
            list of int: The list of token IDs.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair with the smallest merge index (earliest merged)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
