from collections import Counter
from itertools import pairwise

import regex as re
from tqdm import tqdm


def count_byte_pairs(word_freq: list[tuple[list[int], int]]) -> Counter:
    bp_cnt = Counter()
    for ids, freq in word_freq:
        for a, b in pairwise(ids):
            bp_cnt[(a, b)] += freq
    return bp_cnt


def merge_byte_pair(word_freq: list[tuple[list[int], int]], byte_pair: tuple[int, int], new_id: int) -> list[tuple[list[int], int]]:
    new_word_freq = []
    for token_ids, freq in word_freq:
        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == byte_pair[0] and token_ids[i + 1] == byte_pair[1]:
                new_token_ids.append(new_id)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        if len(new_token_ids) > 1:
            new_word_freq.append((new_token_ids, freq))
    return new_word_freq


def train_bbpe(corpus_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size > 256, "vocab_size 必须大于 256"

    with open(corpus_path, "r", encoding="utf-8") as fp:
        corpus = fp.read()

    special_split_pattern = "|".join(special_tokens)
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freq = Counter()
    for corpus_part in re.split(special_split_pattern, corpus):
        word_freq.update(re.findall(pretokenization_pattern, corpus_part))
    word_freq = [(list(k.encode("utf-8")), v) for k, v in word_freq.items()]  # 用 list 存储 word 的编码和对应计数
    word_freq = [(word_ids, cnt) for word_ids, cnt in word_freq if len(word_ids) > 1]  # 滤除无需过滤的字符

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges = []
    total_merges = vocab_size - len(vocab)

    # 使用tqdm创建进度条
    with tqdm(total=total_merges, desc="Building vocabulary") as pbar:
        while len(vocab) < vocab_size and word_freq:
            bp_cnt = count_byte_pairs(word_freq)

            # 这段逻辑只是为了过 test 用例：When computing merges, deterministically
            # break ties in pair frequency by preferring the lexicographically greater pair.
            max_freq = -1
            idx1 = idx2 = 0
            for (p1, p2), freq in bp_cnt.items():
                if freq > max_freq:
                    max_freq = freq
                    idx1, idx2 = p1, p2
                elif freq == max_freq and (vocab[p1], vocab[p2]) > (vocab[idx1], vocab[idx2]):
                    idx1, idx2 = p1, p2

            vocab[next_id] = vocab[idx1] + vocab[idx2]
            merges.append((vocab[idx1], vocab[idx2]))
            word_freq = merge_byte_pair(word_freq, (idx1, idx2), next_id)
            next_id += 1
            pbar.update(1)
    return vocab, merges


if __name__ == "__main__":
    V, M = train_bbpe("data/TinyStoriesV2-GPT4-valid.txt", 500, ["<|endoftext|>"])
