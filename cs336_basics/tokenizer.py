from collections import Counter
from itertools import pairwise
from typing import BinaryIO
import regex as re
from tqdm import tqdm
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries

def chunk_corpus(file: BinaryIO, num_chunks: int, split_special_token: bytes) -> list[str]:
    boundaries = find_chunk_boundaries(file, num_chunks, split_special_token)

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    ret = []
    for start, end in pairwise(boundaries):
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        ret.append(chunk)
    return ret

def process(args: tuple[str, str, str]) -> Counter:
    text, special_split_pattern, pretokenization_pattern = args
    word_freq = Counter()
    for part in re.split(special_split_pattern, text):
        word_freq.update(re.findall(pretokenization_pattern, part))
    return word_freq

def parallel_word_frequency(text_list, special_tokens, num_proc):
    special_split_pattern = "|".join(special_tokens)
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with Pool(num_proc) as pool:
        results = pool.map(
            process,
            [(text, special_split_pattern, pretokenization_pattern) for text in text_list]
        )

    word_freq = sum(results, Counter())
    word_freq = [(list(k.encode("utf-8")), v) for k, v in word_freq.items()]
    word_freq = [(word_ids, cnt) for word_ids, cnt in word_freq if len(word_ids) > 1]
    return word_freq

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


def train_bbpe(corpus_path: str, vocab_size: int, special_tokens: list[str], num_proc: int=1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size > 256, "vocab_size 必须大于 256"

    with open(corpus_path, "rb") as fp:
        corpus_list = chunk_corpus(fp, num_proc, b"<|endoftext|>")

    # word_freq = Counter()
    # for corpus_part in re.split(special_split_pattern, corpus):
    #     word_freq.update(re.findall(pretokenization_pattern, corpus_part))
    # word_freq = [(list(k.encode("utf-8")), v) for k, v in word_freq.items()]  # 用 list 存储 word 的编码和对应计数
    # word_freq = [(word_ids, cnt) for word_ids, cnt in word_freq if len(word_ids) > 1]  # 滤除无需过滤的字符
    word_freq = parallel_word_frequency(corpus_list, special_tokens, num_proc)

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
    V, M = train_bbpe("data/TinyStoriesV2-GPT4-valid.txt", 12800, ["<|endoftext|>"], 8)
    from IPython import embed

    embed()
