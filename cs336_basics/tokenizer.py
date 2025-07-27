from collections import defaultdict
from itertools import pairwise

import regex as re
from tqdm import tqdm

type BytePair = tuple[bytes, bytes]
type TokenizedWord = tuple[bytes, ...]
type TokenizedWordFreq = tuple[TokenizedWord, int]
type TokenizedWordFreqList = list[TokenizedWordFreq]
type Vocab = dict[int, bytes]
type Merges = list[BytePair]


def count_byte_pairs(word_freq_list: TokenizedWordFreqList) -> defaultdict[BytePair, int]:
    byte_pair_counter = defaultdict(int)
    for tokens, freq in word_freq_list:
        for a, b in pairwise(tokens):
            byte_pair_counter[(a, b)] += freq
    return byte_pair_counter


def merge_byte_pair(word_freq_list: TokenizedWordFreqList, byte_pair: BytePair) -> TokenizedWordFreqList:
    merged_word_freq = []
    merged_token = byte_pair[0] + byte_pair[1]
    p0, p1 = byte_pair

    for tokens, freq in word_freq_list:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if len(new_tokens) > 1:
            merged_word_freq.append((tuple(new_tokens), freq))
    return merged_word_freq


def train_bbpe(corpus_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[Vocab, Merges]:
    assert vocab_size > 256, "vocab_size 必须大于 256"

    with open(corpus_path, "r", encoding="utf-8") as fp:
        corpus = fp.read()

    special_split_pattern = "|".join(re.escape(token) for token in special_tokens)
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 统计词频
    word_counter = defaultdict(int)
    for segment in re.split(special_split_pattern, corpus):
        for word in re.findall(pretokenization_pattern, segment):
            word_counter[word] += 1

    # 转换为字节元组并过滤单字节单词
    word_freq_list: TokenizedWordFreqList = []
    for word, freq in word_counter.items():
        byte_tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
        if len(byte_tokens) > 1:
            word_freq_list.append((byte_tokens, freq))

    # 初始化词汇表（0-255为单字节）
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges = []
    total_merges = vocab_size - len(vocab)

    # 使用tqdm创建进度条
    with tqdm(total=total_merges, desc="Building vocabulary") as pbar:
        while len(vocab) < vocab_size and word_freq_list:
            byte_pair_counter = count_byte_pairs(word_freq_list)

            # 这段逻辑只是为了过 test 用例：When computing merges, deterministically
            # break ties in pair frequency by preferring the lexicographically greater pair.
            best_pair = max(byte_pair_counter.items(), key=lambda x: (x[1], x[0]))[0]

            # 更新词汇表和合并记录
            vocab[next_id] = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            next_id += 1

            word_freq_list = merge_byte_pair(word_freq_list, best_pair)
            pbar.update(1)
    return vocab, merges


if __name__ == "__main__":
    V, M = train_bbpe("data/TinyStoriesV2-GPT4-valid.txt", 500, ["<|endoftext|>"])
