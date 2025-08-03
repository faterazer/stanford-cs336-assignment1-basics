import heapq
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from typing import BinaryIO, Self

import regex as re
from tqdm import tqdm

multiprocessing.set_start_method("spawn", force=True)


class BytePair:
    def __init__(self, byte1: bytes, byte2: bytes) -> None:
        """
        包装 bytes 类型，用于搭配 heapq 实现最大堆，这段只是为了满足作业要求，保证过 test 用例：When computing merges,
        deterministically break ties in pair frequency by preferring the lexicographically greater pair.
        """
        self.byte1 = byte1
        self.byte2 = byte2

    def __lt__(self, other: Self) -> bool:
        return (self.byte1, self.byte2) > (other.byte1, other.byte2)

    def __eq__(self, other: Self) -> bool:
        return (self.byte1, self.byte2) == (other.byte1, other.byte2)

    def __str__(self) -> str:
        return f"{self.byte1} - {self.byte2}"

    def __repr__(self) -> str:
        return f"BytePair({self.byte1}, {self.byte2})"


type Vocab = dict[int, bytes]
type Merges = list[tuple[bytes, bytes]]


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if not isinstance(split_special_token, bytes):
        raise TypeError("Must represent special token as a bytestring")
    if len(split_special_token) <= 0:
        raise ValueError("split_special_token must not be empty")

    # Get total file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Handle case where file is smaller than desired chunks
    if file_size < desired_num_chunks:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    MINI_CHUNK_SIZE = 65536

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks)]
    chunk_boundaries.append(file_size)

    # Adjust boundaries to align with special token
    for i in range(1, len(chunk_boundaries) - 1):
        current_pos = chunk_boundaries[i]
        file.seek(current_pos)

        while True:
            mini_chunk = file.read(MINI_CHUNK_SIZE)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[i] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[i] = current_pos + found_at
                break
            current_pos += MINI_CHUNK_SIZE

        if chunk_boundaries[i] == file_size:  # 已达文件 EOF，提前退出
            chunk_boundaries = chunk_boundaries[: i + 1]
            break
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(file_path: str, start: int, end: int, special_split_pattern: re.Pattern, pretokenization_pattern: re.Pattern) -> Counter:
    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    return Counter(word for segment in re.split(special_split_pattern, chunk) for word in re.findall(pretokenization_pattern, segment))


def count_words_parallel(
    corpus_path: str, split_special_token: bytes, special_tokens: list[str], pretokenization_pattern: str, num_readers: int
) -> Counter:
    # 读取文件+分块
    with open(corpus_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_readers, split_special_token)

    # 并行统计词频
    special_split_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
    pretokenization_pattern = re.compile(pretokenization_pattern)
    tasks = [(corpus_path, start, end, special_split_pattern, pretokenization_pattern) for start, end in pairwise(boundaries)]
    word_counter = Counter()
    if num_readers > 1:
        print(f"Training with {num_readers} readers")
        with ProcessPoolExecutor(max_workers=num_readers) as executor:
            for result in executor.map(pretokenize_chunk, *zip(*tasks)):
                word_counter.update(result)
    else:
        for task in tasks:
            word_counter.update(pretokenize_chunk(*task))
    return word_counter


def train_bbpe(
    corpus_path: str, vocab_size: int, special_tokens: list[str], split_special_token: bytes, num_readers: int = 0
) -> tuple[Vocab, Merges]:
    if vocab_size <= 256:
        raise ValueError("Vocab size must be larger than 256")

    # 当 num_readers = 0 时，自动检测 cpu 核心数
    if num_readers <= 0:
        num_readers = os.cpu_count() or 1  # 检测失败时，兜底为 1

    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counter = count_words_parallel(corpus_path, split_special_token, special_tokens, pretokenization_pattern, num_readers)

    # 5 个核心数据结构
    # pair2freq：记录 pair 频数
    # pair2tokenized_words：对特定 pair，记录所有包含该 pair 的 word 在 tokenized_word_list 中的下标
    # tokenized_word_list：全部单词的列表
    # word2freq：记录 word 频数
    # pair_heap：最大堆，用于加速获取 best_pair
    pair2freq = defaultdict(int)
    pair2tokenized_word_indices = defaultdict(set)
    tokenized_word_list = []
    word2freq = defaultdict(int)
    for word, count in word_counter.items():
        word_bytes = word.encode("utf-8")
        if len(word_bytes) <= 1:  # 单字节 word 直接过滤
            continue
        word2freq[word_bytes] = count
        tokens = tuple(bytes([b]) for b in word_bytes)
        tokenized_word_list.append(tokens)
        idx = len(tokenized_word_list) - 1
        for p0, p1 in pairwise(tokens):
            pair2freq[(p0, p1)] += count
            pair2tokenized_word_indices[(p0, p1)].add(idx)
    pair_heap = [(-freq, BytePair(p0, p1)) for (p0, p1), freq in pair2freq.items()]
    heapq.heapify(pair_heap)

    # 初始化词汇表（0-255为单字节）
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges = []
    total_merges = vocab_size - len(vocab)

    with tqdm(total=total_merges, desc="Building vocabulary") as pbar:
        for _ in range(total_merges):
            freq = 0
            while pair_heap:
                freq, best_pair = heapq.heappop(pair_heap)
                freq, p0, p1 = -freq, best_pair.byte1, best_pair.byte2
                if freq == pair2freq.get((p0, p1), 0) and freq > 0:
                    break
                else:
                    freq = 0
            if not pair_heap and freq <= 0:
                print("数据训练完毕，提前退出")
                break
            merged_token = p0 + p1

            # 更新 vocab 和 merges
            vocab[next_id] = merged_token
            merges.append((p0, p1))
            next_id += 1

            # 更新核心数据结构
            affected_tokenized_word_indices = pair2tokenized_word_indices[(p0, p1)]
            for idx in list(affected_tokenized_word_indices):  # 用 list 避免运行时修改
                old_tokens = tokenized_word_list[idx]  # 保存旧 tokens
                new_tokens = []
                i = 0
                while i < len(old_tokens):
                    if i < len(old_tokens) - 1 and old_tokens[i] == p0 and old_tokens[i + 1] == p1:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(old_tokens[i])
                        i += 1
                tokenized_word_list[idx] = tuple(new_tokens)

                word = b"".join(new_tokens)  # 仍是原 word_bytes
                word_freq = word2freq[word]

                # 减去所有旧 pair
                for pair in pairwise(old_tokens):
                    pair2freq[pair] -= word_freq
                    pair2tokenized_word_indices[pair].discard(idx)
                    if pair2freq[pair] > 0:
                        heapq.heappush(pair_heap, (-pair2freq[pair], BytePair(*pair)))
                    else:
                        pair2freq.pop(pair, None)
                    if not pair2tokenized_word_indices[pair]:
                        pair2tokenized_word_indices.pop(pair, None)

                # 添加所有新 pair
                for pair in pairwise(new_tokens):
                    pair2freq[pair] += word_freq
                    heapq.heappush(pair_heap, (-pair2freq[pair], BytePair(*pair)))
                    pair2tokenized_word_indices[pair].add(idx)
            pbar.update(1)
    return vocab, merges
