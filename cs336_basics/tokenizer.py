import base64
import heapq
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Self

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


def _initialize_stats(word_counter: Counter) -> tuple[defaultdict, defaultdict, list, defaultdict, list]:
    """从词频计数器初始化所有核心数据结构。"""
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

    return pair2freq, pair2tokenized_word_indices, tokenized_word_list, word2freq, pair_heap


def _update_stats_after_merge(
    idx: int,
    p0: bytes,
    p1: bytes,
    merged_token: bytes,
    pair2freq: defaultdict,
    pair2tokenized_word_indices: defaultdict,
    tokenized_word_list: list,
    word2freq: defaultdict,
    pair_heap: list,
):
    """在一个词中合并一个 pair 后，更新所有相关的数据结构。"""
    old_tokens = tokenized_word_list[idx]
    word_bytes = b"".join(old_tokens)  # 重建原始 word bytes 作为 key
    word_freq = word2freq[word_bytes]

    # 为这个词重建 token 列表
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

    # 减去所有旧 pair 的频率
    for pair in pairwise(old_tokens):
        pair2freq[pair] -= word_freq
        pair2tokenized_word_indices[pair].discard(idx)
        if pair2freq[pair] > 0:
            heapq.heappush(pair_heap, (-pair2freq[pair], BytePair(*pair)))
        else:
            pair2freq.pop(pair, None)
        if not pair2tokenized_word_indices[pair]:
            pair2tokenized_word_indices.pop(pair, None)

    # 添加所有新 pair 的频率
    for pair in pairwise(new_tokens):
        pair2freq[pair] += word_freq
        heapq.heappush(pair_heap, (-pair2freq[pair], BytePair(*pair)))
        pair2tokenized_word_indices[pair].add(idx)


def train_bbpe(
    corpus_path: str, vocab_size: int, special_tokens: list[str], split_special_token: bytes, num_readers: int = 0
) -> tuple[Vocab, Merges]:
    if vocab_size <= 256:
        raise ValueError("Vocab size must be larger than 256")

    if num_readers <= 0:
        num_readers = os.cpu_count() or 1

    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counter = count_words_parallel(corpus_path, split_special_token, special_tokens, pretokenization_pattern, num_readers)

    # 初始化核心数据结构
    pair2freq, pair2tokenized_word_indices, tokenized_word_list, word2freq, pair_heap = _initialize_stats(word_counter)

    # 初始化词汇表
    vocab: Vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges: Merges = []
    total_merges = vocab_size - len(vocab)

    # 主合并循环
    with tqdm(total=total_merges, desc="Building vocabulary") as pbar:
        for _ in range(total_merges):
            # 寻找最佳 pair
            best_pair = None
            while pair_heap:
                neg_freq, current_pair = heapq.heappop(pair_heap)
                p0, p1 = current_pair.byte1, current_pair.byte2
                # 检查堆中的频率是否过时
                if -neg_freq == pair2freq.get((p0, p1), 0) and -neg_freq > 0:
                    best_pair = current_pair
                    break

            if best_pair is None:
                print("数据训练完毕，提前退出")
                break

            p0, p1 = best_pair.byte1, best_pair.byte2
            merged_token = p0 + p1

            # 更新 vocab 和 merges
            vocab[next_id] = merged_token
            merges.append((p0, p1))
            next_id += 1

            # 更新核心数据结构
            affected_indices = pair2tokenized_word_indices.get((p0, p1), set())
            for idx in list(affected_indices):  # 使用 list 避免在迭代时修改集合
                _update_stats_after_merge(
                    idx, p0, p1, merged_token, pair2freq, pair2tokenized_word_indices, tokenized_word_list, word2freq, pair_heap
                )
            pbar.update(1)

    return vocab, merges


class BBPETokenizer:
    def __init__(self, vocab: Vocab, merges: Merges, special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.id_to_bytes = vocab
        self.bytes_to_id = {b: i for i, b in self.id_to_bytes.items()}
        self.merges = merges
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = set()
        self.special_split_pattern = None

        if special_tokens:
            self.add_special_tokens(special_tokens)
            self.special_tokens = sorted(special_tokens, reverse=True)
            self.special_split_pattern = re.compile(f"({'|'.join([re.escape(token) for token in self.special_tokens])})")
            self.special_tokens = set(self.special_tokens)

        self.pre_tokenize_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        with open(vocab_filepath, "r", encoding="utf-8") as fp:
            vocab_serializable = json.load(fp)
            vocab = {int(k): base64.b64decode(v) for k, v in vocab_serializable.items()}
        with open(merges_filepath, "r", encoding="utf-8") as fp:
            merges_serializable = json.load(fp)
            merges = [(base64.b64decode(a), base64.b64decode(b)) for a, b in merges_serializable]
        return cls(vocab, merges, special_tokens)

    def add_special_tokens(self, special_tokens: list[str]) -> None:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.bytes_to_id:
                continue
            new_id = len(self.id_to_bytes)
            self.id_to_bytes[new_id] = token_bytes
            self.bytes_to_id[token_bytes] = new_id

    def _bbpe_merge(self, text: str) -> list[int]:
        if text in self.special_tokens:
            return [self.bytes_to_id[text.encode("utf-8")]]
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        while True:
            pair = min(pairwise(tokens), key=lambda x: self.merge_rank.get(x, float("inf")), default=None)
            if pair not in self.merge_rank:
                break
            merged_token = pair[0] + pair[1]
            size = i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[size] = merged_token
                    i += 2
                else:
                    tokens[size] = tokens[i]
                    i += 1
                size += 1
            del tokens[size:]
        return [self.bytes_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        segments = re.split(self.special_split_pattern, text) if self.special_split_pattern else [text]
        token_ids = []
        for segment in segments:
            if segment in self.special_tokens:
                token_ids.extend(self._bbpe_merge(segment))
            else:
                for match in re.finditer(self.pre_tokenize_pattern, segment):
                    token_ids.extend(self._bbpe_merge(match.group(0)))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[idx] for idx in ids)
        return text.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        save_path = Path(path)
        os.makedirs(save_path, exist_ok=True)
        vocab_serializable = {k: base64.b64encode(v).decode("ascii") for k, v in self.vocab.items()}
        with open(save_path / "vocab.json", "w", encoding="utf-8") as fp:
            json.dump(vocab_serializable, fp, ensure_ascii=False)
        merges_serializable = [(base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")) for a, b in self.merges]
        with open(save_path / "merges.json", "w", encoding="utf-8") as fp:
            json.dump(merges_serializable, fp, ensure_ascii=False)
        with open(save_path / "special_tokens.json", "w", encoding="utf-8") as fp:
            json.dump(list(self.special_tokens), fp, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> Self:
        load_path = Path(path)
        with open(load_path / "vocab.json", "r", encoding="utf-8") as fp:
            vocab_serializable = json.load(fp)
            vocab = {int(k): base64.b64decode(v) for k, v in vocab_serializable.items()}
        with open(load_path / "merges.json", "r", encoding="utf-8") as fp:
            merges_serializable = json.load(fp)
            merges = [(base64.b64decode(a), base64.b64decode(b)) for a, b in merges_serializable]
        with open(load_path / "special_tokens.json", "r", encoding="utf-8") as fp:
            special_tokens = json.load(fp)
        return cls(vocab, merges, special_tokens)
