"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
import logging
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
# 配置参数
local_dir = "edu_fineweb10B"  # 本地保存数据的目录
remote_name = "sample-10BT"  # Hugging Face 数据集的分支名称
shard_size = int(1e8)  # 每个分片的 token 数量（100M）
val_size = int(1e7)  # 验证集的 token 数量（10M）

# 创建缓存目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)  # 如果目录不存在则创建

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting tokenization and sharding process...")

# 下载数据集
logging.info("Loading dataset...")
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 初始化 tokenizer
enc = tiktoken.get_encoding("gpt2")  # 加载 GPT-2 的 tokenizer
eot = enc._special_tokens['<|endoftext|>']  # 获取 <|endoftext|> 特殊 token，用于分隔文档

def tokenize(doc):
    """
    将单个文档 tokenize 并返回 uint16 数组。
    :param doc: 包含文本的字典。
    :return: numpy 数组，包含 tokenized 数据。
    """
    try:
        tokens = [eot]  # 添加 <|endoftext|> 作为文档分隔符
        tokens.extend(enc.encode_ordinary(doc["text"]))  # 将文本 tokenize
        tokens_np = np.array(tokens)  # 转换为 numpy 数组
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        return tokens_np.astype(np.uint16)  # 转换为 uint16 以节省空间
    except Exception as e:
        logging.error(f"Error tokenizing document: {e}")
        return np.array([eot], dtype=np.uint16)  # 返回一个空文档

def write_datafile(filename, tokens_np):
    """
    将 tokenized 数据保存为 .npy 文件。
    :param filename: 文件名。
    :param tokens_np: 包含 tokenized 数据的 numpy 数组。
    """
    try:
        np.save(filename, tokens_np)  # 保存为 .npy 文件
    except Exception as e:
        logging.error(f"Error writing file {filename}: {e}")

# 设置多进程数量（CPU 核心数的一半）
nprocs = max(1, os.cpu_count() // 2)

# 使用多进程池处理数据
with mp.Pool(nprocs) as pool:
    shard_index = 0  # 当前分片的索引
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 预分配数组，用于存储当前分片的 token
    token_count = 0  # 当前分片的 token 数量
    progress_bar = None  # 进度条

    # 多进程 tokenize 数据集
    for tokens in pool.imap(tokenize, fw, chunksize=64):  # 调整 chunksize 以提高性能
        # 如果当前分片有足够空间，将 token 添加到分片中
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # 初始化或更新进度条
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # 如果当前分片已满，写入文件并开始新的分片
            if shard_index == 0 and token_count > val_size:
                # 第一个分片的前 10M token 作为验证集
                val_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_val_{shard_index:06d}")
                write_datafile(val_filename, all_tokens_np[:val_size])
                # 剩余部分作为第一个训练分片
                train_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}")
                write_datafile(train_filename, all_tokens_np[val_size:token_count])
            else:
                # 后续分片作为训练集
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                write_datafile(filename, all_tokens_np[:token_count])
            shard_index += 1
            progress_bar = None
            # 将当前文档的剩余 token 写入下一个分片
            remainder = shard_size - token_count
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 重新分配内存

    # 写入最后一个分片（如果有剩余 token）
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

# 记录日志，表示任务完成
logging.info("Tokenization and sharding completed.")