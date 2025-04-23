import regex as re
from tqdm import tqdm

def get_stats(ids,stats=None):
    if not stats:
        counts={}
    else:
        counts=stats
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts

def merge(ids,pair,idx):
    new_ids=[]
    i=0
    while i<len(ids):
        if i<len(ids)-1 and (ids[i],ids[i+1]) == pair:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def get_top_pair(counts):
    return sorted(counts.items(),key=lambda x:x[1],reverse=True)[0][0]

class BasicTokenizer:

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in tqdm(range(num_merges)):
            stats = get_stats(ids)
            pair = get_top_pair(stats)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in tqdm(ids))
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            stats = get_stats(ids)
            valid_stats = {pair: count for pair, count in stats.items() if pair in self.merges}
            if not valid_stats:
                break  # 没有可合并的字节对，提前终止
            pair = min(valid_stats, key=self.merges.get) 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
class GPTtokenizer:
    def __init__(self,pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        self.compiled_pattern = re.compile(pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_chunks=re.findall(self.compiled_pattern,text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in tqdm(range(num_merges)):
            stats={}
            for chunk_ids in ids:
                stats = get_stats(chunk_ids,stats)
            pair = get_top_pair(stats)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab
    def decode(self,ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            valid_stats = {pair: count for pair, count in stats.items() if pair in self.merges}
            if not valid_stats:
                break
            pair = min(valid_stats, key=self.merges.get)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids