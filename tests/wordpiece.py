# wordpiece_simple.py
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

# -------------------------
# 文本预处理 (简单)
# -------------------------
def normalize_text(text: str) -> str:
    # 小写 + 简单的非字母数字分隔（可根据需求修改）
    text = text.lower()
    # 保持汉字/英文字母/数字/空格 和 基本标点
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff\s\.\,\!\?\-]", "", text)
    return text

def word_tokenize(text: str) -> List[str]:
    # 简单按空格分词（保留中文字符块）
    return [w for w in re.split(r"\s+", text) if w]

# -------------------------
# 构建初始符号表：每个单词展开为字符序列 + </w> 作为单词边界
# 这与 BPE 常见做法类似；WordPiece 用子词，常用 "##" 表示非首子词
# -------------------------
def get_initial_vocab(corpus_words: List[str]) -> Dict[Tuple[str,...], int]:
    """返回字典： token sequence tuple -> 频率
    每个单词表示为字符序列，尾部添加 </w> 标记作为单词边界
    """
    vocab = Counter()
    for w in corpus_words:
        # 将单词按字符分割，中文按字符自然分割
        chars = list(w)
        # 添加结束符，避免跨单词合并
        token = tuple(chars + ["</w>"])
        vocab[token] += 1
    return dict(vocab)

# -------------------------
# 统计所有相邻对的频率
# -------------------------
def get_pair_stats(vocab: Dict[Tuple[str,...], int]) -> Counter:
    pairs = Counter()
    for token_seq, freq in vocab.items():
        for i in range(len(token_seq) - 1):
            pairs[(token_seq[i], token_seq[i+1])] += freq
    return pairs

# -------------------------
# 将最频繁的一对合并
# -------------------------
def merge_pair(vocab: Dict[Tuple[str,...], int], pair: Tuple[str,str]) -> Dict[Tuple[str,...], int]:
    bigram = pair
    merged_vocab = {}
    pattern = tuple(bigram)
    for token_seq, freq in vocab.items():
        i = 0
        new_seq = []
        while i < len(token_seq):
            # 如果匹配到pair则合并
            if i < len(token_seq) - 1 and token_seq[i] == pattern[0] and token_seq[i+1] == pattern[1]:
                new_seq.append(token_seq[i] + token_seq[i+1])
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        merged_vocab[tuple(new_seq)] = merged_vocab.get(tuple(new_seq), 0) + freq
    return merged_vocab

# -------------------------
# 训练函数（BPE-style，生成子词表）
# -------------------------
def train_wordpiece_bpe(
    corpus: List[str],
    vocab_size: int,
    min_freq: int = 2,
    verbose: bool = True
) -> List[str]:
    """
    corpus: list of raw text strings
    vocab_size: 期望输出的子词数量（含基础字符和特殊符号）
    返回：词表（list），其中包含 '##' 前缀用于非首子词
    """
    # 1. 预处理 & 分词成单词
    words = []
    for line in corpus:
        line = normalize_text(line)
        words.extend(word_tokenize(line))

    # 2. 初始字典（字符 + </w>）
    word_counts = Counter(words)
    # Expand into token sequences (char-level with end token) with frequency
    corpus_words = []
    for w, cnt in word_counts.items():
        for _ in range(cnt):
            corpus_words.append(w)
    vocab = get_initial_vocab(corpus_words)

    # 3. 迭代合并直到达到 vocab_size
    # 初始化词表：所有单字符（不含 </w>）作为可用子词
    # 最后我们会把合并得到的符号转换成 WordPiece 风格（首子词无前缀，后续子词加 "##"）
    merges = []
    while True:
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        most_common_pair, freq = pairs.most_common(1)[0]
        if freq < min_freq:
            break
        # 合并该对
        vocab = merge_pair(vocab, most_common_pair)
        merges.append(most_common_pair)
        # 估计当前字表大小：基础字符 + 合并产生的新token
        if verbose and len(merges) % 100 == 0:
            print(f"merges: {len(merges)}, top pair: {most_common_pair}, freq={freq}")
        # 终止条件（粗略）：当合并数量 + 基础字符达到 vocab_size
        # 基础字符集：
        base_chars = set()
        for seq in vocab.keys():
            for s in seq:
                base_chars.add(s)
        est_vocab_size = len(base_chars)
        if est_vocab_size >= vocab_size:
            break

    # 构建最终词表：从vocab的token序列中提取所有独特符号
    tokens = set()
    for seq in vocab.keys():
        for s in seq:
            tokens.add(s)
    # 移除结束符 </w>，并构造 wordpiece 风格：带 "##" 的续接子词
    tokens.discard("</w>")
    # 分类哪些是以原始字符起始（首子词）或非首（含合并后包含多个字符）
    # 简单策略：如果 token 长度>1，则把它标记为可能的 subword；在分词时我们用 "##" 规则
    token_list = sorted(tokens, key=lambda x: (-len(x), x))  # 长的优先
    # 附加特殊token
    final_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + token_list
    # 裁剪到指定大小
    final_vocab = final_vocab[:vocab_size]
    return final_vocab

# -------------------------
# 基于训练好的词表做 Greedy WordPiece 分词
# -------------------------
def build_lookup_vocab(vocab_list: List[str]) -> Dict[str,int]:
    return {tok: i for i, tok in enumerate(vocab_list)}

def wordpiece_tokenize(word: str, vocab_lookup: Dict[str,int], unk_token="[UNK]") -> List[str]:
    """
    Greedy longest-match algorithm for WordPiece.
    对一个单词进行分词，返回子词列表（续接子词以 '##' 前缀表示）
    """
    if not word:
        return []
    sub_tokens = []
    start = 0
    end = len(word)
    while start < len(word):
        cur_end = end
        matched = None
        while cur_end > start:
            piece = word[start:cur_end]
            # 如果不是在首位置，使用 '##' 前缀查找
            if start > 0:
                piece_lookup = "##" + piece
            else:
                piece_lookup = piece
            if piece_lookup in vocab_lookup:
                matched = piece_lookup
                break
            cur_end -= 1
        if matched is None:
            # 无任何匹配，返回UNK
            return [unk_token]
        sub_tokens.append(matched)
        start = cur_end
    return sub_tokens

def tokenize_text(text: str, vocab_lookup: Dict[str,int], unk_token="[UNK]") -> List[str]:
    text = normalize_text(text)
    words = word_tokenize(text)
    tokens = []
    for w in words:
        toks = wordpiece_tokenize(w, vocab_lookup, unk_token=unk_token)
        tokens.extend(toks)
    return tokens

# -------------------------
# 简单示例：训练 + 分词
# -------------------------
if __name__ == "__main__":
    # 简单语料示例（中文/英文混合亦可）
    corpus = [
        "This is a simple example.",
        "We demonstrate a tiny WordPiece-like trainer.",
        "Tokenization of words: playing, played, plays, play.",
        "舌诊 数据 示例 测试。",
        "Another example with playing played play."
    ]

    print("Training small vocab...")
    vocab = train_wordpiece_bpe(corpus, vocab_size=200, verbose=True)
    print("Vocab size:", len(vocab))
    print("Some sample tokens:", vocab[:80])

    lookup = build_lookup_vocab(vocab)
    test_sentences = [
        "We are playing today.",
        "I played yesterday.",
        "测试 舌诊 示例"
    ]
    for s in test_sentences:
        toks = tokenize_text(s, lookup)
        print(s, "->", toks)
