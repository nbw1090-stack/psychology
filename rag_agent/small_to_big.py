"""
Sentence Window Retrieval (Small-to-Big) — LangChain 实现。

参考 LlamaIndex 的 SentenceWindowNodeParser + MetadataReplacementPostProcessor 模式：
- 每个句子作为一个独立节点嵌入向量库（精确语义匹配）
- 检索后，用该句子前后 N 句组成的"窗口"替换原文（充足生成上下文）

LangChain 组件使用：
- RecursiveCharacterTextSplitter：句子级切分（中文句号/问号/感叹号为分隔符）
- Document：句子节点和窗口的数据载体

数据流:
    PDF → 父chunk(500字) ──RecursiveCharacterTextSplitter──→ 句子节点(≤300字)
        每个节点: page_content=单句（嵌入用）
                   metadata["window"]=前后各window_size句（检索后替换用）
"""

from typing import List

from langchain_core.documents import Document

from . import logger


def _split_chinese_sentences(text: str) -> List[str]:
    """将中文文本按句末标点拆分为独立句子，保留标点在句末。

    与 RecursiveCharacterTextSplitter(chunk_size=1) 不同，此函数：
    - 保证每句以标点结尾（而非下一句以标点开头）
    - 过滤空句子和纯标点句子
    - 合并过短句子（< 5 字）到相邻句
    """
    raw: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？…!?\n":
            stripped = buf.strip()
            if stripped and not all(c in "。！？；：，、\n\r\t " for c in stripped):
                raw.append(stripped)
            buf = ""
    remaining = buf.strip()
    if remaining and not all(c in "。！？；：，、\n\r\t " for c in remaining):
        raw.append(remaining)

    # 合并过短句子
    merged: List[str] = []
    for s in raw:
        if merged and len(s) < 5:
            merged[-1] += s
        else:
            merged.append(s)
    return [s for s in merged if len(s) >= 3]


def _build_window(
    sentences: List[str],
    centre_idx: int,
    window_size: int = 2,
) -> str:
    """以 sentences[centre_idx] 为核心句，取前后各 window_size 句组成窗口。

    例：window_size=2 时，窗口为 前2句 + 核心句 + 后2句 = 最多5句。
    """
    start = max(0, centre_idx - window_size)
    end = min(len(sentences), centre_idx + window_size + 1)
    parts = [s for s in sentences[start:end] if s]
    # 用空串 join（切分时已保留标点），保证中文可读性
    return "".join(parts)


def create_sentence_nodes(
    parent_docs: List[Document],
    window_size: int = 2,
) -> List[Document]:
    """将父文档拆为句子节点。每个节点嵌入单句，附带前后 window_size 句的窗口。

    这是 LlamaIndex SentenceWindowNodeParser 的 LangChain 等价实现。

    Args:
        parent_docs: 父文档（当前为 500 字 chunk），需有 metadata["chunk_id"]
        window_size: 窗口半径。实际窗口 = 2*window_size + 1 句

    Returns:
        句子节点列表:
        - page_content:  单个句子（参与 embedding 和 BM25 检索）
        - metadata["window"]: 前后 window_size 句组成的完整窗口文本
        - metadata["parent_id"]: 父 chunk 的 chunk_id
    """
    nodes: List[Document] = []
    node_id = 0

    for parent in parent_docs:
        source = parent.metadata.get("source", "")
        page = parent.metadata.get("page")
        parent_id = parent.metadata.get("chunk_id")

        # 用中文句子拆分器切分父 chunk
        sentences = _split_chinese_sentences(parent.page_content)

        if not sentences:
            continue

        for i, sent in enumerate(sentences):
            window = _build_window(sentences, i, window_size=window_size)
            nodes.append(Document(
                page_content=sent,
                metadata={
                    "source": source,
                    "page": page,
                    "parent_id": parent_id,
                    "window": window,
                    "sentence_node_id": node_id,
                },
            ))
            node_id += 1

    return nodes


def replace_with_window(docs: List[Document]) -> List[Document]:
    """检索后将每个句子节点的 page_content 替换为其窗口文本，并按窗口去重。

    这是 LlamaIndex MetadataReplacementPostProcessor 的等价实现。

    对于非句子节点（如层级摘要，metadata 中无 "window"），保持原样。

    Args:
        docs: 检索返回的文档列表（句子节点 + 可能的摘要节点）

    Returns:
        替换后的文档列表（窗口文本），已去重
    """
    seen: set = set()
    result: List[Document] = []

    for doc in docs:
        window = doc.metadata.get("window", "")
        if not window:
            # 非句子节点（如层级摘要），直接保留
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)
            continue

        # 用窗口前 80 字去重
        key = window[:80]
        if key in seen:
            continue
        seen.add(key)

        result.append(Document(
            page_content=window,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page"),
                "parent_id": doc.metadata.get("parent_id"),
                "window_original": doc.page_content,  # 保留原句供调试
            },
        ))

    return result
