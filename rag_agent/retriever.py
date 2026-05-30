from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import logger


@dataclass
class RetrievalConfig:
    top_k: int = 30
    retrieval_mode: str = "hybrid"  # "vector" | "bm25" | "hybrid"
    enable_rerank: bool = True
    rerank_top_k: int = 8
    enable_query_rewrite: bool = True
    enable_hyde: bool = False          # HyDE: 生成假设文档替代/补充查询


DEFAULT_SEPARATORS = [
    "\n\n", "\n", "。", "！", "？", "；", "：", "，",
    ". ", "? ", "! ", " ", "",
]


# ------------------ BM25 关键词检索 ------------------

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_tf: np.ndarray | None = None
        self._avgdl: float = 0.0
        self._documents: List[Document] = []

    def _tokenize(self, text: str) -> List[str]:
        import jieba
        jieba.setLogLevel(jieba.logging.INFO + 1)  # suppress "Building prefix dict" spam
        tokens = list(jieba.cut(text))
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def index(self, documents: List[Document]) -> None:
        self._documents = documents
        tokenized = [" ".join(self._tokenize(d.page_content)) for d in documents]
        self._vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self._doc_tf = self._vectorizer.fit_transform(tokenized).toarray()
        dl = self._doc_tf.sum(axis=1)
        self._avgdl = float(np.mean(dl)) if len(dl) > 0 else 1.0

    def _score(self, query: str) -> List[Tuple[int, float]]:
        if self._vectorizer is None or self._doc_tf is None:
            return []
        query_tokens = self._tokenize(query)
        vocab = self._vectorizer.vocabulary_
        idf = dict(zip(vocab.keys(), self._vectorizer.idf_))

        scores = np.zeros(len(self._documents))
        for token in query_tokens:
            if token not in vocab:
                continue
            col = vocab[token]
            tf_col = self._doc_tf[:, col]
            dl = self._doc_tf.sum(axis=1)
            idf_val = idf[token]

            numerator = tf_col * (self.k1 + 1)
            denominator = tf_col + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            scores += idf_val * numerator / np.where(denominator > 0, denominator, 1e-8)

        scored = [(idx, float(score)) for idx, score in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def search(self, query: str, top_k: int = 8) -> List[Tuple[Document, float]]:
        scored = self._score(query)
        results = []
        for idx, score in scored[:top_k]:
            if score <= 0:
                continue
            results.append((self._documents[idx], 1.0 / (1.0 + score)))
        return results


# ------------------ 向量检索 ------------------

def _vector_search(
    query: str, vectorstore: VectorStore, top_k: int
) -> List[Tuple[Document, float]]:
    logger.sub("向量检索 (Dense / Semantic)")
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    logger.info(f"  向量检索命中 {len(results)} 条")
    if logger.is_verbose():
        for i, (doc, score) in enumerate(results):
            preview = doc.page_content[:100].replace("\n", " ")
            logger.info(f"    [{i + 1}] score={score:.4f}  {preview}...")
    return results


def _bm25_search(
    query: str, bm25: BM25Retriever, top_k: int
) -> List[Tuple[Document, float]]:
    logger.sub("BM25 检索 (Sparse / Keyword)")
    results = bm25.search(query, top_k=top_k)
    logger.info(f"  BM25 命中 {len(results)} 条")
    if logger.is_verbose():
        for i, (doc, score) in enumerate(results):
            preview = doc.page_content[:100].replace("\n", " ")
            logger.info(f"    [{i + 1}] score={score:.4f}  {preview}...")
    return results


# ------------------ RRF 混合融合 ------------------

def _rrf_fusion(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    logger.sub("RRF 混合融合")
    id_to_doc: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vector_results):
        cid = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        if cid not in id_to_doc:
            id_to_doc[cid] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        cid = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        if cid not in id_to_doc:
            id_to_doc[cid] = doc

    scores: dict[str, float] = {}

    for rank, (doc, _) in enumerate(vector_results):
        cid = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        cid = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    merged = [(id_to_doc[cid], sc) for cid, sc in scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"  向量 {len(vector_results)} + BM25 {len(bm25_results)} → 融合 {len(merged)} 条")
    if logger.is_verbose():
        for i, (doc, score) in enumerate(merged):
            preview = doc.page_content[:80].replace("\n", " ")
            logger.info(f"    [{i + 1}] rrf_score={score:.4f}  {preview}...")

    return merged


# ------------------ LLM Rerank ------------------

RERANK_SYSTEM = """你是一个文档相关性评估专家。请评估以下文档与用户问题的相关性。

对每个文档，给出 0-10 的评分：
- 10: 完美匹配，直接回答用户问题
- 7-9: 高度相关，包含回答问题所需的大部分信息
- 4-6: 部分相关，包含一些有用信息
- 1-3: 略有关联，但不是直接相关
- 0: 完全不相关

严格按照以下JSON格式输出，不要输出任何其他内容：
{"scores": [分数1, 分数2, ...]}
"""


def _llm_rerank(
    llm: BaseChatModel,
    query: str,
    documents: List[Tuple[Document, float]],
    top_k: int = 4,
) -> List[Document]:
    logger.sub("LLM Rerank (精排)")
    if not documents:
        return []

    docs_only = [d for d, _ in documents]
    if len(docs_only) <= top_k:
        logger.info(f"  候选数 {len(docs_only)} <= top_k={top_k}, 跳过多余精排")
        return docs_only[:top_k]

    doc_texts = []
    for i, (doc, _) in enumerate(documents):
        text = doc.page_content[:800].replace("\n", " ")
        doc_texts.append(f"[文档{i + 1}]: {text}")

    prompt = "请评估以下文档与用户问题的相关性：\n\n"
    prompt += f"用户问题: {query}\n\n"
    prompt += "\n\n".join(doc_texts)
    prompt += "\n\n请给出 JSON 评分："

    messages = [
        SystemMessage(content=RERANK_SYSTEM),
        HumanMessage(content=prompt),
    ]

    logger.info(f"  发送 {len(docs_only)} 条候选文档给 LLM 评分 ...")
    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.split("```")[0].strip()

        import json
        scores_data = json.loads(content)
        scores = scores_data.get("scores", [])

        if len(scores) != len(docs_only):
            logger.info(f"  评分数量不匹配 ({len(scores)} vs {len(docs_only)})，使用原始排序")
            return docs_only[:top_k]

        scored = list(zip(docs_only, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in scored[:top_k]]

        logger.info(f"  Rerank 完成: {len(documents)} → {len(result)} 条")
        if logger.is_verbose():
            for i, (doc, sc) in enumerate(scored[:top_k]):
                preview = doc.page_content[:80].replace("\n", " ")
                logger.info(f"    [{i + 1}] rerank_score={sc}  {preview}...")

        return result
    except Exception as e:
        logger.info(f"  Rerank 失败 ({e})，使用原始排序")
        return docs_only[:top_k]


# ------------------ 查询改写 ------------------

QUERY_REWRITE_PROMPT = """你是一个查询优化专家。用户的原始问题可能不够精确，请将其改写为一个或多个更有利于检索的查询。

要求：
- 保持原问题的核心意图
- 拆解复杂问题为多个简单查询
- 补充可能的同义词和相关术语
- 每行一个查询，用换行分隔
- 只输出查询文本，不要加任何解释或编号

原始问题: {query}

改写后的查询："""


HYDE_PROMPT = """你是一个心理学领域的知识专家。请根据用户的问题，写一段假设的参考答案。

要求：
- 尽量全面、专业地回答问题
- 使用学术化的中文表述
- 控制在200-400字
- 只输出答案文本，不要加解释

用户问题: {query}

假设答案："""


def hyde_generate(llm: BaseChatModel, query: str) -> str:
    """HyDE (Hypothetical Document Embeddings): 生成假设文档，用于替代/补充原始查询做检索。

    参考论文: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (arXiv 2212.10496)
    """
    logger.sub("HyDE 假设文档生成")
    messages = [
        HumanMessage(content=HYDE_PROMPT.format(query=query)),
    ]
    try:
        response = llm.invoke(messages)
        hypothesis = response.content.strip()
        logger.info(f"  原始查询: \"{query}\"")
        logger.info(f"  假设文档 ({len(hypothesis)} 字): \"{hypothesis[:120]}...\"")
        return hypothesis
    except Exception as e:
        logger.info(f"  HyDE 生成失败 ({e})，使用原始查询")
        return query


def rewrite_query(llm: BaseChatModel, query: str) -> List[str]:
    logger.sub("查询改写 (Query Rewrite)")
    messages = [
        HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=query)),
    ]
    try:
        response = llm.invoke(messages)
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        if not lines:
            return [query]
        rewritten = lines
        logger.info(f"  原始: \"{query}\"")
        for i, rq in enumerate(rewritten):
            logger.info(f"  改写[{i + 1}]: \"{rq}\"")
        return rewritten
    except Exception as e:
        logger.info(f"  查询改写失败 ({e})，使用原始查询")
        return [query]


# ------------------ 统一检索入口 ------------------

def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    result = []
    for doc in docs:
        sig = doc.metadata.get("chunk_id", doc.page_content[:100])
        if sig not in seen:
            seen.add(sig)
            result.append(doc)
    return result


def retrieve(
    query: str,
    vectorstore: VectorStore,
    llm: BaseChatModel | None = None,
    bm25: BM25Retriever | None = None,
    config: RetrievalConfig | None = None,
) -> List[Document]:
    cfg = config or RetrievalConfig()
    logger.section("检索阶段 (Retrieval)")
    logger.keyval("查询问题", query)
    logger.keyval("检索模式", cfg.retrieval_mode)
    logger.keyval("top_k", str(cfg.top_k))
    logger.keyval("Rerank", f"{cfg.enable_rerank} (top_k={cfg.rerank_top_k})")
    logger.keyval("HyDE", str(cfg.enable_hyde))

    # 1. HyDE 生成假设文档（优先于普通改写）
    if cfg.enable_hyde and llm:
        hyde_query = hyde_generate(llm, query)
        queries = [query, hyde_query]  # 原始查询 + 假设文档都检索
    elif cfg.enable_query_rewrite and llm:
        queries = rewrite_query(llm, query)
    else:
        queries = [query]

    # 2. 多查询检索
    all_candidates: List[Tuple[Document, float]] = []
    for q in queries:
        if cfg.retrieval_mode == "vector":
            candidates = list(_vector_search(q, vectorstore, cfg.top_k))
        elif cfg.retrieval_mode == "bm25":
            if bm25 is None:
                raise ValueError("bm25 retriever is required for bm25/hybrid mode")
            candidates = list(_bm25_search(q, bm25, cfg.top_k))
        elif cfg.retrieval_mode == "hybrid":
            if bm25 is None:
                raise ValueError("bm25 retriever is required for hybrid mode")
            vec_results = _vector_search(q, vectorstore, cfg.top_k)
            bm25_results = _bm25_search(q, bm25, cfg.top_k)
            candidates = _rrf_fusion(vec_results, bm25_results)
        else:
            raise ValueError(f"Unknown retrieval mode: {cfg.retrieval_mode}")
        all_candidates.extend(candidates)

    # 3. 对多个查询的结果做融合（简单模式：合并并按 score 降序）
    all_candidates.sort(key=lambda x: x[1])
    seen_ids = set()
    deduped: List[Tuple[Document, float]] = []
    for doc, score in all_candidates:
        cid = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped.append((doc, score))

    logger.info(f"  候选总计: {len(deduped)} 条（去重后）")

    # 4. Rerank
    if cfg.enable_rerank and llm and deduped:
        docs = _llm_rerank(llm, query, deduped, cfg.rerank_top_k)
    else:
        docs = _deduplicate_docs([d for d, _ in deduped[:cfg.rerank_top_k]])

    logger.info(f"  最终返回: {len(docs)} 条文档")
    if logger.is_verbose():
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:100].replace("\n", " ")
            logger.info(f"    [{i + 1}] {src}  {preview}...")

    return docs


def expand_context(
    docs: List[Document],
    vectorstore: Chroma,
    window: int = 1,
) -> List[Document]:
    """对检索结果扩展前后 N 个段落。

    通过 chunk_id 查找 prev_chunk_id / next_chunk_id，
    再从向量库按 metadata 过滤取出相邻段落。
    """
    if window <= 0 or not docs:
        return docs

    logger.sub("上下文扩展 (Context Expansion)")
    logger.info(f"  扩展窗口: 前后各 {window} 段")

    neighbor_ids: set[str] = set()
    for doc in docs:
        prev_id = doc.metadata.get("prev_chunk_id", "")
        next_id = doc.metadata.get("next_chunk_id", "")
        if prev_id:
            neighbor_ids.add(prev_id)
        if next_id:
            neighbor_ids.add(next_id)

    # 排除已在结果中的 chunk
    existing_ids = {doc.metadata.get("chunk_id", "") for doc in docs}
    fetch_ids = neighbor_ids - existing_ids

    if not fetch_ids:
        logger.info(f"  无需扩展（已包含相邻段落）")
        return docs

    # 从 ChromaDB 按 chunk_id 过滤查询
    expanded: List[Document] = []
    try:
        collection = vectorstore._collection
        for cid in fetch_ids:
            results = collection.get(
                where={"chunk_id": cid},
                include=["documents", "metadatas"],
            )
            if results and results.get("documents"):
                meta = dict(results["metadatas"][0]) if results["metadatas"] else {}
                meta["is_expanded"] = True
                expanded.append(Document(
                    page_content=results["documents"][0],
                    metadata=meta,
                ))
    except Exception as e:
        logger.info(f"  上下文扩展查询失败 ({e})，跳过扩展")
        return docs

    # 合并并排序
    merged = list(docs) + expanded

    # 按 (book_title, chapter_index, paragraph_index) 排序
    def _sort_key(doc: Document) -> Tuple:
        m = doc.metadata
        return (
            m.get("book_title", ""),
            m.get("chapter_index", 0),
            m.get("paragraph_index", 0),
        )
    merged.sort(key=_sort_key)

    # 去重
    seen: set[str] = set()
    deduped: List[Document] = []
    for doc in merged:
        cid = doc.metadata.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(doc)

    logger.info(f"  原始 {len(docs)} → 扩展后 {len(deduped)} 条 "
                f"(新增 {len(deduped) - len(docs)} 段相邻段落)")
    return deduped


def format_context(docs: List[Document]) -> str:
    # 分离核心检索结果和上下文扩展结果
    primary = [d for d in docs if not d.metadata.get("is_expanded")]
    supplementary = [d for d in docs if d.metadata.get("is_expanded")]

    def _source_label(doc: Document) -> str:
        book = doc.metadata.get("book_title", "")
        chapter = doc.metadata.get("chapter_title", "")
        para_idx = doc.metadata.get("paragraph_index")
        if book:
            source_info = book
        else:
            source_info = doc.metadata.get("source", "unknown")
        if chapter:
            source_info += f" > {chapter}"
        if para_idx is not None:
            source_info += f" (第{para_idx + 1}段)"
        return source_info

    parts = []
    for i, doc in enumerate(primary, 1):
        label = _source_label(doc)
        parts.append(f"[文档 {i}] (来源: {label})\n{doc.page_content}")

    if supplementary:
        parts.append("【以下为补充上下文，仅供参考】")
        for i, doc in enumerate(supplementary, 1):
            label = _source_label(doc)
            parts.append(f"[补充 {i}] (来源: {label})\n{doc.page_content}")

    context = "\n\n---\n\n".join(parts)

    logger.sub("上下文组装 (Context)")
    logger.info(f"  核心 {len(primary)} + 补充 {len(supplementary)} = {len(docs)} 个文档块, "
                f"总长度 {len(context)} 字符")
    if logger.is_verbose():
        for i, part in enumerate(parts):
            logger.content_block(f"文档块 {i + 1}", part, max_len=300)

    return context
