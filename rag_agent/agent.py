"""
LangGraph 多步 Agent — 将单趟 RAG pipeline 演进为有条件分支的 Agent 架构。

节点:
  query_router    — 启发式查询分类 (0 LLM)
  simple_retrieve — 包装现有 retrieve() (0 额外 LLM)
  macro_retrieve  — 层级摘要检索 (0 LLM)
  decompose       — 复杂问题拆解 (1 LLM)
  multi_retrieve  — 多子查询检索合并 (0 额外 LLM)
  generate        — 增强版生成 (1 LLM)
  claim_verify    — P0 事实校验 (1 LLM)
  confidence_score— P2 置信度评分 (0 LLM)
"""

import json
from typing import Literal, TypedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from . import logger
from .agent_config import AgentConfig
from .generator import generate
from .retriever import (
    BM25Retriever,
    RetrievalConfig,
    format_context,
    retrieve,
)


# ─── State ───────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    question: str
    query_type: Literal["simple", "complex", "macro"]
    intent: str                              # 用户意图简述
    search_queries: list[str]                # 优化后的检索关键词/查询
    documents: list[Document]
    retrieval_scores: dict[str, float]
    draft_answer: str
    verified_answer: str
    unverifiable_claims: list[str]
    confidence: float
    confidence_level: Literal["high", "medium", "low"]
    answer: str
    metadata: dict


# ─── Prompts ─────────────────────────────────────────────────────

QUERY_UNDERSTAND_PROMPT = """你是一个查询理解专家，精通心理学领域知识。请分析用户的提问，提取检索所需的结构化信息。

知识库包含以下书籍：
- 《焦虑心理学》：焦虑症的类型、诊断、认知行为疗法(CBT)、各流派焦虑理论
- 《乌合之众：大众心理研究》：群体心理学、从众、暗示、传染、群体特征
- 《人人都该懂的心理学》：心理学基础概念、学派、经典实验

分析任务：
1. 判断查询类型：
   - "simple": 单一事实/定义/概念，一次检索可覆盖
   - "complex": 对比/多跳/多概念，需要拆分为多个检索查询
   - "macro": 要求总结/概括整本书或大范围内容
2. 提取核心概念和关键词，生成最适合检索的查询语句
3. 简述用户意图

关键：
- 检索查询应使用文档中可能出现的学术术语（如用"广泛性焦虑障碍"而非"总是担心的病"）
- 复杂问题拆为 2-3 个独立可检索的子查询
- 简单问题输出 1-2 个优化后的查询即可
- 如果用户的问题不在上述书籍范围内，intent 标注为 "out_of_scope"

用户问题: {query}

请严格按以下JSON格式输出：
{{"query_type": "simple或complex或macro", "intent": "用户意图一句话描述", "search_queries": ["检索查询1", "检索查询2"]}}"""


CLAIM_VERIFY_SYSTEM = """你是一个事实核查专家。你的任务是检查回答中的每个声明是否能在提供的上下文中找到依据。

操作原则（非常重要）：
1. 保守判断：只要声明中的核心事实能在上下文中找到实质支持，即视为已验证。不要求措辞完全一致。
2. 逐条修正：只移除或标注那些确实无法在上下文中找到任何支持的声明，保留其余所有已验证的内容。
3. 禁止全盘否定：即使有少量声明不可验证，也不允许将整个回答替换为"据现有文档无法确认"。必须保留已验证的部分。

不要添加任何上下文中没有的新信息。"""

CLAIM_VERIFY_USER = """上下文:
{context}

回答:
{answer}

请逐条检查回答中的声明。只标记确实无法在上下文中找到任何依据的声明。
如果大部分声明可以验证，corrected_answer 应保留已验证内容，仅移除不可验证的部分。

严格按以下JSON格式输出：
{{"verified_claims": ["已验证的声明"], "unverifiable_claims": ["不可验证的声明"], "corrected_answer": "保留已验证内容，仅移除不可验证部分的修正回答"}}"""


# ─── 辅助函数 ────────────────────────────────────────────────────

def _compute_retrieval_scores(
    query: str,
    docs: list[Document],
    embeddings,
) -> dict[str, float]:
    """用 embedding cosine similarity 计算每篇文档与查询的相关度。"""
    if not docs:
        return {}

    query_emb = embeddings.embed_query(query)
    doc_texts = [d.page_content for d in docs]
    doc_embs = embeddings.embed_documents(doc_texts)

    scores = {}
    for i, doc in enumerate(docs):
        cid = doc.metadata.get("chunk_id", str(i))
        q = np.array(query_emb)
        d = np.array(doc_embs[i])
        norm = np.linalg.norm(q) * np.linalg.norm(d) + 1e-8
        scores[cid] = float(np.dot(q, d) / norm)
    return scores


def _macro_retrieve(
    question: str,
    vectorstore: VectorStore,
) -> list[Document]:
    """宏观检索: 摘要优先 + 段落补充 + 融合去重。

    从 pipeline.py 的宏观检索路径提取。
    """
    try:
        summary_docs = vectorstore.similarity_search(
            question, k=6,
            filter={"source": "hierarchy"},
        )
        logger.info(f"[Agent/macro]   摘要检索命中 {len(summary_docs)} 条")
    except Exception:
        all_vs_docs = vectorstore.similarity_search(question, k=12)
        summary_docs = [d for d in all_vs_docs
                        if d.metadata.get("source") == "hierarchy"][:6]
        logger.info(f"[Agent/macro]   摘要检索（无过滤回退）命中 {len(summary_docs)} 条")

    detail_docs = vectorstore.similarity_search(question, k=6)

    seen: set[str] = set()
    docs: list[Document] = []
    for d in summary_docs + detail_docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            docs.append(d)
    return docs[:12]


def _understand_query(
    llm: BaseChatModel,
    query: str,
) -> dict:
    """Query Understanding: 一次 LLM 调用做意图识别 + 关键词抽取。"""
    messages = [
        HumanMessage(content=QUERY_UNDERSTAND_PROMPT.format(query=query)),
    ]
    fallback = {
        "query_type": "simple",
        "intent": "",
        "search_queries": [query],
    }

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # 容错: 提取 ```json ... ```
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        result = json.loads(content)

        # 校验必须字段
        qt = result.get("query_type", "simple")
        if qt not in ("simple", "complex", "macro"):
            qt = "simple"

        queries = result.get("search_queries", [])
        if not queries or not isinstance(queries, list):
            queries = [query]

        return {
            "query_type": qt,
            "intent": result.get("intent", ""),
            "search_queries": queries[:3],
        }
    except Exception as e:
        logger.info(f"[Agent/understand] 解析失败 ({e})，使用原始查询")
        return fallback


def _verify_claims(
    llm: BaseChatModel,
    answer: str,
    context: str,
) -> dict:
    """P0 核心: 校验回答中的声明是否有据。"""
    messages = [
        SystemMessage(content=CLAIM_VERIFY_SYSTEM),
        HumanMessage(content=CLAIM_VERIFY_USER.format(
            context=context[:4000],
            answer=answer[:2000],
        )),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # 容错: 提取 ```json ... ``` 中的内容
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        result = json.loads(content)

        verified = result.get("verified_claims", [])
        unverifiable = result.get("unverifiable_claims", [])
        corrected = result.get("corrected_answer", answer)

        # 回退: 修正后回答退化时使用原始回答
        if not corrected or not corrected.strip():
            corrected = answer
        elif len(corrected) < len(answer) * 0.3 and len(answer) > 50:
            # corrected 不到 draft 的 30%，说明被过度删减
            logger.info("[Agent/claim_verify]   修正回答过度删减，回退到原始回答")
            corrected = answer

        return {
            "verified_claims": verified,
            "unverifiable_claims": unverifiable,
            "corrected_answer": corrected,
        }
    except Exception as e:
        logger.info(f"[Agent/claim_verify] 校验失败 ({e})，使用原始回答")
        return {
            "verified_claims": [],
            "unverifiable_claims": [],
            "corrected_answer": answer,
        }


def _compute_confidence(state: AgentState, config: AgentConfig) -> dict:
    """P2 核心: 纯启发式置信度评分。"""
    docs = state.get("documents", [])
    scores = state.get("retrieval_scores", {})
    unverifiable = state.get("unverifiable_claims", [])

    if not docs:
        answer = state.get("verified_answer") or state.get("draft_answer", "")
        if not answer:
            answer = "文档中未找到相关信息。"
        return {
            "confidence": 0.0,
            "confidence_level": "low",
            "answer": answer,
            "metadata": {
                "query_type": state.get("query_type"),
                "sub_queries": state.get("sub_queries", []),
                "num_documents": 0,
                "top_score": 0.0,
                "unverifiable_claims": unverifiable,
                "confidence": 0.0,
            },
        }

    # 因子 1: Top-3 文档相似度均值 (归一化到 0-1)
    # embedding cosine similarity 在本项目中通常落在 0.35-0.60
    top_scores = sorted(scores.values(), reverse=True)[:3]
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    norm_similarity = max(0.0, min(1.0, (avg_top - 0.35) / 0.25))

    # 因子 2: 声明校验结果
    # 全部通过 → 大幅加分；有不可验证声明 → 惩罚
    if not unverifiable:
        claim_factor = 0.3
    else:
        claim_factor = -len(unverifiable) * 0.12

    # 因子 3: 文档数量加分 (越多越有信心)
    doc_bonus = min(len(docs) / 8.0, 1.0) * 0.1

    # 复合评分
    raw = norm_similarity * 0.5 + claim_factor + doc_bonus
    confidence = max(0.0, min(1.0, raw))

    if confidence >= config.confidence_high_threshold:
        level = "high"
    elif confidence >= config.confidence_low_threshold:
        level = "medium"
    else:
        level = "low"

    answer = state.get("verified_answer") or state.get("draft_answer", "")

    if level == "low":
        answer += "\n\n⚠️ 注意：以上回答的可信度较低，建议查阅原始文献确认。"
    elif level == "medium":
        answer += "\n\n💡 提示：以上回答的部分内容基于有限参考资料，建议进一步验证。"

    return {
        "confidence": confidence,
        "confidence_level": level,
        "answer": answer,
        "metadata": {
            "query_type": state.get("query_type"),
            "sub_queries": state.get("sub_queries", []),
            "num_documents": len(docs),
            "top_score": top_scores[0] if top_scores else 0.0,
            "unverifiable_claims": unverifiable,
            "confidence": confidence,
        },
    }


# ─── 图构建 ──────────────────────────────────────────────────────

def build_agent_graph(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    bm25: BM25Retriever | None,
    embeddings,
    rag_config,
    agent_config: AgentConfig,
):
    """构建 LangGraph Agent 图。"""

    # ── Node: query_understand ──
    # 1 次 LLM 调用：意图识别 + 关键词抽取 + 查询优化
    def query_understand(state: AgentState) -> dict:
        logger.sub("[Agent] 查询理解 (Query Understanding)")
        result = _understand_query(llm, state["question"])
        logger.info(f"  类型: {result['query_type']}")
        logger.info(f"  意图: {result['intent']}")
        for i, sq in enumerate(result["search_queries"]):
            logger.info(f"  检索词[{i + 1}]: {sq}")
        return {
            "query_type": result["query_type"],
            "intent": result["intent"],
            "search_queries": result["search_queries"],
        }

    # ── Node: simple_retrieve ──
    # 用 understanding 产出的检索词，不走 HyDE/rewrite
    def simple_retrieve(state: AgentState) -> dict:
        retrieval_cfg = RetrievalConfig(
            top_k=rag_config.top_k,
            retrieval_mode=rag_config.retrieval_mode,
            enable_rerank=rag_config.enable_rerank,
            rerank_top_k=rag_config.rerank_top_k,
            enable_query_rewrite=False,
            enable_hyde=False,
        )
        queries = state.get("search_queries", [state["question"]])
        primary_query = queries[0]

        docs = retrieve(
            query=primary_query,
            vectorstore=vectorstore,
            llm=llm,
            bm25=bm25,
            config=retrieval_cfg,
        )
        scores = _compute_retrieval_scores(primary_query, docs, embeddings)
        return {"documents": docs, "retrieval_scores": scores}

    # ── Node: macro_retrieve ──
    def macro_retrieve(state: AgentState) -> dict:
        queries = state.get("search_queries", [state["question"]])
        docs = _macro_retrieve(queries[0], vectorstore)
        scores = _compute_retrieval_scores(queries[0], docs, embeddings)
        return {"documents": docs, "retrieval_scores": scores}

    # ── Node: multi_retrieve ──
    # 用 understanding 产出的多个检索词，每个独立检索后合并
    def multi_retrieve(state: AgentState) -> dict:
        retrieval_cfg = RetrievalConfig(
            top_k=rag_config.top_k,
            retrieval_mode=rag_config.retrieval_mode,
            enable_rerank=rag_config.enable_rerank,
            rerank_top_k=rag_config.rerank_top_k,
            enable_query_rewrite=False,
            enable_hyde=False,
        )

        queries = state.get("search_queries", [state["question"]])
        all_docs: list[Document] = []
        all_scores: dict[str, float] = {}

        for sq in queries:
            docs = retrieve(
                query=sq,
                vectorstore=vectorstore,
                llm=llm,
                bm25=bm25,
                config=retrieval_cfg,
            )
            scores = _compute_retrieval_scores(sq, docs, embeddings)
            all_docs.extend(docs)
            all_scores.update(scores)

        # 去重
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in all_docs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:100])
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)

        logger.info(f"[Agent/multi_retrieve] 子查询合并: {len(all_docs)} → {len(unique)} 条")
        return {"documents": unique[:12], "retrieval_scores": all_scores}

    # ── Node: generate ──
    def generate_node(state: AgentState) -> dict:
        context = format_context(state.get("documents", []))
        answer = generate(llm, state["question"], context)
        return {"draft_answer": answer}

    # ── Node: claim_verify ──
    def claim_verify(state: AgentState) -> dict:
        if not agent_config.enable_claim_verification:
            return {
                "verified_answer": state.get("draft_answer", ""),
                "unverifiable_claims": [],
            }

        logger.sub("[Agent] 声明校验 (Claim Verification)")
        context = format_context(state.get("documents", []))
        result = _verify_claims(llm, state["draft_answer"], context)

        unverifiable = result["unverifiable_claims"]
        if unverifiable:
            logger.info(f"[Agent/claim_verify]   不可验证声明 ({len(unverifiable)} 条):")
            for c in unverifiable[:5]:
                logger.info(f"    - {c[:80]}...")
        else:
            logger.info("[Agent/claim_verify]   全部声明已验证通过")

        return {
            "verified_answer": result["corrected_answer"],
            "unverifiable_claims": unverifiable,
        }

    # ── Node: confidence_score ──
    def confidence_node(state: AgentState) -> dict:
        if not agent_config.enable_confidence_scoring:
            return {
                "confidence": 1.0,
                "confidence_level": "high",
                "answer": state.get("verified_answer") or state.get("draft_answer", ""),
                "metadata": {},
            }
        return _compute_confidence(state, agent_config)

    # ── 条件路由 ──
    def route_by_type(state: AgentState) -> str:
        qt = state.get("query_type", "simple")
        if qt == "macro":
            return "macro_retrieve"
        elif qt == "complex":
            return "multi_retrieve"
        return "simple_retrieve"

    # ── 构建图 ──
    graph = StateGraph(AgentState)

    graph.add_node("query_understand", query_understand)
    graph.add_node("simple_retrieve", simple_retrieve)
    graph.add_node("macro_retrieve", macro_retrieve)
    graph.add_node("multi_retrieve", multi_retrieve)
    graph.add_node("generate", generate_node)
    graph.add_node("claim_verify", claim_verify)
    graph.add_node("confidence_score", confidence_node)

    graph.add_edge(START, "query_understand")
    graph.add_conditional_edges("query_understand", route_by_type)
    graph.add_edge("simple_retrieve", "generate")
    graph.add_edge("macro_retrieve", "generate")
    graph.add_edge("multi_retrieve", "generate")
    graph.add_edge("generate", "claim_verify")
    graph.add_edge("claim_verify", "confidence_score")
    graph.add_edge("confidence_score", END)

    return graph.compile()
