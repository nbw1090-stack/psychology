"""
RAG 评估主执行器。

编排检索评估和生成评估的完整流程：
1. 检索评估：LLM 标注相关 chunk → 计算 Recall@K, Precision@K, MRR, NDCG, Hit Rate, MAP
2. 生成评估：运行完整 RAG 管线 → 评估 Faithfulness 和 Answer Relevance
3. 详细日志：可选地记录每条用例的中间过程（chunk 标注、检索排名、LLM 裁判理由）
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from tqdm import tqdm

from ..pipeline import RAGPipeline, RAGConfig
from ..retriever import (
    retrieve, RetrievalConfig, BM25Retriever,
    _vector_search, _bm25_search, _rrf_fusion, _deduplicate_docs,
    _llm_rerank,
)
from ..vectorstore import load_vectorstore
from .. import logger as log

from .test_cases import TestCase, TestSuite
from .metrics import compute_all_retrieval_metrics
from .generation_eval import GenerationEvaluator

# 用于 LLM 标注 chunk 相关性的 prompt
CHUNK_LABEL_PROMPT = """你是一个文档相关性评估专家。请判断以下文本段落是否与用户问题相关。

用户问题: {question}

文本段落: {chunk}

请只回答 "相关" 或 "不相关"，不要输出其他任何内容。
如果段落中包含可以用于回答问题的信息，则为"相关"；否则为"不相关"。"""


@dataclass
class ModeResult:
    """单个检索模式的评估结果"""
    mode: str  # "vector" | "bm25" | "hybrid"
    metrics: Dict[str, float] = field(default_factory=dict)  # 各项指标平均值


@dataclass
class CaseResult:
    """单个测试用例的评估结果"""
    test_case: TestCase
    retrieval_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    generation_results: Dict[str, float] = field(default_factory=dict)
    generation_reasoning: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvalResult:
    """完整评估结果"""
    test_suite_name: str
    retrieval: Dict[str, Dict[str, float]] = field(default_factory=dict)  # mode → metrics
    generation: Dict[str, float] = field(default_factory=dict)  # metric → avg_score
    case_results: List[CaseResult] = field(default_factory=list)
    per_category: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)


class RAGEvaluator:
    """
    RAG 评估主类。

    编排检索评估和生成评估流程，汇总各项指标。
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        test_suite: TestSuite,
        llm=None,
    ):
        self.pipeline = pipeline
        self.test_suite = test_suite
        self.llm = llm or pipeline.llm
        self.gen_evaluator = GenerationEvaluator(self.llm)
        # 中间过程详情：每条用例一份记录
        self._details: List[dict] = []

    def get_details(self) -> List[dict]:
        """获取评估中间过程详情"""
        return self._details

    def save_details(self, filepath: str) -> None:
        """
        将评估中间过程详情保存为 JSON 文件。

        包含每条用例的：
        - 候选 chunk 文本 & LLM 相关性标注
        - 各检索模式的排名 & 逐条指标
        - 生成答案 & 评估分数 & 裁判理由
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._details, f, ensure_ascii=False, indent=2)
        print(f"\n评估详情已保存到: {filepath}")

    # ==================== 检索评估 ====================

    def _get_retrieval_docs(
        self, question: str, mode: str, top_k: int = 20, enable_rerank: bool = False
    ) -> List[Tuple[any, float]]:
        """
        使用指定检索模式获取文档列表。

        Args:
            question: 查询问题
            mode: "vector" | "bm25" | "hybrid" | "hybrid_reranked"
            top_k: 检索数量
            enable_rerank: 是否对结果应用 LLM reranker

        Returns:
            List of (Document, score) tuples
        """
        vs = load_vectorstore(
            self.pipeline.db_dir,
            self.pipeline._get_embeddings(),
            self.pipeline.config.collection_name,
        )

        if mode == "vector":
            raw = list(_vector_search(question, vs, top_k))
        elif mode == "bm25":
            bm25 = self.pipeline._get_bm25(vs)
            raw = list(_bm25_search(question, bm25, top_k))
        elif mode in ("hybrid", "hybrid_reranked"):
            bm25 = self.pipeline._get_bm25(vs)
            vec_results = list(_vector_search(question, vs, top_k))
            bm25_results = list(_bm25_search(question, bm25, top_k))
            raw = list(_rrf_fusion(vec_results, bm25_results))[:top_k]
        else:
            raise ValueError(f"未知检索模式: {mode}")

        # Apply reranker if requested (and mode implies it)
        if enable_rerank or mode == "hybrid_reranked":
            rerank_k = self.pipeline.config.rerank_top_k
            reranked = _llm_rerank(self.llm, question, raw, top_k=rerank_k)
            # Return as (doc, 0.0) tuples to maintain interface
            return [(doc, 0.0) for doc in reranked]

        return raw

    def _label_chunk_relevance(
        self, question: str, chunk_text: str
    ) -> bool:
        """
        用 LLM 判断单个 chunk 是否与问题相关。

        Returns:
            True 表示相关，False 表示不相关
        """
        from langchain_core.messages import HumanMessage

        prompt = CHUNK_LABEL_PROMPT.format(
            question=question,
            chunk=chunk_text[:1000],
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            return "相关" in text and "不相关" not in text
        except Exception as e:
            log.info(f"  [Eval] chunk 标注失败: {e}")
            return False

    def _build_ground_truth(
        self, question: str, candidate_chunks: List[str]
    ) -> Tuple[Set[str], Dict[str, float]]:
        """
        通过 LLM 标注构建 ground truth：判断每个候选 chunk 是否相关。

        Args:
            question: 查询问题
            candidate_chunks: 候选 chunk 文本列表（已去重）

        Returns:
            (relevant_ids, graded_relevance):
                - relevant_ids: 相关 chunk 的 ID 集合
                - graded_relevance: {chunk_id: score} 分级评分
        """
        relevant_ids = set()
        graded_relevance = {}

        for i, chunk_text in enumerate(candidate_chunks):
            # 用 chunk 文本的 hash 作为 ID
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            is_relevant = self._label_chunk_relevance(question, chunk_text)
            if is_relevant:
                relevant_ids.add(chunk_id)
                graded_relevance[chunk_id] = 1.0
            else:
                graded_relevance[chunk_id] = 0.0

        return relevant_ids, graded_relevance

    def _get_chunk_id(self, doc) -> str:
        """获取文档的唯一标识"""
        return hashlib.md5(doc.page_content.encode()).hexdigest()[:12]

    def evaluate_retrieval(
        self,
        modes: List[str] = None,
        k_values: List[int] = None,
        top_k: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        评估检索性能。

        对每条测试用例，分别用各检索模式检索 top-K 文档，
        通过 LLM 标注构建 ground truth，计算各项检索指标。

        Args:
            modes: 检索模式列表，默认 ["vector", "bm25", "hybrid"]
            k_values: K 值列表，默认 [1, 3, 5, 8]
            top_k: 每种模式检索的文档数（用于 ground truth 标注）
            verbose: 是否显示进度条

        Returns:
            {mode: {metric_name: avg_score}} 各级模式汇总指标
        """
        if modes is None:
            modes = ["vector", "bm25", "hybrid"]
        if k_values is None:
            k_values = [1, 3, 5, 8]

        log.section("检索评估 (Retrieval Evaluation)")
        log.keyval("测试用例数", str(len(self.test_suite)))
        log.keyval("检索模式", ", ".join(modes))
        log.keyval("K 值", ", ".join(str(k) for k in k_values))

        # 汇总：{mode: {metric: [scores]}}
        all_scores: Dict[str, Dict[str, List[float]]] = {
            mode: {} for mode in modes
        }

        # 清空上次详情
        self._details = []

        cases = self.test_suite.test_cases
        iterator = tqdm(cases, desc="检索评估") if verbose else cases

        for tc in iterator:
            if verbose:
                tqdm.write(f"\n  [{tc.id}] {tc.question[:60]}...")

            # Step 1: 收集所有模式的检索结果，构建候选池
            all_chunks: Dict[str, any] = {}  # chunk_id → doc
            all_chunks_data: Dict[str, dict] = {}  # chunk_id → {text, source, page}
            mode_rankings: Dict[str, List[str]] = {}  # mode → [chunk_ids]

            for mode in modes:
                docs_with_scores = self._get_retrieval_docs(tc.question, mode, top_k)
                mode_rankings[mode] = []
                for doc, score in docs_with_scores:
                    cid = self._get_chunk_id(doc)
                    # 用第一次遇到的 score
                    if cid not in all_chunks:
                        all_chunks[cid] = doc
                        all_chunks_data[cid] = {
                            "text": doc.page_content[:500],
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page"),
                            "chunk_id": doc.metadata.get("chunk_id"),
                            "best_score": float(score),
                        }
                    mode_rankings[mode].append(cid)

            # Step 2: LLM 标注构建 ground truth
            candidate_texts = [
                all_chunks[cid].page_content for cid in all_chunks
            ]
            unique_chunks = list(dict.fromkeys(candidate_texts))
            if verbose:
                tqdm.write(f"    候选 chunk 数: {len(unique_chunks)}, LLM 标注中...")

            # 逐条标注并记录结果
            chunk_labels: Dict[str, dict] = {}
            relevant_ids = set()
            graded_relevance = {}
            for idx, chunk_text in enumerate(unique_chunks):
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                is_relevant = self._label_chunk_relevance(tc.question, chunk_text)
                if is_relevant:
                    relevant_ids.add(chunk_id)
                    graded_relevance[chunk_id] = 1.0
                else:
                    graded_relevance[chunk_id] = 0.0
                # 记录标注结果
                chunk_labels[chunk_id] = {
                    "text_preview": chunk_text[:200],
                    "relevant": is_relevant,
                    "index": idx,
                }

            if verbose:
                tqdm.write(f"    相关 chunk: {len(relevant_ids)} / {len(unique_chunks)}")

            # Step 3: 计算每种模式的指标
            per_mode_metrics: Dict[str, dict] = {}
            for mode in modes:
                ranked_ids = mode_rankings[mode]
                metrics = compute_all_retrieval_metrics(
                    relevant_ids, ranked_ids, k_values, graded_relevance
                )
                per_mode_metrics[mode] = dict(metrics)
                for metric_name, score in metrics.items():
                    if metric_name not in all_scores[mode]:
                        all_scores[mode][metric_name] = []
                    all_scores[mode][metric_name].append(score)

            # Step 4: 保存本条用例的详细记录
            detail = {
                "id": tc.id,
                "category": tc.category,
                "question": tc.question,
                "reference_answer": tc.reference_answer[:300],
                "candidate_stats": {
                    "total_unique": len(unique_chunks),
                    "relevant": len(relevant_ids),
                    "irrelevant": len(unique_chunks) - len(relevant_ids),
                },
                "chunks": {
                    cid: {
                        **all_chunks_data.get(cid, {}),
                        "relevant": chunk_labels.get(cid, {}).get("relevant", False),
                    }
                    for cid in all_chunks
                },
                "mode_rankings": mode_rankings,
                "per_mode_metrics": per_mode_metrics,
            }
            self._details.append(detail)

        # Step 4: 汇总平均
        summary = {}
        for mode in modes:
            summary[mode] = {}
            for metric_name, scores in all_scores[mode].items():
                summary[mode][metric_name] = (
                    sum(scores) / len(scores) if scores else 0.0
                )

        return summary

    # ==================== 生成评估 ====================

    def evaluate_generation(
        self,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        评估生成质量。

        对每条测试用例运行完整 RAG 管线，评估 Faithfulness 和 Answer Relevance。

        Returns:
            {metric_name: avg_score} 各项指标平均值
        """
        log.section("生成评估 (Generation Evaluation)")
        log.keyval("测试用例数", str(len(self.test_suite)))

        faith_scores = []
        relevance_scores = []

        # 建立用例 ID → detail 映射（如果之前没有跑检索评估则新建）
        detail_map = {d["id"]: d for d in self._details}
        if not detail_map:
            # 单独跑生成评估时，初始化空的 detail 记录
            for tc in self.test_suite.test_cases:
                detail_map[tc.id] = {
                    "id": tc.id,
                    "category": tc.category,
                    "question": tc.question,
                    "reference_answer": tc.reference_answer[:300],
                }
            self._details = list(detail_map.values())

        cases = self.test_suite.test_cases
        iterator = tqdm(cases, desc="生成评估") if verbose else cases

        for tc in iterator:
            if verbose:
                tqdm.write(f"\n  [{tc.id}] {tc.question[:60]}...")

            # 运行完整管线
            result = self.pipeline.query(tc.question)
            answer = result["answer"]
            chunks = result["chunks"]
            contexts = [doc.page_content for doc in chunks]

            if verbose:
                tqdm.write(f"    回答: {answer[:100]}...")

            # 评估
            eval_result = self.gen_evaluator.evaluate(
                tc.question, answer, contexts
            )

            faith_scores.append(eval_result["faithfulness"])
            relevance_scores.append(eval_result["answer_relevance"])

            if verbose:
                tqdm.write(
                    f"    Faithfulness: {eval_result['faithfulness']:.2f}, "
                    f"Relevance: {eval_result['answer_relevance']:.2f}"
                )

            # 记录生成评估详情
            detail = detail_map.get(tc.id, {})
            detail["generation"] = {
                "answer": answer,
                "contexts": [
                    {
                        "text": ctx[:500],
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page"),
                    }
                    for ctx, doc in zip(contexts, chunks)
                ],
                "faithfulness": {
                    "score": eval_result["faithfulness"],
                    "reasoning": eval_result["faithfulness_reasoning"],
                },
                "answer_relevance": {
                    "score": eval_result["answer_relevance"],
                    "reasoning": eval_result["relevance_reasoning"],
                },
            }

        summary = {
            "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
            "answer_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
        }

        log.info(f"\n生成评估完成:")
        log.info(f"  Faithfulness:       {summary['faithfulness']:.3f}")
        log.info(f"  Answer Relevance:   {summary['answer_relevance']:.3f}")

        return summary

    # ==================== 按类别评估 ====================

    def evaluate_by_category(
        self,
        modes: List[str] = None,
        k_values: List[int] = None,
        top_k: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        按测试用例类别分别评估检索性能。
        """
        if modes is None:
            modes = ["vector", "bm25", "hybrid"]
        if k_values is None:
            k_values = [1, 3, 5, 8]

        results = {}
        categories = list(set(tc.category for tc in self.test_suite))

        for cat in categories:
            cat_cases = self.test_suite.get_by_category(cat)
            if not cat_cases:
                continue

            if verbose:
                print(f"\n{'='*60}")
                print(f"  类别: {cat} ({len(cat_cases)} 条)")
                print(f"{'='*60}")

            # 临时替换 test_suite
            original_cases = self.test_suite.test_cases
            self.test_suite.test_cases = cat_cases
            try:
                results[cat] = self.evaluate_retrieval(modes, k_values, top_k, verbose)
            finally:
                self.test_suite.test_cases = original_cases

        return results

    # ==================== 完整评估 ====================

    def run_full_eval(
        self,
        modes: List[str] = None,
        k_values: List[int] = None,
        top_k: int = 20,
        verbose: bool = True,
        details_path: str = None,
    ) -> EvalResult:
        """
        运行完整评估：检索 + 生成 + 按类别拆分。

        Args:
            modes: 检索模式列表
            k_values: K 值列表
            top_k: 候选池大小
            verbose: 是否显示进度
            details_path: 若提供，保存逐用例详情到此 JSON 文件

        Returns:
            EvalResult 包含所有评估数据
        """
        print("\n" + "=" * 60)
        print("  RAG 效果评估")
        print("=" * 60)
        print(f"  测试用例数: {len(self.test_suite)}")
        print(f"  覆盖类别:   {', '.join(self.test_suite.categories)}")
        print()

        # 1. 检索评估
        retrieval_summary = self.evaluate_retrieval(modes, k_values, top_k, verbose)

        # 2. 生成评估
        generation_summary = self.evaluate_generation(verbose)

        # 3. 按类别评估
        per_category = self.evaluate_by_category(modes, k_values, top_k, verbose)

        # 4. 保存详细过程
        if details_path:
            self.save_details(details_path)

        return EvalResult(
            test_suite_name=self.test_suite.name,
            retrieval=retrieval_summary,
            generation=generation_summary,
            per_category=per_category,
        )
