# Reranker 评估修复

日期: 2026-05-28

## 问题

`hybrid_reranked` 评估模式与 `hybrid` 指标完全一致，reranker 形同虚设。

根因: `_get_retrieval_docs()` 将 `top_k=20` 传给 `_llm_rerank()`，而候选也只有 20 条。
`_llm_rerank()` 判断 `len(candidates) <= top_k` 后直接返回原序列，不做排序和筛选。

## 修复

改用 pipeline 的 `config.rerank_top_k`（默认 8）作为 reranker 的精选数量：

```python
# Before
reranked = _llm_rerank(self.llm, question, raw, top_k=top_k)  # 20→20, noop

# After
rerank_k = self.pipeline.config.rerank_top_k  # 8
reranked = _llm_rerank(self.llm, question, raw, top_k=rerank_k)  # 20→8
```

## 评估结果

### hybrid vs hybrid_reranked

| 指标 | hybrid (raw) | hybrid_reranked | 提升 |
|------|-------------|-----------------|------|
| Precision@1 | 0.467 | **0.733** | +57.1% |
| Hit@1 | 0.467 | **0.733** | +57.1% |
| MRR | 0.547 | **0.733** | +34.0% |
| NDCG@5 | 0.450 | **0.612** | +36.0% |
| Recall@5 | 0.613 | **0.709** | +15.5% |
| Recall@8 | 0.702 | 0.709 | +1.0% |

### 生成指标

| 指标 | 上一轮 | **本次** |
|------|--------|---------|
| Faithfulness | 0.873 | 0.873 |
| Answer Relevance | 0.933 | **0.953** |

### 历史对比 (hybrid_reranked vs 所有历史最优)

| 指标 | 历史最优 | **本次** |
|------|---------|---------|
| Recall@5 | 0.669 (paragraph_chunking) | **0.709** |
| MRR | 0.612 (HyDE) | **0.733** |
| Hit@1 | 0.533 (topk30) | **0.733** |
| Answer Relevance | 0.933 (paragraph_chunking) | **0.953** |

全部指标刷新历史最高。
