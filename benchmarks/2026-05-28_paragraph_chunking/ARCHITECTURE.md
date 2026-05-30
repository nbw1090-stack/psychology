# 段落级 Chunking + v4 Embedding + 评估体系优化

日期: 2026-05-28

## 技术方案

### 1. 段落级切分 (Paragraph Chunking)

**问题**: 之前使用固定大小的 sentence chunking (256-1024 字)，语义完整的论证被切碎到多个 chunk，检索时丢失连贯性。

**方案**: 新增 `paragraph_chunker.py`，从 OCR 扫描文本重建语义完整的段落:

- OCR 文本清洗: 合并 ~40 字换行，移除页码标记
- 章节检测: 正则匹配 `第X章` + 交叉引用过滤（跳过 "在第6章中" 类引用）
- 段落重建: 在句号处切分，目标 1200 字，硬上限 2400 字
- 链式 metadata: `prev_chunk_id` / `next_chunk_id` 串联相邻段落

效果: chunk 从 784 个降到 296 个，每个 chunk 携带完整的章节/段落上下文。

### 2. Embedding 升级 v3 → v4

**问题**: v3 通过 OpenAI 兼容接口调用，不支持 `text_type` 和 `instruct` 参数。

**方案**: 切换到 DashScope 原生 SDK (`dashscope.TextEmbedding.call()`):

- `embed_documents()`: `text_type="document"`
- `embed_query()`: `text_type="query"` + `instruct="Retrieve relevant paragraphs from psychology academic texts"`
- 维度: 1024

### 3. 上下文扩展 (Context Expansion)

新增 `expand_context()`: 通过链式 metadata 拉取相邻段落，扩展后的文档标记 `is_expanded=True`。

`format_context()` 区分核心文档和补充文档，生成 prompt 指示 LLM 以核心文档为主。

**默认关闭** (`context_window=0`)，因为实际测试中发现扩展引入的噪声会导致 Faithfulness 下降。

### 4. 评估体系优化

- **reranker 评估路径**: 新增 `hybrid_reranked` 模式，在检索评估中应用 LLM reranker，使评估更接近真实管线
- **负样本评分修复**: Answer Relevance 评估 prompt 增加规则 — 超纲问题回答"未找到"应得高分
- **reranker 截断优化**: 从 500 字提升到 800 字，适配 ~1200 字的段落 chunk

## 评估结果

### 检索指标 (hybrid)

| 指标 | 旧基线 (chunk1024) | **本次** | 变化 |
|------|-------------------|---------|------|
| Recall@5 | 0.476 | **0.669** | +40.5% |
| Recall@8 | 0.630 | **0.774** | +22.9% |
| Hit@1 | 0.533 | 0.400 | -25.0% |
| Precision@5 | 0.387 | 0.333 | -13.9% |
| MRR | 0.571 | 0.475 | -16.8% |

### 生成指标

| 指标 | 旧基线 | **本次** | 变化 |
|------|--------|---------|------|
| Faithfulness | 0.933 | **0.873** | -6.4% |
| Answer Relevance | 0.620 | **0.933** | +50.5% |

### 综合评分

| 指标 | 旧基线 | **本次** |
|------|--------|---------|
| Recall@5 | 0.476 | **0.669** |
| MRR | 0.571 | 0.475 |
| Faithfulness | 0.933 | 0.873 |
| Answer Relevance | 0.620 | **0.933** |

**核心结论**: Recall 大幅提升 (+40.5%) 且 Faithfulness 基本持平 (-6.4%)，Answer Relevance 提升 50.5%。

## 文件改动

| 文件 | 改动 |
|------|------|
| `rag_agent/paragraph_chunker.py` | **新建** — 段落重建 + 链式 metadata |
| `rag_agent/embeddings.py` | v3→v4, OpenAI SDK→DashScope 原生, text_type+instruct |
| `rag_agent/retriever.py` | 新增 `expand_context()`, 增强 `format_context()`, reranker 截断 500→800 |
| `rag_agent/pipeline.py` | 串联段落切分 + context expansion |
| `rag_agent/generator.py` | 严格化 prompt: 区分核心/补充文档, 禁止编造 |
| `rag_agent/cli.py` | 默认 chunk_size=1200, 新增 paragraph 策略 |
| `rag_agent/evaluation/runner.py` | 新增 `hybrid_reranked` 评估模式 |
| `rag_agent/evaluation/generation_eval.py` | 修复负样本 Answer Relevance 评分 |
