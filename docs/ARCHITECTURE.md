# RAG 系统架构设计

**最后更新**: 2026-05-28
**版本**: v2.0

---

## 一、系统全景

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLI 入口 (cli.py)                        │
│  python main.py ingest | query | chat | evaluate                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     Pipeline (pipeline.py)                        │
│  RAGPipeline.ingest()  — 文档导入 + 段落切分 + 层级索引构建        │
│  RAGPipeline.query()   — 检索 + 生成                              │
└───┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────────┐
│paragraph │ │embedd- │ │vector- │ │retriever│ │   generator.py   │
│chunker.py│ │ings.py │ │store.py│ │  .py    │ │                  │
│OCR清洗   │ │DashScope│ │ChromaDB│ │BM25+向量│ │DeepSeek-V4-Flash │
│章节检测   │ │text-emb│ │        │ │+RRF融合 │ │严格grounded生成  │
│段落重建   │ │v4      │ │        │ │LLM Rerank│ │区分核心/补充文档  │
│链式meta  │ │+instruct│ │        │ │HyDE     │ │                  │
└──────────┘ └────────┘ └────────┘ └────────┘ └──────────────────┘
    ▲
┌───┴──────────┐
│ hierarchy.py │
│ 章节检测     │
│ LLM摘要生成  │
│ 层级路由     │
└──────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     Evaluation (evaluation/)                      │
│  test_cases.py  runner.py  metrics.py  generation_eval.py        │
│  支持 hybrid_reranked 评估模式 (含 LLM Rerank)                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、模块职责

### 2.1 `paragraph_chunker.py` — 段落级切分 (核心)

从 OCR 扫描文本重建语义完整的段落级 chunks。

| 步骤 | 实现 |
|------|------|
| OCR 文本清洗 | 合并 ~40 字换行，移除页码标记 (`第 X 页`) |
| 章节检测 | 正则 `第X章/卷` + 交叉引用过滤 + 小章节合并 |
| 段落重建 | 句号处切分，目标 1200 字，硬上限 2400 字 |
| 链式 metadata | `chunk_id`, `prev/next_chunk_id`, `book_title`, `chapter_title` |

**metadata 结构**:
```python
{
    "source": "焦虑心理学.pdf",
    "book_title": "焦虑心理学",
    "chapter_title": "第七章 广泛性焦虑症",
    "chapter_index": 6,
    "paragraph_index": 2,
    "chunk_id": "焦虑心理学_6_2",
    "prev_chunk_id": "焦虑心理学_6_1",
    "next_chunk_id": "焦虑心理学_6_3",
    "page": 134,
}
```

### 2.2 `embeddings.py` — 向量化

| 功能 | 实现 |
|------|------|
| 模型 | DashScope `text-embedding-v4` (原生 SDK) |
| 维度 | 1024 (可配置 32-2560) |
| 文档向量化 | `text_type="document"` |
| 查询向量化 | `text_type="query"` + `instruct="Retrieve relevant paragraphs..."` |
| 批量 | 一次性批量调用 |

> **关键决策**: v4 的 `text_type` 和 `instruct` 参数只能通过 DashScope 原生 SDK 使用，OpenAI 兼容接口不支持。因此从 `openai.OpenAI` 切换到了 `dashscope.TextEmbedding`。

### 2.3 `retriever.py` — 检索

| 组件 | 说明 |
|------|------|
| Vector 检索 | ChromaDB cosine similarity |
| BM25 检索 | jieba 分词 + TF-IDF + BM25 (k1=1.5, b=0.75) |
| RRF 融合 | k=60，向量 + BM25 各取 top-K 后加权合并 |
| LLM Rerank | 候选 docs 用 LLM 评分 (0-10)，截断 800 字/段，精选 top-8 |
| HyDE | LLM 生成假设文档，替代原始查询做检索 |
| 查询改写 | LLM 拆解/补充用户查询 |
| 上下文扩展 | 通过链式 metadata 拉取前后段落 (默认关闭) |
| format_context | 分离核心/补充文档，增强来源显示 |

**检索流程 (微观问题)**:
```
问题 → [HyDE 或 查询改写]
     → 向量检索(30) + BM25检索(30)
     → RRF融合 → LLM Rerank(精选8)
     → [可选: Context Expansion]
     → format_context (核心+补充分离)
```

### 2.4 `generator.py` — 生成

| 功能 | 实现 |
|------|------|
| LLM | DeepSeek-V4-Flash (via DashScope) |
| Prompt 策略 | 区分核心文档和补充上下文 |
| 幻觉防护 | 禁止引用不存在的文档、禁止编造数据/术语/标准 |

### 2.5 `hierarchy.py` — 层级索引

```
L1 文档级 (3 nodes)
  ├── 全书摘要 (LLM 基于各章摘要汇总)
  └── 章节列表
       │
L2 章节级 (~30 nodes)
  ├── 章节标题
  ├── LLM 摘要
  └── jieba 关键词
       │
L3 段落级 (296 chunks) — paragraph_chunker 产出
```

**检索路由**:
- **宏观问题** (is_macro_query): L1+L2 摘要优先 → L3 补充 → 融合
- **微观问题**: 完整管线 (BM25+向量+RRF+Rerank)

### 2.6 `evaluation/` — 效果评估

| 组件 | 功能 |
|------|------|
| test_cases.py | 15 条测试用例，6 种类型 |
| runner.py | 支持 hybrid / hybrid_reranked 模式 |
| metrics.py | Recall@K, Precision@K, MRR, NDCG, Hit Rate, MAP |
| generation_eval.py | Faithfulness + Answer Relevance (LLM-as-Judge) |
| reporter.py | 控制台表格 + JSON 导出 |

---

## 三、数据文件布局

```
data/
├── documents/                          # 原始 PDF (gitignore)
│   ├── 焦虑心理学.pdf
│   ├── 乌合之众：大众心理研究.pdf
│   └── 人人都该懂的心理学.pdf
│
├── chroma_db/                          # 向量库 (gitignore)
│   ├── chroma.sqlite3                  # 320 条向量 (296段落 + 24摘要)
│   └── bm25_index.pkl                 # BM25 索引
│
└── chunks/                             # 切分产物 (git tracked)
    ├── _manifest.json                  # 切分统计
    ├── 焦虑心理学_*.json               # 68 段落
    ├── 乌合之众：大众心理研究_*.json    # 102 段落
    ├── 人人都该懂的心理学_*.json        # 126 段落
    └── hierarchy_*.json                # 层级索引
```

---

## 四、API 与模型依赖

| 服务 | 模型 | 用途 | 环境变量 |
|------|------|------|----------|
| DashScope | text-embedding-v4 (原生SDK) | 向量化 | DASHSCOPE_API_KEY |
| DashScope | DeepSeek-V4-Flash | 生成 + 摘要 + 精排 | OPENAI_API_KEY |
| ChromaDB | SQLite | 向量存储 | 无 |

---

## 五、关键设计决策

| 决策 | 理由 |
|------|------|
| 段落级切分替代 sentence chunking | 语义完整性更好，chunk 从 784 降到 296，Recall@5 +40% |
| DashScope 原生 SDK 替代 OpenAI 兼容接口 | v4 的 text_type/instruct 只在原生 SDK 可用 |
| 默认关闭 Context Expansion | 实测引入噪声导致 Faithfulness 从 0.93 跌到 0.61 |
| 生成 prompt 区分核心/补充文档 | 防止扩展段落噪声影响回答质量 |
| LLM Rerank 精选 8 条而非全量返回 | MRR 从 0.547 提升到 0.733 (+34%) |
| 层级摘要与段落同向量空间 | 微观查询时摘要被自然淘汰，无需硬隔离 |
