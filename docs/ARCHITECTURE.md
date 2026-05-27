# RAG 系统架构设计

**日期**: 2026-05-23  
**版本**: v1.0

---

## 一、系统全景

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI 入口 (cli.py)                      │
│  python main.py ingest | query | interactive | evaluate       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   Pipeline (pipeline.py)                      │
│  RAGPipeline.ingest()  — 文档导入 + 索引构建                   │
│  RAGPipeline.query()   — 检索 + 生成                          │
└───┬──────────┬──────────┬──────────┬────────────┬────────────┘
    │          │          │          │            │
    ▼          ▼          ▼          ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
│document│ │embedd- │ │vector- │ │retriever │ │  generator   │
│  .py   │ │ings.py │ │store.py│ │   .py    │ │    .py       │
│PDF加载 │ │DashScope│ │ChromaDB│ │BM25+向量 │ │DeepSeek-V4   │
│切分落盘│ │text-emb│ │        │ │+RRF+Rerank│ │Flash (生成)  │
└────────┘ │edding-v3│ └────────┘ └──────────┘ └──────────────┘
           └────────┘
    ▲
┌───┴──────────┐
│ hierarchy.py │  ← 新增：三层索引
│ 章节检测     │
│ LLM摘要生成  │
│ 层级路由     │
└──────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Evaluation (evaluation/)                    │
│  test_cases.py  runner.py  metrics.py  generation_eval.py    │
│  reporter.py                                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、模块职责

### 2.1 `document.py` — 文档加载与切分

| 功能 | 实现 |
|------|------|
| PDF 加载 | `fitz` (PyMuPDF) 直接提取文本，抑制加密 PDF 报错 |
| OCR 回退 | `pdf2image` + `pytesseract`，扫描版 PDF 自动 OCR |
| 文本切分 | `sentence` 策略：按句号/问号/感叹号断句，保证不在句子中间切断 |
| Chunk 参数 | size=500, overlap=80 (可配置) |
| 本地落盘 | `save_chunks_to_json()` — 每个 PDF 一个 JSON，按时间戳命名 |

**数据流**:
```
PDF → fitz提取文本 → Document对象 → sentence切分 → chunks → JSON落盘
```

### 2.2 `embeddings.py` — 向量化

| 功能 | 实现 |
|------|------|
| 模型 | DashScope `text-embedding-v3` |
| 维度 | 1024 |
| 批量 | 每批 10 条文本 |
| 接口 | 兼容 LangChain `Embeddings` 基类 |

### 2.3 `vectorstore.py` — 向量存储

| 功能 | 实现 |
|------|------|
| 存储引擎 | ChromaDB (SQLite3 + Parquet) |
| 路径 | `data/chroma_db/` |
| Collection | `rag_agent` (1244 条记录) |
| 批量写入 | 每批 10 条 |

### 2.4 `retriever.py` — 检索

| 模式 | 说明 |
|------|------|
| `vector` | 纯语义检索 (Cosine Similarity) |
| `bm25` | 纯关键词检索 (jieba 分词 + TF-IDF + BM25) |
| `hybrid` | RRF 融合 (k=60)，向量 + BM25 各取 top-K 后加权合并 |
| LLM Rerank | 对候选 docs 用 LLM 评分 (0-10)，取 top rerank_top_k |
| Query Rewrite | LLM 改写用户问题，增强召回 |

**检索流程 (微观问题)**:
```
问题 → [改写] → 向量检索 + BM25检索 → RRF融合 → [LLM Rerank] → Top-K docs
```

### 2.5 `generator.py` — 生成

| 功能 | 实现 |
|------|------|
| LLM | DeepSeek-V4-Flash (via DashScope) |
| System Prompt | 严格基于上下文回答，信息不足时明确告知 |
| 输入 | System Prompt + 上下文 + 用户问题 |

### 2.6 `hierarchy.py` — 层级索引 (新增)

三层结构：

```
L1 文档级 (3 nodes)
  ├── 全书摘要 (LLM 基于各章摘要汇总)
  └── 章节列表
       │
L2 章节级 (~50 nodes)
  ├── 章节标题
  ├── LLM 摘要 (100-200字)
  └── jieba 关键词
       │
L3 段落级 (1208 chunks) — 不变
```

**章节检测策略**:

| PDF | 检测方式 | 结果 |
|-----|----------|------|
| 焦虑心理学.pdf | 正则 `第X章` + 目录提取 | 13 节 ✅ |
| 乌合之众.pdf | 正则 `提要` + 卷名 | 13 节 ✅ |
| 人人都该懂的心理学.pdf | 正则 `第X章` + 目录提取 | 7 节 ⚠️ |

**检索路由** (`is_macro_query`):

```
问题 ──→ 含"总结/概括/全书/核心观点/讲了什么/框架/解读..."?
         │
    Yes  ├─→ L1+L2 摘要检索 (k=6, filter source="hierarchy")
         │   └─→ L3 段落补充 (k=6, 无过滤)
         │   └─→ 去重合并 → LLM 生成
         │
    No   └─→ 混合检索 (BM25+向量+hybrid+rerank)
              └─→ 检索全部 1244 条 (L3段落 + L1/L2摘要)
              └─→ 具体问题的段落相似度远高于摘要
              └─→ 摘要被自然挤出 top-K ≈ 等效 L3 查询
```

> **注意**：No 分支没有 `source="hierarchy"` 过滤，摘要和段落在同一向量空间中竞争。对于具体概念查询（如"What is GAD?"），段落 chunk 的相似度天然高于章节摘要，摘要会被自然淘汰。这比硬性过滤更灵活——边界模糊的问题仍有机会命中摘要。

#### 路由设计详解

两个分支的**本质区别**不在于"查不查 L3"，而在于**摘要的参与方式**：

| 维度 | Yes (宏观) | No (微观) |
|------|-----------|-----------|
| 摘要参与 | **强制优先** — 先单独查 `source="hierarchy"` 取 6 条 | **被动竞争** — 与 1208 条段落混合排序 |
| 段落参与 | 补充 6 条 | 走完整检索管线 (BM25+向量+RRF+Rerank) |
| 摘要命中概率 | 100%（保证命中） | ~0%（被具体段落的高相似度挤出） |
| 为什么这样设计 | 宏观问题需要摘要的全局视野 | 边界模糊问题（如"焦虑症有哪些治疗方法"）既非纯宏观也非纯微观，不应人为剥夺摘要参与权 |

**核心设计思想**：不把 L1/L2 摘要和 L3 段落放在两个隔离的"池"里，而是放在**同一个向量空间**中。宏观问题时，通过显式过滤把摘要"捞"出来；微观问题时，让向量相似度自然决定——如果摘要碰巧和问题高度相关，它就有资格被召回。

### 2.7 `evaluation/` — 效果评估

| 组件 | 功能 |
|------|------|
| `test_cases.py` | 15 条测试用例，覆盖 6 种类型 |
| `runner.py` | 检索评估 + 生成评估编排 |
| `metrics.py` | Recall@K, Precision@K, MRR, NDCG, Hit Rate, MAP |
| `generation_eval.py` | Faithfulness + Answer Relevance (LLM-as-Judge) |
| `reporter.py` | 控制台表格 + JSON 导出 |

**测试用例类型**:

| 类型 | 数量 | 示例 |
|------|------|------|
| factual_lookup | 3 | "广泛性焦虑症的诊断标准是什么？" |
| definition | 3 | "什么是习得性无助？" |
| comparison | 2 | "焦虑和恐惧的核心区别？" |
| multi_hop | 2 | "CBT 核心技术如何应用于焦虑症？" |
| summary | 2 | "乌合之众的核心论点是什么？" |
| negative | 3 | "Python 如何实现多线程？" (不应回答) |

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
│   └── chroma.sqlite3                  # 1244 条向量
│
└── chunks/                             # 文本产物 (git tracked)
    ├── _manifest.json                  # 切分统计
    ├── 焦虑心理学_*.json               # 269 chunks
    ├── 乌合之众：大众心理研究_*.json    # 419 chunks
    ├── 人人都该懂的心理学_*.json        # 520 chunks
    └── hierarchy_*.json                # 三层索引 (36KB)
```

---

## 四、API 与模型依赖

| 服务 | 模型/接口 | 用途 | 环境变量 |
|------|-----------|------|----------|
| DashScope | `text-embedding-v3` | 文本向量化 (1024 dim) | `DASHSCOPE_API_KEY` |
| DashScope | `DeepSeek-V4-Flash` | RAG 生成 + 摘要 + 精排 | `OPENAI_API_KEY` |
| ChromaDB | 本地 SQLite | 向量存储与检索 | 无 |

---

## 五、关键设计决策

| 决策 | 理由 |
|------|------|
| 自建层级索引而非 GraphRAG | 3 本书规模小，GraphRAG 成本高 ($15-60)，过度设计 |
| 用 fitz 替代 PyMuPDFLoader | 加密 PDF 报错可抑制，文本提取更可控 |
| 章节摘要存为向量而非独立索引 | 复用 Chroma，检索统一，source 字段区分层级 |
| 宏观判断用关键词而非 LLM | 16 个关键词覆盖大多数场景，零延迟零成本 |
| Chunk JSON 按 PDF 分文件 | 支持增量索引，方便人工审查和调优 |
| 层级 JSON 独立于 chunk JSON | 层级关系可能重新生成，不污染 chunk 数据 |
| 评估详情自动保存时间戳 | 不被覆盖，支持多次评估结果对比 |

---

## 六、已知局限与改进方向

详见 `docs/RAG_issue.md` — Issue #2。

| 局限 | 影响 | 优先度 |
|------|------|--------|
| 宏观判断靠关键词 (16个) | 漏判率中等 | 中 |
| L3 段落偏边缘（附录/书评） | 宏观回答不够聚焦正文 | 中 |
| 人人都该懂的心理学 章节检测偏少 | 该书层级粒度粗 | 低 |
| 来源显示 `hierarchy` 不友好 | 用户体验 | 低 |
| 摘要检索无 rerank | 宏观问题排序精度 | 低 |
