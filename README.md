# Psychology RAG — 心理学文档智能问答系统

基于 **RAG (Retrieval-Augmented Generation)** 的心理学文档问答系统，支持段落级切分、层级索引、混合检索 + LLM 精排，以及完整的检索+生成效果评估。

## 特性

- **段落级切分**: OCR 文本重建为语义完整段落 (~1200字)，携带章节/前后链式 metadata
- **v4 Embedding**: DashScope text-embedding-v4，启用 text_type + instruct 参数
- **层级索引**: 文档级 → 章节级 → 段落级三层索引，宏观/微观问题自动路由
- **混合检索 + Rerank**: Vector + BM25 → RRF 融合 → LLM 精选，支持 HyDE 和查询改写
- **严格生成**: 区分核心/补充文档，禁止编造引用
- **完整评估**: 6 项检索指标 + 2 项生成质量指标 (LLM-as-Judge)，15 条测试用例

## 当前指标 (2026-05-28, hybrid_reranked)

| 指标 | 数值 | 说明 |
|------|------|------|
| Recall@5 | **0.709** | 前 5 条结果覆盖 71% 的相关信息 |
| MRR | **0.733** | 首个相关结果平均在第 1.4 位 |
| Hit@1 | **0.733** | 73% 的情况第一条就命中 |
| Faithfulness | 0.873 | 87% 的回答内容有据可依 |
| Answer Relevance | **0.953** | 95% 的回答切题 |

> 所有检索指标均为历史最高。完整评估历史见 [benchmarks/README.md](benchmarks/README.md)。

## 架构

```
Raw Documents (PDF)
    ↓ [MuPDF + OCR text extraction]
    ↓ [Paragraph Chunker: OCR清洗 → 章节检测 → 段落重建]
Paragraph Chunks (~1200 chars, with chapter/prev/next metadata)
    ↓ [text-embedding-v4 (text_type=document, dim=1024)]
    ↓ + [Hierarchy: L1 全书摘要 → L2 章节摘要 → L3 段落]
ChromaDB (320 records) + BM25 Index
    ↓
User Query
    ↓ [HyDE / Query Rewrite]
    ↓ [is_macro_query? → L1+L2摘要优先 / 否则 ↓]
    ↓ [Vector(top-30) + BM25(top-30) → RRF Fusion]
    ↓ [LLM Rerank → Top-8]
Context (core docs + supplementary)
    ↓ [DeepSeek-V4-Flash: strict grounded generation]
Answer + Sources
```

## 快速开始

```bash
# 安装
pip install -r requirements.txt
export DASHSCOPE_API_KEY="your-api-key"

# 摄入文档 (默认: 段落切分 1200字)
python main.py ingest

# 查询 (默认: hybrid + rerank + HyDE)
python main.py query "什么是灾难化思维？"

# 交互对话
python main.py chat --show-sources

# 评估
python main.py evaluate --eval-modes hybrid,hybrid_reranked
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `hybrid` | 检索模式: vector / bm25 / hybrid |
| `--top-k` | `30` | 候选检索数量 |
| `--rerank-top-k` | `8` | LLM 精选保留数量 |
| `--chunk-size` | `1200` | 段落目标大小 |
| `--chunk-strategy` | `paragraph` | 切分策略: paragraph / sentence / recursive |
| `--no-rerank` | — | 禁用 LLM 精排 |
| `--no-rewrite` | — | 禁用查询改写 |
| `--no-hyde` | — | 禁用 HyDE 假设文档检索 |

## 语料库

| 文档 | 段落数 | 平均字数 |
|------|--------|---------|
| 焦虑心理学 | 68 | 1212 |
| 乌合之众：大众心理研究 | 102 | 1217 |
| 人人都该懂的心理学 | 126 | 1248 |
| **合计** | **296** | **1227** |

## 技术栈

| 层级 | 技术 |
|------|------|
| Embedding | DashScope text-embedding-v4 (1024-dim, text_type + instruct) |
| LLM | DeepSeek-V4-Flash (via DashScope) |
| 向量数据库 | ChromaDB (SQLite) |
| BM25 | scikit-learn TfidfVectorizer + jieba |
| 文档解析 | MuPDF + pytesseract OCR |

## RAG 迭代历程

| 版本 | 日期 | 变更 | 核心收益 |
|------|------|------|---------|
| v1 基线 | 05-22 | sentence chunk (500字) + v3 embedding | 端到端管线跑通 |
| v2 层级索引 | 05-23 | L1文档→L2章节→L3段落 三层索引 | MRR +50%, 宏观问题可回答 |
| v3 扩大检索 | 05-25 | top_k 8→30 | MRR +11%, Hit@8=0.800 |
| v4 HyDE | 05-27 | 假设文档检索 | MRR=0.612 (+27%) |
| v5 Sentence Window | 05-27 | 单句嵌入+上下文扩展 | 全面退化, 放弃 |
| **v6 段落切分+v4** | **05-28** | **段落级切分(1200字)+v4+严格生成** | **Recall@5 +40%, Relevance +50%** |
| v6.1 Reranker修复 | 05-28 | 评估中reranker生效 (20→8精选) | MRR +34%, 全指标最高 |

> 核心洞察: 整个迭代围绕 **语义完整性 vs 检索精度** 的矛盾。从 500 字 sentence 到 1200 字 paragraph，配合 reranker 过滤噪声，找到了平衡点。

详见 [docs/RAG_issue.md](docs/RAG_issue.md)。

## 项目状态

**RAG 索引侧**: 已达瓶颈，指标稳定在历史最高水平。

**下一阶段**: Agent 层优化，目标 Faithfulness 87% → 95%+，支持商用发布。

| 方向 | 目标 | 优先级 |
|------|------|--------|
| 两阶段生成 (Claim Verification) | Faithfulness 95%+ | P0 |
| 查询分解 (Query Decomposition) | 复杂问题 Relevance 提升 | P1 |
| 置信度感知 (Confidence-Aware) | 降低误导风险 | P2 |

详见 [docs/AGENT_ROADMAP.md](docs/AGENT_ROADMAP.md)。

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构设计，模块职责与数据流 |
| [docs/RAG_issue.md](docs/RAG_issue.md) | RAG 迭代记录，每轮变更的动机/效果/根因 |
| [docs/AGENT_ROADMAP.md](docs/AGENT_ROADMAP.md) | Agent 层优化路线图，商用发布检查清单 |
| [benchmarks/README.md](benchmarks/README.md) | 评估基线归档与对比 |

## 目录结构

```
psychology/
├── main.py                     # 入口
├── rag_agent/                  # 核心模块
│   ├── cli.py                  # 命令行接口
│   ├── pipeline.py             # RAG 流水线与配置
│   ├── document.py             # 文档加载与切分 (sentence/recursive)
│   ├── paragraph_chunker.py    # 段落级切分 (章节检测+链式metadata)
│   ├── embeddings.py           # DashScope text-embedding-v4
│   ├── vectorstore.py          # Chroma 向量存储
│   ├── retriever.py            # 混合检索 + RRF + Rerank
│   ├── generator.py            # LLM 答案生成 (严格grounded)
│   ├── hierarchy.py            # 三层层级索引
│   ├── logger.py               # 结构化日志
│   └── evaluation/             # 评估子系统
├── tests/                      # 单元测试
├── data/
│   ├── documents/              # 源文档 (PDF)
│   ├── chunks/                 # 切分产物 + manifest
│   └── chroma_db/              # Chroma 持久化
├── docs/                       # 架构文档
└── benchmarks/                 # 评估基线归档
```

## License

MIT
