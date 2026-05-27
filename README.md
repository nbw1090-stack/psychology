# 🧠 Psychology RAG —— 心理学文档智能问答系统

基于 **RAG (Retrieval-Augmented Generation)** 的心理学文档问答系统，支持文档摄入、多模式检索、LLM 生成回答，以及完整的检索+生成效果评估。

## ✨ 特性

- 📄 **文档处理**：支持 PDF（数字版/扫描版 OCR）与 TXT 文档，句级感知分块
- 🔍 **多模式检索**：向量语义搜索 · BM25 关键词检索 · Hybrid 混合检索（RRF 融合）
- 🤖 **LLM 生成**：基于 DeepSeek-V4-Flash，严格依据检索上下文作答
- 🔄 **查询优化**：LLM 查询改写 + LLM 精排重排序
- 📊 **完整评估体系**：6 项检索指标 + 2 项生成质量指标（LLM-as-Judge）
- 🧪 **15 条测试用例**：覆盖事实查询、定义、对比、多跳推理、总结、负样本
- 📈 **基线归档**：评估结果自动归档，支持版本对比

## 🏗️ 架构

```
Raw Documents (PDF/TXT)
    ↓ [MuPDF / pytesseract OCR]
Text Extraction
    ↓ [Sentence-Aware Chunking]
Chunks (500 chars, 80 overlap)
    ↓ [DashScope text-embedding-v3]
Embeddings (1024-dim)
    ↓
Chroma Vector Store + BM25 Index
    ↓
User Query
    ↓ [Query Rewriting (LLM)]
    ↓ [Vector + BM25 → RRF Fusion]
Candidate Chunks
    ↓ [LLM Reranking → Top-K]
Context
    ↓ [DeepSeek-V4-Flash]
Answer + Sources
```

## 📁 目录结构

```
psychology/
├── main.py                     # 入口
├── rag_agent/                  # 核心模块
│   ├── cli.py                  # 命令行接口
│   ├── pipeline.py             # RAG 流水线与配置
│   ├── document.py             # 文档加载与分块
│   ├── embeddings.py           # DashScope 嵌入封装
│   ├── vectorstore.py          # Chroma 向量存储
│   ├── retriever.py            # 多模式检索引擎
│   ├── generator.py            # LLM 答案生成
│   ├── logger.py               # 结构化日志
│   └── evaluation/             # 评估子系统
│       ├── test_cases.py       # 测试用例定义
│       ├── metrics.py          # 检索指标计算
│       ├── generation_eval.py  # 生成质量评估
│       ├── runner.py           # 评估执行器
│       └── reporter.py         # 结果格式化与导出
├── tests/                      # 单元测试（103 个）
├── data/
│   ├── documents/              # 源文档
│   ├── chunks/                 # 分块缓存
│   └── chroma_db/              # Chroma 持久化
├── docs/                       # 文档与问题记录
└── benchmarks/                 # 评估基线归档
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [DashScope API Key](https://dashscope.aliyun.com/)（用于 Embedding 和 LLM）

### 安装

```bash
git clone <repo-url>
cd psychology

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置 API Key

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 1. 摄入文档

```bash
# 将 PDF/TXT 文档放入 data/documents/，然后运行：
python main.py ingest \
  --docs-dir data/documents \
  --db-dir data/chroma_db \
  --chunk-size 500 \
  --chunk-overlap 80 \
  --collection rag_agent
```

### 2. 查询

```bash
# 单次查询
python main.py query "什么是广泛性焦虑障碍(GAD)？" --mode hybrid

# 交互式对话
python main.py chat --show-sources --mode hybrid
```

### 3. 运行评估

```bash
python main.py evaluate \
  --eval-modes vector,bm25,hybrid \
  --k-values 1,3,5,8 \
  --by-category \
  --output results.json
```

### 4. 运行测试

```bash
pytest tests/ -v
```

## 📖 使用指南

### CLI 命令

| 命令 | 说明 |
|------|------|
| `python main.py ingest` | 摄入文档，构建向量索引 + BM25 索引 |
| `python main.py query "<问题>"` | 单次问答 |
| `python main.py chat` | 交互式对话模式 |
| `python main.py evaluate` | 运行评估套件 |

### 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `hybrid` | 检索模式：`vector` / `bm25` / `hybrid` |
| `--top-k` | `8` | 检索返回的 chunk 数量 |
| `--rerank-top-k` | `4` | LLM 精排后保留数量 |
| `--no-rerank` | — | 禁用 LLM 精排 |
| `--no-rewrite` | — | 禁用查询改写 |

### 评估参数

| 参数 | 说明 |
|------|------|
| `--eval-modes` | 要对比的检索模式（逗号分隔） |
| `--k-values` | 评估的 K 值列表 |
| `--categories` | 按类别筛选测试用例 |
| `--by-category` | 输出按类别拆分的指标 |
| `--retrieval-only` | 仅评估检索指标 |
| `--generation-only` | 仅评估生成质量 |
| `--output` | JSON 结果输出路径 |
| `--save-details` | 逐用例详情保存路径 |

### 评估指标

**检索指标：** Recall@K · Precision@K · MRR · NDCG@K · Hit Rate · MAP

**生成指标：** Faithfulness（忠实度） · Answer Relevance（回答相关性）

## 📊 基线结果（2026-05-22）

| 指标 | vector | bm25 | **hybrid** |
|------|--------|------|-------------|
| Recall@5 | 0.514 | 0.487 | **0.564** |
| MRR | 0.330 | 0.365 | **0.383** |
| Hit@5 | 0.467 | 0.467 | **0.533** |
| Faithfulness | — | — | **1.000** |
| Relevance | — | — | **0.527** |

> Hybrid（混合检索）在所有核心指标上全面领先。详见 [`benchmarks/2026-05-22_baseline/`](benchmarks/2026-05-22_baseline/ARCHITECTURE.md)。

## 📚 语料库

| 文档 | Chunk 数 |
|------|----------|
| 焦虑心理学 | 269 |
| 乌合之众：大众心理研究 | 419 |
| 人人都该懂的心理学 | 520 |
| **合计** | **1,208** |

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| 嵌入模型 | DashScope text-embedding-v3 (1024-dim) |
| LLM | DeepSeek-V4-Flash |
| 向量数据库 | Chroma (SQLite) |
| BM25 | scikit-learn TfidfVectorizer + jieba 分词 |
| 文档解析 | MuPDF + pytesseract OCR |
| 测试框架 | pytest |

## 📝 已知问题

- **全局性总结受限**：当前 chunk 级检索无法有效回答"这本书讲了什么"等宏观问题，详见 [`docs/RAG_issue.md`](docs/RAG_issue.md)

## 📄 License

MIT
