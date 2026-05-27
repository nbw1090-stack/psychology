# Benchmarks

RAG 系统评估结果归档。每次架构变更、参数调优后在此目录下创建新的基线快照。

## 目录结构

```
benchmarks/
├── README.md                           # 本文件
└── YYYY-MM-DD_<描述>/                  # 每次评估一个目录
    ├── ARCHITECTURE.md                 # 当时架构快照 + 评估结果
    └── results.json                    # 详细指标数据（可机器读取）
```

## 基线记录

| 日期 | 描述 | 关键指标 |
|------|------|----------|
| [2026-05-27](./2026-05-27_hyde/README.md) | HyDE (假设文档嵌入) | hybrid MRR=0.612 (+27%), Answer Relevance=0.607 ⭐ — MRR 首次破0.6，排序质量新高 |
| [2026-05-27](./2026-05-27_sentence_window/README.md) | Sentence Window (Small-to-Big) | hybrid Recall@5=0.361 ⚠️ — 全面退化，单句嵌入不适用于领域知识检索 |
| [2026-05-25](./2026-05-25_topk30/README.md) | 扩大检索网 (top_k 8→30) | hybrid MRR=0.483 (+11%), Hit@8=0.800 — Recall 持平 |
| [2026-05-23](./2026-05-23_hierarchical/README.md) | 三层层级索引 | vector MRR=0.494 (+50%), Answer Relevance=0.573 (+9%) |
| [2026-05-22](./2026-05-22_baseline/ARCHITECTURE.md) | 首次基线评估 (flat chunks) | hybrid Recall@5=0.564, MRR=0.383, Hit@5=0.533 |

## 如何使用

每次重大改动后（如更换 chunk 策略、调整检索参数、更换 Embedding 模型），运行评估并创建新快照：

```bash
# 1. 创建归档目录
mkdir -p benchmarks/YYYY-MM-DD_<简短描述>

# 2. 运行评估（详情自动保存为 benchmarks/details_<时间戳>.json）
python main.py evaluate --by-category \
    --output benchmarks/YYYY-MM-DD_<简短描述>/results.json

# 指定详情文件路径：
python main.py evaluate --save-details benchmarks/YYYY-MM-DD_<描述>/details.json

# 不保存详情：
python main.py evaluate --no-save-details

# 3. 编写 ARCHITECTURE.md（参考 baseline 模板）
#    记录变更内容 + 评估结果对比

# 4. 更新本 README 的基线记录表
```
