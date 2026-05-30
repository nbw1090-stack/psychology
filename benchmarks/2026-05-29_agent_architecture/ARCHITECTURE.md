# Agent 层架构升级

**日期**: 2026-05-29
**变更类型**: 架构级 — 从单趟 RAG Pipeline 演进为 LangGraph 多步 Agent

---

## 变更动机

RAG 索引侧优化已达天花板（Recall@5=0.709, MRR=0.733, Faithfulness=0.873）。Faithfulness 87% 低于商用 95% 门槛，根因是单趟 pipeline 无法补救检索遗漏、无法拆解复杂查询、没有自我校验。

---

## 新架构

```
query_understand (1 LLM) ─── 意图识别 + 关键词抽取 + 查询优化
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
simple  macro      multi
retrieve retrieve   retrieve
    │    │            │
    └────┼────────────┘
         ▼
     generate (1 LLM) ─── 基于 context 的 grounded 生成
         ▼
   claim_verify (1 LLM) ─── 事实校验，过滤无据声明
         ▼
   confidence_score (0 LLM) ─── 纯启发式置信度评分
         ▼
        END
```

### 与旧架构对比

```
旧: query_router(规则) → [HyDE/rewrite] → retrieve → generate → END
新: query_understand(LLM) → retrieve(用理解后的关键词) → generate → claim_verify → confidence → END
```

### 关键变化

| 变化 | 旧 | 新 |
|------|----|----|
| 查询理解 | 启发式规则 classify_query() | LLM 1 次调用：意图 + 关键词 + 类型 |
| 查询增强 | HyDE 生成假设文档 | Query Understanding 产出结构化检索词 |
| 事实校验 | 无 | Claim Verification (P0) |
| 置信度输出 | 无 | 启发式评分 + 提示语 (P2) |
| 复杂查询路由 | decompose 独立节点 | 合并进 query_understand |

---

## 评估结果

### 基线 vs Agent（各 4 次运行）

#### Faithfulness（忠实度，目标 95%+）

| Run | 基线 (Pipeline) | Agent |
|-----|----------------|-------|
| 1 | 0.847 | 0.893 |
| 2 | 0.873 | **0.940** |
| 3 | 0.887 | 0.867 |
| 4 | 0.827 | 0.913 |
| **均值** | **0.858** | **0.903 (+5.2%)** |

#### Answer Relevance（答案相关性，目标 90%+）

| Run | 基线 (Pipeline) | Agent |
|-----|----------------|-------|
| 1 | 0.940 | 0.907 |
| 2 | 0.880 | 0.947 |
| 3 | 0.960 | 0.947 |
| 4 | 0.973 | 0.853 |
| **均值** | **0.938** | **0.913 (-2.7%)** |

### 逐用例对比（首次运行）

| 用例 | 基线 F | Agent F | 变化 | 基线 R | Agent R | 变化 |
|------|--------|---------|------|--------|---------|------|
| tc_fact_001 (GAD诊断) | 1.00 | 1.00 | = | 0.90 | 0.90 | = |
| tc_fact_002 (社交焦虑) | 0.80 | 1.00 | +0.20 | 1.00 | 1.00 | = |
| tc_fact_003 (群体特征) | 0.80 | 1.00 | +0.20 | 1.00 | 1.00 | = |
| tc_def_001 (习得性无助) | 1.00 | 1.00 | = | 0.90 | 0.90 | = |
| tc_def_002 (群体极化) | 1.00 | 1.00 | = | 0.80 | 1.00 | +0.20 |
| tc_def_003 (灾难化思维) | 1.00 | 1.00 | = | 1.00 | 1.00 | = |
| tc_cmp_001 (焦虑vs恐惧) | 0.20 | 1.00 | **+0.80** | 0.90 | 0.90 | = |
| tc_cmp_002 (精神分析vs行为) | 0.80 | 0.00 | -0.80 | 1.00 | 0.90 | -0.10 |
| tc_mh_001 (CBT技术) | 0.60 | 0.80 | +0.20 | 1.00 | 1.00 | = |
| tc_mh_002 (从众心理) | 0.70 | 1.00 | +0.30 | 1.00 | 0.50 | -0.50 |
| tc_sum_001 (乌合之众) | 1.00 | 1.00 | = | 0.90 | 1.00 | +0.10 |
| tc_sum_002 (焦虑症类型) | 0.80 | 0.60 | -0.20 | 1.00 | 1.00 | = |
| tc_neg_001 (马斯洛) | 1.00 | 1.00 | = | 0.90 | 0.80 | -0.10 |
| tc_neg_002 (Python) | 1.00 | 1.00 | = | 0.80 | 0.80 | = |
| tc_neg_003 (诺贝尔) | 1.00 | 1.00 | = | 1.00 | 0.90 | -0.10 |

---

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `rag_agent/agent_config.py` | 新建 | AgentConfig 配置 + classify_query() |
| `rag_agent/agent.py` | 新建 | LangGraph StateGraph + 7 个节点 |
| `rag_agent/pipeline.py` | 修改 | 新增 AgentRAGPipeline 包装类 |
| `rag_agent/cli.py` | 修改 | query/chat/evaluate 加 --agent flag |
| `rag_agent/retriever.py` | 修改 | jieba 延迟导入，抑制冗余日志 |

---

## 探索记录：补偿检索循环（已放弃）

尝试在 claim_verify 失败时自动补检索 + 重新生成。3 次评估结果：

- Faithfulness: 0.847, 0.940, 0.927 → 均值 0.905 (与 v1 持平)
- Relevance: 0.913, 0.833, 0.873 → 均值 0.873 (**-4.0%**)

放弃原因：补偿检索引入更多文档导致 LLM 偏题，Relevance 明显下降。在当前模型（DeepSeek-Flash）和文档库规模（3 本书）下收益不明显。

---

## 下一步方向

1. **升级生成模型** — DeepSeek-Flash 不稳定（run-to-run 方差 ±0.04），换更强模型是最大杠杆
2. **检索补偿作为可选功能** — 换模型后可重新测试
3. **Answer Relevance 守门** — 生成后检查回答是否覆盖问题所有部分
