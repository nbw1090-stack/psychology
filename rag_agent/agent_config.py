"""
Agent 配置与查询分类。

- AgentConfig: Agent 特性的开关与阈值
- classify_query: 启发式查询复杂度分类 (0 LLM 调用)
"""

import re
from dataclasses import dataclass
from typing import Literal

from .hierarchy import is_macro_query

# 复杂查询正则模式 (对比/多跳/因果/并列)
_COMPLEX_PATTERNS = [
    re.compile(p) for p in [
        r"[对比比较].*[和与及]",
        r"[区别差异不同].*[和与及]",
        r"[原因为什么].*[关联关系影响]",
        r".*[以及和].*如何",
        r"(?:同时|此外|另外|而且|并且)",
        r"(?:哪些|分别|各自)",
        r"(?:如何).*(?:以及|和|与)",
    ]
]


@dataclass
class AgentConfig:
    """Agent 特性配置。"""

    # P0: 两阶段生成 (Claim Verification)
    enable_claim_verification: bool = True

    # P1: 查询分解
    enable_query_decomposition: bool = True
    max_sub_queries: int = 3

    # P2: 置信度感知
    enable_confidence_scoring: bool = True
    confidence_high_threshold: float = 0.70
    confidence_low_threshold: float = 0.40

    # 全局开关: 关闭后降级为简单 pipeline
    disable_agent: bool = False


def classify_query(question: str) -> Literal["simple", "complex", "macro"]:
    """启发式查询复杂度分类。

    优先级: macro > complex > simple
    """
    # 1. 宏观问题 (复用 hierarchy 的关键词列表)
    if is_macro_query(question):
        return "macro"

    # 2. 复杂问题 (正则匹配)
    for pattern in _COMPLEX_PATTERNS:
        if pattern.search(question):
            return "complex"

    # 3. 长问题启发式 (>50 字多半是多部分问题)
    if len(question) > 50:
        return "complex"

    # 4. 多问号
    if question.count("？") > 1 or question.count("?") > 1:
        return "complex"

    return "simple"
