"""
生成质量评估 —— 基于 LLM-as-Judge 直接调用。

评估两个维度：
- Faithfulness（忠实度）：答案中的声明是否被检索到的上下文所支撑
- Answer Relevance（答案相关性）：答案是否直接回应了用户的问题

使用 LangChain ChatOpenAI LLM 直接发 prompt，不依赖 langchain.evaluation 模块。
"""

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .. import logger as log


# --------------- Faithfulness 评估 prompt ---------------

FAITHFULNESS_SYSTEM = """你是一个严谨的文档一致性评估专家。你的任务是判断一个回答是否忠实于提供的上下文。

评估规则：
1. 将回答拆解为独立的声明（statement）
2. 逐条检查每个声明是否能从上下文中推断出来
3. 如果所有声明都能从上下文中推断，则为完全忠实
4. 如果有声明与上下文矛盾或上下文中未提及，则扣分

请给出 0-10 的分数，并简要说明理由。
- 10分：所有声明都完全能从上下文中推断
- 7-9分：绝大多数声明能从上下文中推断，只有极少数细节无法确认
- 4-6分：部分声明能从上下文中推断，但存在较多无法确认或矛盾的信息
- 1-3分：只有极少数声明能从上下文中推断
- 0分：完全无法从上下文中推断，或与上下文完全矛盾

输出格式（严格遵循）：
分数: <整数>
理由: <一句话说明>"""

FAITHFULNESS_USER = """请评估以下回答是否忠实于上下文。

用户问题: {question}

上下文:
{context}

回答:
{answer}

请给出评分和理由："""


# --------------- Answer Relevance 评估 prompt ---------------

RELEVANCE_SYSTEM = """你是一个严谨的问答质量评估专家。你的任务是判断一个回答是否直接回应了用户的问题。

评估规则：
1. 回答是否完整覆盖了问题的所有关键方面
2. 回答是否偏离了问题主题
3. 回答是否包含与问题无关的内容
4. **重要**：如果回答指出"文档中未找到相关信息"或类似表述，且该问题确实超出了给定文档的范畴（如技术问题、时事新闻、文档未覆盖的学科知识等），则说明回答正确识别了信息缺失，应给予 8-10 分。只有当文档中明明有相关信息却被忽略时，才应扣分。

请给出 0-10 的分数，并简要说明理由。
- 10分：完美回答，直接、完整、准确地回应了问题
- 7-9分：较好地回应了问题，可能有少量细节不完整
- 4-6分：部分回应了问题，但遗漏了重要方面或包含较多无关内容
- 1-3分：回答与问题关联很弱，大部分不相关
- 0分：完全答非所问

输出格式（严格遵循）：
分数: <整数>
理由: <一句话说明>"""

RELEVANCE_USER = """请评估以下回答是否切题。

用户问题: {question}

回答:
{answer}

请给出评分和理由："""


def _parse_score_response(text: str) -> dict:
    """
    解析 LLM 返回的评分结果。

    支持格式：
    - "分数: 8" / "分数：8"
    - "分数: -5"（负数会被 clamp 到 0）
    - 行中间出现 "分数: X" 也能识别

    Args:
        text: LLM 返回文本

    Returns:
        {"score": float (0-10), "reasoning": str}
    """
    import re

    score = 0.0
    reasoning = ""
    score_found = False

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()

        # 匹配分数行：在行中查找 "分数" 后跟冒号和数字（可能带负号）
        if not score_found:
            m = re.search(r"分数\s*[：:]\s*(-?\d+)", line)
            if m:
                try:
                    score = float(m.group(1))
                    score_found = True
                except ValueError:
                    pass

        # 匹配理由行
        if not reasoning:
            if line.startswith("理由:") or line.startswith("理由："):
                reasoning = line.split(":", 1)[-1].split("：", 1)[-1].strip()

    if not reasoning:
        reasoning = text.strip()[:500]

    return {"score": min(10.0, max(0.0, score)), "reasoning": reasoning[:500]}


def evaluate_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    llm: BaseChatModel,
) -> dict:
    """
    评估生成的回答是否忠实于检索到的上下文。

    Args:
        question: 用户问题
        answer: 生成的回答
        contexts: 检索到的上下文文本列表
        llm: 用于评估的 LLM (ChatOpenAI 等)

    Returns:
        dict: {score: float (0-1), reasoning: str}
    """
    if not answer or not contexts:
        return {"score": 0.0, "reasoning": "答案或上下文为空"}

    context_text = "\n\n---\n\n".join(
        f"[来源{i+1}]: {ctx[:800]}" for i, ctx in enumerate(contexts[:5])
    )

    messages = [
        SystemMessage(content=FAITHFULNESS_SYSTEM),
        HumanMessage(content=FAITHFULNESS_USER.format(
            question=question,
            context=context_text,
            answer=answer[:2000],
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = _parse_score_response(response.content)
        return {
            "score": min(1.0, max(0.0, result["score"] / 10.0)),
            "reasoning": result["reasoning"],
        }
    except Exception as e:
        log.info(f"  [Eval] Faithfulness 评估失败: {e}")
        return {"score": 0.0, "reasoning": f"评估出错: {str(e)[:200]}"}


def evaluate_answer_relevance(
    question: str,
    answer: str,
    llm: BaseChatModel,
) -> dict:
    """
    评估生成的回答是否直接回应了用户的问题。

    Args:
        question: 用户问题
        answer: 生成的回答
        llm: 用于评估的 LLM

    Returns:
        dict: {score: float (0-1), reasoning: str}
    """
    if not answer:
        return {"score": 0.0, "reasoning": "答案为空"}

    messages = [
        SystemMessage(content=RELEVANCE_SYSTEM),
        HumanMessage(content=RELEVANCE_USER.format(
            question=question,
            answer=answer[:2000],
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = _parse_score_response(response.content)
        return {
            "score": min(1.0, max(0.0, result["score"] / 10.0)),
            "reasoning": result["reasoning"],
        }
    except Exception as e:
        log.info(f"  [Eval] Answer Relevance 评估失败: {e}")
        return {"score": 0.0, "reasoning": f"评估出错: {str(e)[:200]}"}


class GenerationEvaluator:
    """
    生成质量评估器 —— 对 RAG 生成的答案进行多维度评估。
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> dict:
        """
        全面评估生成的答案质量。

        Returns:
            dict: {
                "faithfulness": float (0-1),
                "faithfulness_reasoning": str,
                "answer_relevance": float (0-1),
                "relevance_reasoning": str,
            }
        """
        faith_result = evaluate_faithfulness(question, answer, contexts, self.llm)
        relevance_result = evaluate_answer_relevance(question, answer, self.llm)

        return {
            "faithfulness": faith_result["score"],
            "faithfulness_reasoning": faith_result["reasoning"],
            "answer_relevance": relevance_result["score"],
            "relevance_reasoning": relevance_result["reasoning"],
        }
