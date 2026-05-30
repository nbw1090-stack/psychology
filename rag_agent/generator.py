import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from . import logger


def create_llm(
    temperature: float = 0.3,
) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        temperature=temperature,
    )
    logger.info(f"[LLM] 模型: deepseek-v4-flash, temperature={temperature}")
    return llm


SYSTEM_PROMPT = """你是一个基于文档的问答助手。请严格根据以下提供的上下文来回答用户的问题。

规则:
- 只根据「核心文档」（不带"补充"标记的文档）来回答问题。补充上下文仅供参考背景，不要作为主要依据。
- 如果核心文档中没有足够信息来回答问题，请明确说"提供的文档中未找到相关信息"，不要编造内容。
- 不要引用不存在的文档编号，也不要编造文档中没有的数据、术语或标准。
- 回答时请引用具体的来源文档。
- 保持回答简洁准确。"""


def generate(
    llm: ChatOpenAI,
    query: str,
    context: str,
) -> str:
    logger.sub("生成阶段 (Generation)")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"上下文:\n{context}\n\n问题: {query}"),
    ]

    if logger.is_verbose():
        logger.content_block("System Prompt", SYSTEM_PROMPT)
        logger.content_block("User Prompt (含上下文 + 问题)",
                             f"查询: {query}\n上下文长度: {len(context)} 字符")
        logger.info("  [LLM] 开始调用大模型生成回答 ...")

    response = llm.invoke(messages)

    if logger.is_verbose():
        logger.content_block("LLM 返回", response.content)
        token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        if token_usage:
            logger.info(f"  [LLM] Token 用量: input={token_usage.get('prompt_tokens')}, "
                        f"output={token_usage.get('completion_tokens')}, "
                        f"total={token_usage.get('total_tokens')}")

    return response.content
