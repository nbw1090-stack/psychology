import os
from typing import List

from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

from . import logger

DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"


class DashScopeEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int = 1024,
        instruct: str | None = None,
    ):
        import dashscope
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        dashscope.api_key = api_key
        self._model = model or DASHSCOPE_EMBEDDING_MODEL
        self._dimension = dimension
        self._instruct = instruct or "Retrieve relevant paragraphs from psychology academic texts"
        logger.info(f"[Embeddings] 模型: {self._model}, dim={self._dimension}, instruct=enabled")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        if logger.is_verbose():
            previews = [t[:80] + "..." if len(t) > 80 else t for t in texts]
            logger.info(f"[Embeddings] 批量向量化 {len(texts)} 条文本 (text_type=document)")
            for i, p in enumerate(previews):
                logger.keyval(f"chunk[{i}]", p)
        resp = TextEmbedding.call(
            model=self._model,
            input=texts,
            text_type="document",
            dimension=self._dimension,
        )
        result = [item["embedding"] for item in resp.output["embeddings"]]
        if logger.is_verbose():
            usage = resp.usage or {}
            tokens = usage.get("total_tokens", "N/A") if isinstance(usage, dict) else "N/A"
            logger.info(f"[Embeddings] 返回 {len(result)} 个向量, dim={len(result[0])}, "
                        f"tokens={tokens}")
        return result

    def embed_query(self, text: str) -> List[float]:
        if logger.is_verbose():
            preview = text[:80] + "..." if len(text) > 80 else text
            logger.info(f"[Embeddings] 查询向量化 (text_type=query): \"{preview}\"")
        resp = TextEmbedding.call(
            model=self._model,
            input=[text],
            text_type="query",
            dimension=self._dimension,
            instruct=self._instruct,
        )
        return resp.output["embeddings"][0]["embedding"]


def create_embeddings(
    api_key: str | None = None,
    dimension: int = 1024,
    instruct: str | None = None,
) -> DashScopeEmbeddings:
    return DashScopeEmbeddings(api_key=api_key, dimension=dimension, instruct=instruct)
