import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

from .document import load_documents, chunk_documents, save_chunks_to_json, ChunkingConfig
from .embeddings import create_embeddings
from .vectorstore import create_vectorstore, add_documents, load_vectorstore, get_all_documents
from .retriever import (
    retrieve, format_context, expand_context, RetrievalConfig, BM25Retriever,
)
from .generator import create_llm, generate
from .hierarchy import (
    build_hierarchy, save_hierarchy, HierarchyIndex,
    flatten_l1_l2_texts, is_macro_query,
)
from .paragraph_chunker import paragraph_chunk_documents
from . import logger


def _create_agent_pipeline(db_dir, config, agent_config):
    """延迟导入并构建 AgentRAGPipeline，避免 langgraph 未安装时影响基础功能。"""
    from .agent import build_agent_graph
    from .agent_config import AgentConfig

    cfg = agent_config or AgentConfig()
    return AgentRAGPipeline(db_dir=db_dir, config=config, agent_config=cfg)


DB_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
DOCS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


@dataclass
class RAGConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 80
    chunk_strategy: str = "paragraph"   # "paragraph" | "sentence" | "recursive"
    top_k: int = 30
    collection_name: str = "rag_agent"

    retrieval_mode: str = "hybrid"
    enable_rerank: bool = True
    rerank_top_k: int = 8
    enable_query_rewrite: bool = True
    enable_hyde: bool = True           # HyDE: 生成假设文档后再检索
    context_window: int = 0            # 前后各扩展 N 段 (默认关闭)


class RAGPipeline:
    def __init__(
        self,
        db_dir: str | None = None,
        config: RAGConfig | None = None,
    ):
        self.db_dir = db_dir or DB_DIR_DEFAULT
        self.config = config or RAGConfig()
        self._embeddings = None
        self._llm = None
        self._bm25: BM25Retriever | None = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = create_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def _get_bm25(self, vectorstore) -> BM25Retriever:
        if self._bm25 is not None:
            return self._bm25

        bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            logger.info("[BM25] 加载已有索引")
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
                if self._bm25._documents:
                    logger.info(f"[BM25] 索引已加载: {len(self._bm25._documents)} 文档")
            return self._bm25

        logger.info("[BM25] 未找到索引文件，正在从向量库构建 BM25 索引 ...")
        all_docs = get_all_documents(vectorstore)
        if not all_docs:
            logger.info("[BM25] 向量库为空，跳过 BM25 构建")
            self._bm25 = BM25Retriever()
            return self._bm25

        self._bm25 = BM25Retriever()
        self._bm25.index(all_docs)
        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)
        logger.info(f"[BM25] 索引构建完成并保存: {len(all_docs)} 文档")
        return self._bm25

    def ingest(self, docs_dir: str | None = None) -> int:
        logger.section("文档导入 (Ingestion)")
        source = docs_dir or DOCS_DIR_DEFAULT
        logger.keyval("文档目录", source)
        logger.keyval("向量库目录", self.db_dir)
        logger.keyval("Chunk 策略", self.config.chunk_strategy)
        logger.keyval("Chunk 大小", str(self.config.chunk_size))

        docs = load_documents(source)
        if not docs:
            logger.info("未找到文档，跳过导入")
            return 0

        logger.info(f"共加载 {len(docs)} 个文档")
        print(f"共加载 {len(docs)} 个文档")

        # ========== 切分策略选择 ==========
        if self.config.chunk_strategy == "paragraph":
            chunks = paragraph_chunk_documents(
                docs,
                target_size=self.config.chunk_size,
                max_size=self.config.chunk_size * 2,
            )
            chunk_cfg = ChunkingConfig(
                chunk_size=self.config.chunk_size,
                strategy="paragraph",
            )
        else:
            chunk_cfg = ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=self.config.chunk_strategy,
            )
            chunks = chunk_documents(docs, chunk_cfg)

        save_chunks_to_json(chunks, config=chunk_cfg)

        # ========== 层级索引构建 ==========
        chunks_by_source: dict[str, list] = defaultdict(list)
        for i, ch in enumerate(chunks):
            src = ch.metadata.get("source", "unknown")
            if "chunk_id" not in ch.metadata:
                ch.metadata["chunk_id"] = i
            chunks_by_source[src].append(ch)

        hierarchy = HierarchyIndex()
        summary_llm = create_llm(temperature=0.1)
        for src, src_chunks in chunks_by_source.items():
            src_name = os.path.basename(src)
            node = build_hierarchy(src_chunks, summary_llm, src_name)
            hierarchy.documents.append(node)

        save_hierarchy(hierarchy)

        summary_texts = flatten_l1_l2_texts(hierarchy)
        from langchain_core.documents import Document as LangchainDocument
        summary_docs_lc = [
            LangchainDocument(page_content=t, metadata={"source": "hierarchy", "level": "summary"})
            for t in summary_texts
        ]
        logger.info(f"[Hierarchy] {len(summary_docs_lc)} 条摘要待写入向量库")
        # ========== 层级索引构建结束 ==========

        embeddings = create_embeddings()
        vs = create_vectorstore(
            self.db_dir,
            embeddings,
            self.config.collection_name,
        )
        add_documents(vs, chunks)

        if summary_docs_lc:
            add_documents(vs, summary_docs_lc)
            logger.info(f"[Hierarchy] {len(summary_docs_lc)} 条摘要已写入向量库")

        bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            os.remove(bm25_path)
            logger.info("[BM25] 旧索引已删除，将在下次查询时重建")

        logger.info(f"导入完成: {len(chunks)} 个 chunks -> {self.db_dir}")
        return len(chunks)

    def query(self, question: str) -> dict:
        logger.section(f"RAG 问答: {question}")

        embeddings = self._get_embeddings()
        vs = load_vectorstore(
            self.db_dir,
            embeddings,
            self.config.collection_name,
        )

        bm25 = None
        if self.config.retrieval_mode in ("bm25", "hybrid"):
            bm25 = self._get_bm25(vs)

        # ===== 层级检索路由 =====
        if is_macro_query(question):
            logger.info("[Hierarchy] 检测到宏观问题，启用层级检索")
            try:
                summary_docs = vs.similarity_search(
                    question, k=6,
                    filter={"source": "hierarchy"},
                )
                logger.info(f"[Hierarchy]   摘要检索命中 {len(summary_docs)} 条")
            except Exception:
                all_vs_docs = vs.similarity_search(question, k=12)
                summary_docs = [d for d in all_vs_docs
                                if d.metadata.get("source") == "hierarchy"][:6]
                logger.info(f"[Hierarchy]   摘要检索（无过滤回退）命中 {len(summary_docs)} 条")

            detail_docs = vs.similarity_search(question, k=6)
            seen = set()
            docs = []
            for d in summary_docs + detail_docs:
                key = d.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
            docs = docs[:12]
            logger.info(f"[Hierarchy]   融合后共 {len(docs)} 条")
        else:
            retrieval_cfg = RetrievalConfig(
                top_k=self.config.top_k,
                retrieval_mode=self.config.retrieval_mode,
                enable_rerank=self.config.enable_rerank,
                rerank_top_k=self.config.rerank_top_k,
                enable_query_rewrite=self.config.enable_query_rewrite,
                enable_hyde=self.config.enable_hyde,
            )
            docs = retrieve(
                query=question,
                vectorstore=vs,
                llm=self.llm,
                bm25=bm25,
                config=retrieval_cfg,
            )
        # ===== 层级检索路由结束 =====

        # 上下文扩展：命中段落后拉取前后相邻段落
        if self.config.context_window > 0:
            docs = expand_context(docs, vs, window=self.config.context_window)

        context = format_context(docs)
        answer = generate(self.llm, question, context)

        logger.section("问答结果")
        logger.info(f"  {answer}")

        return {
            "question": question,
            "answer": answer,
            "sources": list(set(d.metadata.get("source", "unknown") for d in docs)),
            "chunks": docs,
        }


class AgentRAGPipeline:
    """基于 LangGraph 的多步 Agent RAG 管线。

    包装现有 RAGPipeline 组件，通过 LangGraph StateGraph 编排多步推理。
    返回格式与 RAGPipeline.query() 兼容。
    """

    def __init__(
        self,
        db_dir: str | None = None,
        config: RAGConfig | None = None,
        agent_config=None,
        lazy: bool = False,
    ):
        self.db_dir = db_dir or DB_DIR_DEFAULT
        self.config = config or RAGConfig()
        self._agent_config = agent_config
        self._base = RAGPipeline(db_dir=self.db_dir, config=self.config)
        self._graph = None

        if not lazy:
            self._ensure_graph()

    @property
    def llm(self):
        return self._base.llm

    def _ensure_graph(self):
        if self._graph is not None:
            return

        from .agent import build_agent_graph
        from .agent_config import AgentConfig

        cfg = self._agent_config or AgentConfig()

        embeddings = self._base._get_embeddings()
        vs = load_vectorstore(
            self.db_dir, embeddings, self.config.collection_name,
        )
        bm25 = None
        if self.config.retrieval_mode in ("bm25", "hybrid"):
            bm25 = self._base._get_bm25(vs)

        self._graph = build_agent_graph(
            vectorstore=vs,
            llm=self.llm,
            bm25=bm25,
            embeddings=embeddings,
            rag_config=self.config,
            agent_config=cfg,
        )

    def query(self, question: str) -> dict:
        logger.section(f"Agent 问答: {question}")

        self._ensure_graph()
        final_state = self._graph.invoke({"question": question})

        docs = final_state.get("documents", [])
        answer = final_state.get("answer", "")
        metadata = final_state.get("metadata", {})

        logger.section("问答结果")
        logger.info(f"  {answer}")
        if metadata:
            logger.info(f"  [元数据] 查询类型={metadata.get('query_type')}, "
                         f"置信度={metadata.get('confidence', 'N/A')}")

        return {
            "question": question,
            "answer": answer,
            "sources": list(set(d.metadata.get("source", "unknown") for d in docs)),
            "chunks": docs,
            "metadata": metadata,
        }
