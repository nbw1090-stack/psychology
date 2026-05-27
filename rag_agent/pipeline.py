import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

from .document import load_documents, chunk_documents, save_chunks_to_json, ChunkingConfig
from .embeddings import create_embeddings
from .vectorstore import create_vectorstore, add_documents, load_vectorstore, get_all_documents
from .retriever import (
    retrieve, format_context, RetrievalConfig, BM25Retriever,
)
from .generator import create_llm, generate
from .hierarchy import (
    build_hierarchy, save_hierarchy, HierarchyIndex,
    flatten_l1_l2_texts, is_macro_query,
)
from . import logger


DB_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
DOCS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_strategy: str = "sentence"
    top_k: int = 30
    collection_name: str = "rag_agent"

    retrieval_mode: str = "hybrid"
    enable_rerank: bool = True
    rerank_top_k: int = 8
    enable_query_rewrite: bool = True
    enable_hyde: bool = True           # HyDE: 生成假设文档后再检索


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
        logger.keyval("Chunk 大小", str(self.config.chunk_size))
        logger.keyval("Chunk 重叠", str(self.config.chunk_overlap))

        docs = load_documents(source)
        if not docs:
            logger.info("未找到文档，跳过导入")
            return 0

        logger.info(f"共加载 {len(docs)} 个文档")
        print(f"共加载 {len(docs)} 个文档")

        chunk_cfg = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=self.config.chunk_strategy,
        )
        chunks = chunk_documents(docs, chunk_cfg)

        # 本地化存储父 chunk 结果，方便调优
        save_chunks_to_json(chunks, config=chunk_cfg)

        # ========== 层级索引构建 ==========
        # 按 source 分组 chunks
        chunks_by_source: dict[str, list] = defaultdict(list)
        for i, ch in enumerate(chunks):
            src = ch.metadata.get("source", "unknown")
            # 注入 chunk_id 到 metadata，方便层级映射
            ch.metadata["chunk_id"] = i
            chunks_by_source[src].append(ch)

        hierarchy = HierarchyIndex()
        summary_llm = create_llm(temperature=0.1)
        for src, src_chunks in chunks_by_source.items():
            src_name = os.path.basename(src)
            node = build_hierarchy(src_chunks, summary_llm, src_name)
            hierarchy.documents.append(node)

        # 保存层级索引 JSON
        save_hierarchy(hierarchy)

        # 将 L1/L2 摘要向量化，写入向量库（标记 source="hierarchy"）
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

        # 写入层级摘要到向量库
        if summary_docs_lc:
            add_documents(vs, summary_docs_lc)
            logger.info(f"[Hierarchy] {len(summary_docs_lc)} 条摘要已写入向量库")

        # 删除旧的 BM25 索引，force rebuild on next query
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
            # 先检索 L1/L2 摘要（source="hierarchy"），需要额外调用
            try:
                summary_docs = vs.similarity_search(
                    question, k=6,
                    filter={"source": "hierarchy"},
                )
                logger.info(f"[Hierarchy]   摘要检索命中 {len(summary_docs)} 条")
            except Exception:
                # 如果不支持 filter，回退到无过滤检索
                all_vs_docs = vs.similarity_search(question, k=12)
                summary_docs = [d for d in all_vs_docs
                                if d.metadata.get("source") == "hierarchy"][:6]
                logger.info(f"[Hierarchy]   摘要检索（无过滤回退）命中 {len(summary_docs)} 条")

            # 补充 L3 段落
            detail_docs = vs.similarity_search(question, k=6)
            # 去重 + 合并
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
