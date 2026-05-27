"""
层级索引模块：构建 文档级 → 章节级 → 段落级 三层索引。

- L1 文档级：全书摘要 + 章节列表
- L2 章节级：每章标题 + 摘要 + 关键词
- L3 段落级：原始 chunks（不变，由现有 document.py 管理）

检索路由策略：
- 宏观问题（总结/概括/全书）→ 优先 L1/L2 摘要，补充 L3 段落
- 微观问题（概念/事实）→ 直接 L3 段落检索（现有行为）
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import logger

HIERARCHY_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "chunks"
)


# ====================== 数据结构 ======================

@dataclass
class Chapter:
    """章节节点"""
    title: str                              # 章节标题
    level: int = 2                          # 1=文档级, 2=章节级
    start_chunk_id: int = 0
    end_chunk_id: int = 0
    chunk_ids: List[int] = field(default_factory=list)
    summary: str = ""                       # LLM 生成的摘要
    keywords: List[str] = field(default_factory=list)


@dataclass
class DocumentNode:
    """文档节点"""
    source: str                             # PDF 文件名
    title: str = ""
    summary: str = ""                       # 全书摘要
    chapters: List[Chapter] = field(default_factory=list)
    total_chunks: int = 0


@dataclass
class HierarchyIndex:
    """完整层级索引"""
    documents: List[DocumentNode] = field(default_factory=list)
    generated_at: str = ""


# ====================== 章节检测 ======================

def _is_toc_chunk(content: str) -> bool:
    """判断一个 chunk 是否包含目录"""
    head = content[:600]
    return any(m in head for m in ["目录", "目  录", "目 录"])


def _parse_toc_for_chapters(toc_content: str) -> List[str]:
    """从目录 chunk 中提取章节标题列表"""
    chapters: List[str] = []
    pattern = re.compile(
        r"(第[一二三四五六七八九十百零\d]+[章卷编篇])"
        r"[　\s]*[：:]*\s*(.+?)(?=第[一二三四五六七八九十百零\d]+[章卷编篇]|$)"
    )
    for prefix, title in pattern.findall(toc_content):
        chapters.append(f"{prefix}　{title.strip()}")
    return chapters


def _detect_chapters_by_heading(
    chunks: List[Document], source_name: str
) -> List[Chapter]:
    """通用章节检测：通过正则匹配章节标题"""
    chapter_pattern = re.compile(
        r"(第[一二三四五六七八九十百零\d]+[章卷编篇])[　\s]*[：:]*\s*(.+)"
    )
    special_titles = [
        ("作者前言", "前言"),
        ("前言", "前言"),
        ("导言：", "导言"),
        ("导言", "导言"),
        ("后记", "后记"),
        ("译者后记", "译者后记"),
    ]

    detected: List[Chapter] = []
    current_title: Optional[str] = None
    current_start: int = 0
    current_ids: List[int] = []

    for i, chunk in enumerate(chunks):
        content = chunk.page_content[:300]

        # 匹配 "第X章" / "第X卷" 等
        m = chapter_pattern.search(content)
        if m:
            title = m.group(0).split("\n")[0].strip()
            if current_title is not None:
                detected.append(Chapter(
                    title=current_title, level=2,
                    start_chunk_id=current_start, end_chunk_id=i - 1,
                    chunk_ids=list(current_ids),
                ))
            current_title = title
            current_start = i
            current_ids = [i]
            continue

        # 匹配特殊标题（前言、导言、后记等）
        found = False
        for st_pattern, st_label in special_titles:
            if content.strip().startswith(st_pattern) and len(content.strip()) < 200:
                if current_title is not None:
                    detected.append(Chapter(
                        title=current_title, level=2,
                        start_chunk_id=current_start, end_chunk_id=i - 1,
                        chunk_ids=list(current_ids),
                    ))
                current_title = st_label
                current_start = i
                current_ids = [i]
                found = True
                break
        if found:
            continue

        # 乌合之众风格：第X页提要
        if "提要" in content[:100]:
            sm = re.search(r"(第\d+页)?\s*提要[：:]\s*(.+)", content)
            if sm:
                title = f"提要：{sm.group(2).split('/')[0].strip()}"
                if current_title is not None:
                    detected.append(Chapter(
                        title=current_title, level=2,
                        start_chunk_id=current_start, end_chunk_id=i - 1,
                        chunk_ids=list(current_ids),
                    ))
                current_title = title
                current_start = i
                current_ids = [i]
                continue

        if current_title is not None:
            current_ids.append(i)

    # 尾部章节
    if current_title is not None:
        detected.append(Chapter(
            title=current_title, level=2,
            start_chunk_id=current_start, end_chunk_id=len(chunks) - 1,
            chunk_ids=list(current_ids),
        ))

    return detected


def detect_chapters(
    chunks: List[Document], source_name: str
) -> List[Chapter]:
    """检测文档的章节边界"""
    logger.info(f"[Hierarchy] 检测章节: {source_name}")

    # 1. 查找目录 chunk
    toc_chapters: List[str] = []
    for chunk in chunks:
        if _is_toc_chunk(chunk.page_content[:600]):
            toc_chapters = _parse_toc_for_chapters(chunk.page_content)
            if toc_chapters:
                logger.info(f"[Hierarchy]   目录提取到 {len(toc_chapters)} 个标题")
                break

    # 2. 扫描检测
    chapters = _detect_chapters_by_heading(chunks, source_name)

    # 3. 用目录信息补充标题
    if toc_chapters and len(toc_chapters) > len(chapters):
        for i, ch in enumerate(chapters):
            if i < len(toc_chapters):
                ch.title = toc_chapters[i]

    # 4. 合并过小的章节（< 3 chunks）
    merged: List[Chapter] = []
    for ch in chapters:
        if merged and len(ch.chunk_ids) < 3:
            prev = merged[-1]
            prev.end_chunk_id = ch.end_chunk_id
            prev.chunk_ids.extend(ch.chunk_ids)
            prev.title = f"{prev.title} + {ch.title}"
        else:
            merged.append(ch)

    logger.info(f"[Hierarchy]   检测到 {len(merged)} 个章节")
    for ch in merged[:5]:
        logger.info(f"[Hierarchy]     {ch.title}  ({len(ch.chunk_ids)} chunks)")
    if len(merged) > 5:
        logger.info(f"[Hierarchy]     ... 还有 {len(merged) - 5} 个")

    return merged


# ====================== 摘要生成 ======================

SUMMARY_SYSTEM = """你是一个学术文档摘要专家。请根据提供的文本内容生成中文摘要。

规则：
- 摘要长度控制在100-200字
- 提取核心论点和关键概念
- 对于章节摘要，突出该章在全书中的位置和作用
- 只输出摘要文本，不要包含"摘要："等前缀"""


def generate_summary(llm: BaseChatModel, text: str, context: str = "") -> str:
    """使用 LLM 生成文本摘要"""
    prompt = f"请为以下内容生成摘要：\n\n{text}"
    if context:
        prompt = f"背景：{context}\n\n{prompt}"

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=prompt),
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.info(f"[Hierarchy] 摘要生成失败: {e}")
        return text[:200].replace("\n", " ") + "..."


def _extract_keywords(text: str, max_kw: int = 5) -> List[str]:
    """使用 jieba 提取关键词"""
    try:
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=max_kw)
    except Exception:
        return []


# ====================== 层级索引构建 ======================

def build_hierarchy(
    chunks: List[Document],
    llm: BaseChatModel,
    source_name: str,
    skip_summaries: bool = False,
) -> DocumentNode:
    """为一个文档构建三层索引"""
    logger.section(f"[Hierarchy] 构建层级索引: {source_name}")

    # Step 1: 检测章节
    chapters = detect_chapters(chunks, source_name)

    # Step 2: 生成章节摘要 (L2) + 关键词
    all_summaries: List[str] = []
    for ch in chapters:
        chapter_text = "\n".join(
            chunks[cid].page_content for cid in ch.chunk_ids
        )[:3000]

        if not skip_summaries:
            logger.info(f"[Hierarchy]   生成摘要: {ch.title}")
            ch.summary = generate_summary(llm, chapter_text)
        else:
            ch.summary = chapter_text[:200].replace("\n", " ") + "..."

        ch.keywords = _extract_keywords(chapter_text)
        all_summaries.append(f"## {ch.title}\n{ch.summary}")

    # Step 3: 生成文档级摘要 (L1)
    doc_summary_text = "\n\n".join(all_summaries)
    if not skip_summaries:
        logger.info(f"[Hierarchy]   生成全书摘要")
        doc_summary = generate_summary(
            llm, doc_summary_text,
            context=f"这是《{source_name.replace('.pdf', '')}》的章节摘要汇总。",
        )
    else:
        doc_summary = doc_summary_text[:500]

    node = DocumentNode(
        source=source_name,
        title=source_name.replace(".pdf", ""),
        summary=doc_summary,
        chapters=chapters,
        total_chunks=len(chunks),
    )
    logger.info(f"[Hierarchy]   全书摘要: {doc_summary[:120]}...")
    return node


# ====================== 序列化 ======================

def _chapter_to_dict(ch: Chapter) -> dict:
    return {
        "title": ch.title,
        "level": ch.level,
        "start_chunk_id": ch.start_chunk_id,
        "end_chunk_id": ch.end_chunk_id,
        "chunk_count": len(ch.chunk_ids),
        "summary": ch.summary,
        "keywords": ch.keywords,
    }


def _dict_to_chapter(d: dict) -> Chapter:
    return Chapter(
        title=d["title"],
        level=d["level"],
        start_chunk_id=d["start_chunk_id"],
        end_chunk_id=d["end_chunk_id"],
        chunk_ids=list(range(d["start_chunk_id"], d["end_chunk_id"] + 1)),
        summary=d.get("summary", ""),
        keywords=d.get("keywords", []),
    )


def save_hierarchy(
    hierarchy: HierarchyIndex, output_dir: str | None = None
) -> str:
    """保存层级索引到 JSON 文件"""
    out_dir = output_dir or HIERARCHY_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"hierarchy_{timestamp}.json")

    data = {
        "generated_at": timestamp,
        "documents": [
            {
                "source": doc.source,
                "title": doc.title,
                "summary": doc.summary,
                "total_chunks": doc.total_chunks,
                "chapter_count": len(doc.chapters),
                "chapters": [_chapter_to_dict(ch) for ch in doc.chapters],
            }
            for doc in hierarchy.documents
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"[Hierarchy] 层级索引已保存: {filepath}")
    print(f"[Hierarchy] 层级索引已保存: {filepath}")
    return filepath


def load_hierarchy(filepath: str) -> HierarchyIndex:
    """从 JSON 文件加载层级索引"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    hierarchy = HierarchyIndex(generated_at=data.get("generated_at", ""))
    for doc_data in data.get("documents", []):
        hierarchy.documents.append(DocumentNode(
            source=doc_data["source"],
            title=doc_data.get("title", ""),
            summary=doc_data.get("summary", ""),
            total_chunks=doc_data.get("total_chunks", 0),
            chapters=[_dict_to_chapter(ch)
                      for ch in doc_data.get("chapters", [])],
        ))
    return hierarchy


# ====================== 检索辅助 ======================

# 宏观问题关键词（触发层级检索）
MACRO_KEYWORDS = [
    "总结", "概括", "本书", "全书", "核心观点", "讲了什么",
    "主要内容", "框架", "脉络", "整体", "概述", "全书结构",
    "核心论点", "这本书", "解读", "主题思想",
]


def is_macro_query(question: str) -> bool:
    """判断是否为宏观问题"""
    return any(kw in question for kw in MACRO_KEYWORDS)


# ====================== 检索辅助 ======================

def flatten_l1_l2_texts(
    hierarchy: HierarchyIndex, source_filter: str | None = None
) -> List[str]:
    """将 L1/L2 摘要展平为文本列表，用于向量化"""
    texts: List[str] = []
    for doc in hierarchy.documents:
        if source_filter and doc.source != source_filter:
            continue
        texts.append(f"[{doc.title}] 全书摘要: {doc.summary}")
        for ch in doc.chapters:
            texts.append(f"[{doc.title}] {ch.title}: {ch.summary}")
    return texts
