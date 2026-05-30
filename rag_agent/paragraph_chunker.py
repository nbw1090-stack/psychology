"""
段落级切分模块：从 OCR/fitz 提取的页面文档重建段落级 chunks，构建链式 metadata。

流程:
    页面文档 → 按 source 分组 → 合并清洗为连续文本 → 检测章节边界
    → 章节内按句子重组段落 → 填充链式 metadata（book/chapter/prev/next）
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from . import logger
from .document import _split_sentences


# ---- 章节检测模式 ----

HEADING_RE = re.compile(r"第[一二三四五六七八九十百零\d]+[章卷编篇]")

SPECIAL_HEADINGS = [
    ("作者前言", "前言"),
    ("前言", "前言"),
    ("导言：", "导言"),
    ("导言", "导言"),
    ("后记", "后记"),
    ("译者后记", "译者后记"),
]

PAGE_MARKER_RE = re.compile(r"(?:^|\n)\s*[-—]*\s*第\s*\d+\s*页\s*[-—]*\s*")

# 截断章节标题时遇到这些标点就停止
TITLE_STOP_RE = re.compile(r"[。；，、\n]")


def _trim_chapter_title(raw: str) -> str:
    """截断过长的章节标题，保留核心部分。"""
    raw = raw.strip()
    m = TITLE_STOP_RE.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw[:40].strip()


@dataclass
class _ChapterSpan:
    title: str
    start: int  # char offset in full text
    end: int


# ---- 文本清洗 ----

def _clean_page_text(text: str) -> str:
    """清洗单页文本：移除页码标记，合并 OCR 换行。"""
    text = PAGE_MARKER_RE.sub("", text)
    lines = text.split("\n")
    merged: List[str] = []
    buf = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 如果 buf 末尾不是句末标点，且新行不是标点开头，则合并
        if buf and buf[-1] not in "。！？…~!?" and stripped[0] not in "，、；：":
            buf += stripped
        else:
            if buf:
                merged.append(buf)
            buf = stripped
    if buf:
        merged.append(buf)
    return "".join(merged)


# ---- 章节检测 ----

def _detect_chapters_in_text(text: str) -> List[_ChapterSpan]:
    """在连续文本中检测章节边界。

    过滤策略：
    1. 目录页密集标题：如果多个标题在 500 字内密集出现，跳过（是目录不是正文）
    2. 交叉引用：标题前有 "在"/"会"/"到"/"从"/"于" 等字词时跳过
    3. 极小章节：正文内容不足 500 字的章节与前一章合并
    """
    raw_markers: List[Tuple[int, str]] = []

    # 标准章节标题："第X章/卷..."
    for m in HEADING_RE.finditer(text):
        pos = m.start()

        # 过滤交叉引用：标题前有动词/介词则跳过
        if pos > 0:
            preceding = text[max(0, pos - 5):pos]
            if any(w in preceding for w in ["在", "会", "到", "从", "于", "和", "与",
                                             "及", "或", "中", "里", "前", "后",
                                             "见", "详见", "如", "同"]):
                continue

        heading = m.group()
        after = text[pos + len(heading):pos + len(heading) + 50]
        after = after.lstrip(" 　：:")
        raw_title = heading + after
        title = _trim_chapter_title(raw_title)
        raw_markers.append((pos, title))

    # 特殊标题（前言、导言、后记等）
    for st_pattern, st_label in SPECIAL_HEADINGS:
        idx = 0
        while True:
            idx = text.find(st_pattern, idx)
            if idx < 0:
                break
            if idx == 0 or text[idx - 1] in "。！？…~!?\n":
                raw_markers.append((idx, st_label))
            idx += len(st_pattern)

    raw_markers.sort(key=lambda x: x[0])

    # 去重：距离太近的标记只保留第一个
    deduped: List[Tuple[int, str]] = []
    for pos, title in raw_markers:
        if deduped and pos - deduped[-1][0] < 50:
            continue
        deduped.append((pos, title))

    if not deduped:
        return []

    spans: List[_ChapterSpan] = []
    for i, (pos, title) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        spans.append(_ChapterSpan(title=title, start=pos, end=end))

    # 合并过小的章节（< 1000 字）到前一章
    # 这样 TOC 中密集的小章节会自动合并到正文章节
    merged: List[_ChapterSpan] = []
    for span in spans:
        size = span.end - span.start
        if merged and size < 1000:
            merged[-1].end = span.end
        else:
            merged.append(_ChapterSpan(title=span.title, start=span.start, end=span.end))

    return merged


# ---- 段落重建 ----

def _split_into_paragraphs(
    text: str,
    target_size: int = 1200,
    max_size: int = 2000,
) -> List[str]:
    """将文本按句子边界重组为段落级 chunk。

    - 优先在 target_size 附近断句
    - 硬上限 max_size，超过则强制在句号处断开
    - 尾部过短的段落合并到前一段
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    paragraphs: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for s in sentences:
        if buf_len + len(s) > max_size and buf_len > 0:
            paragraphs.append("".join(buf))
            buf = []
            buf_len = 0

        buf.append(s)
        buf_len += len(s)

        if buf_len >= target_size:
            paragraphs.append("".join(buf))
            buf = []
            buf_len = 0

    if buf:
        trailing = "".join(buf)
        if paragraphs and len(trailing) < target_size // 3:
            paragraphs[-1] += trailing
        else:
            paragraphs.append(trailing)

    return [p for p in paragraphs if len(p.strip()) >= 20]


# ---- 页码估算 ----

def _estimate_page_for_offset(
    offset: int, page_boundaries: List[Tuple[int, int]]
) -> Optional[int]:
    """根据字符偏移估算页码。page_boundaries: [(start_offset, page_num), ...]"""
    for start_off, page_num in reversed(page_boundaries):
        if offset >= start_off:
            return page_num
    return page_boundaries[0][1] if page_boundaries else None


# ---- 主入口 ----

def paragraph_chunk_documents(
    documents: List[Document],
    target_size: int = 1200,
    max_size: int = 2000,
) -> List[Document]:
    """将页面级文档重建为段落级 chunks，附带链式 metadata。

    每个 chunk 包含:
        source, book_title, chapter_title, chapter_index,
        paragraph_index, chunk_id, prev_chunk_id, next_chunk_id, page
    """
    # 1. 按 source 分组
    by_source: dict[str, List[Document]] = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    all_chunks: List[Document] = []

    for source, pages in by_source.items():
        book_title = os.path.splitext(os.path.basename(source))[0]

        # 2. 按页码排序
        pages.sort(key=lambda d: d.metadata.get("page", 0))

        # 3. 合并清洗页面文本，记录页码边界
        page_boundaries: List[Tuple[int, int]] = []  # (char_offset, page_num)
        full_text = ""
        for p in pages:
            cleaned = _clean_page_text(p.page_content)
            if cleaned:
                page_boundaries.append((len(full_text), p.metadata.get("page")))
                full_text += cleaned

        if not full_text.strip():
            continue

        # 4. 检测章节
        chapters = _detect_chapters_in_text(full_text)

        # 没检测到章节 → 整个文档作为一章
        if not chapters:
            chapters = [_ChapterSpan(title=book_title, start=0, end=len(full_text))]

        # 第一个章节之前的文本归入"前言"
        if chapters[0].start > 50:
            chapters.insert(0, _ChapterSpan(title="前言", start=0, end=chapters[0].start))

        logger.info(f"[ParagraphChunker] {book_title}: 检测到 {len(chapters)} 个章节")
        for ch in chapters[:5]:
            logger.info(f"  {ch.title} ({ch.end - ch.start} 字)")
        if len(chapters) > 5:
            logger.info(f"  ... 还有 {len(chapters) - 5} 个")

        # 5. 每个章节内切分为段落
        source_chunks: List[Document] = []
        for ch_idx, chapter in enumerate(chapters):
            chapter_text = full_text[chapter.start:chapter.end].strip()
            if not chapter_text or len(chapter_text) < 20:
                continue

            paragraphs = _split_into_paragraphs(chapter_text, target_size, max_size)
            para_offset = chapter.start

            for para_idx, para_text in enumerate(paragraphs):
                page = _estimate_page_for_offset(para_offset, page_boundaries)
                chunk_id = f"{book_title}_{ch_idx}_{para_idx}"

                source_chunks.append(Document(
                    page_content=para_text,
                    metadata={
                        "source": source,
                        "book_title": book_title,
                        "chapter_title": chapter.title,
                        "chapter_index": ch_idx,
                        "paragraph_index": para_idx,
                        "chunk_id": chunk_id,
                        "prev_chunk_id": "",
                        "next_chunk_id": "",
                        "page": page,
                    },
                ))
                para_offset += len(para_text)

        # 6. 填充 prev/next 链
        for i in range(len(source_chunks)):
            if i > 0:
                source_chunks[i].metadata["prev_chunk_id"] = source_chunks[i - 1].metadata["chunk_id"]
            if i < len(source_chunks) - 1:
                source_chunks[i].metadata["next_chunk_id"] = source_chunks[i + 1].metadata["chunk_id"]

        logger.info(f"[ParagraphChunker] {book_title}: {len(source_chunks)} 段落")
        all_chunks.extend(source_chunks)

    logger.info(f"[ParagraphChunker] 总计: {len(all_chunks)} 段落")
    if logger.is_verbose():
        for i, chunk in enumerate(all_chunks[:5]):
            meta = chunk.metadata
            preview = chunk.page_content[:80].replace("\n", " ")
            logger.info(
                f"  [{i}] {meta['chapter_title']} > 第{meta['paragraph_index']}段 "
                f"({len(chunk.page_content)}字) {preview}..."
            )
        if len(all_chunks) > 5:
            logger.info(f"  ... 还有 {len(all_chunks) - 5} 个段落")

    return all_chunks
