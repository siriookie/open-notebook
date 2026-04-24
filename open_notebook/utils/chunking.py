"""
Chunking utilities for Open Notebook.

Provides content-type detection and smart text chunking for embedding operations.
Supports HTML, Markdown, and plain text with appropriate splitters for each type.

Key functions:
- detect_content_type(): Detects content type from file extension or content heuristics
- chunk_text(): Splits text into chunks using appropriate splitter for content type

Environment Variables:
    OPEN_NOTEBOOK_CHUNK_SIZE: Maximum chunk size in tokens (default: 400)
    OPEN_NOTEBOOK_CHUNK_OVERLAP: Overlap between chunks in tokens (default: 15% of CHUNK_SIZE)
"""

import os
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from .token_utils import token_count


def _get_chunk_size() -> int:
    """Get chunk size from environment variable or use default."""
    chunk_size_str = os.getenv("OPEN_NOTEBOOK_CHUNK_SIZE")
    if chunk_size_str:
        try:
            chunk_size = int(chunk_size_str)
            if chunk_size < 100:
                logger.warning(
                    f"OPEN_NOTEBOOK_CHUNK_SIZE ({chunk_size}) is too small. "
                    f"Using minimum value of 100."
                )
                return 100
            if chunk_size > 8192:
                logger.warning(
                    f"OPEN_NOTEBOOK_CHUNK_SIZE ({chunk_size}) is very large. "
                    f"This may cause issues with some embedding models."
                )
            logger.info(f"Using custom chunk size: {chunk_size} tokens")
            return chunk_size
        except ValueError:
            logger.warning(
                f"Invalid OPEN_NOTEBOOK_CHUNK_SIZE value: '{chunk_size_str}'. "
                f"Using default: 400"
            )
    return 400


def _get_chunk_overlap(chunk_size: int) -> int:
    """Get chunk overlap from environment variable or calculate default (15% of chunk size)."""
    overlap_str = os.getenv("OPEN_NOTEBOOK_CHUNK_OVERLAP")
    if overlap_str:
        try:
            overlap = int(overlap_str)
            if overlap < 0:
                logger.warning(
                    f"OPEN_NOTEBOOK_CHUNK_OVERLAP ({overlap}) cannot be negative. "
                    f"Using 0."
                )
                return 0
            if overlap >= chunk_size:
                logger.warning(
                    f"OPEN_NOTEBOOK_CHUNK_OVERLAP ({overlap}) cannot be >= chunk size ({chunk_size}). "
                    f"Using 15% of chunk size: {int(chunk_size * 0.15)}"
                )
                return int(chunk_size * 0.15)
            logger.info(f"Using custom chunk overlap: {overlap} tokens")
            return overlap
        except ValueError:
            logger.warning(
                f"Invalid OPEN_NOTEBOOK_CHUNK_OVERLAP value: '{overlap_str}'. "
                f"Using default: 15% of chunk size"
            )
    return int(chunk_size * 0.15)


# Constants (computed at import time from environment variables)
CHUNK_SIZE = _get_chunk_size()
CHUNK_OVERLAP = _get_chunk_overlap(CHUNK_SIZE)
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Threshold for heuristics to override extension

logger.debug(
    f"Chunking configuration: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}"
)


class ContentType(Enum):
    """Content type for chunking strategy selection."""

    HTML = "html"
    MARKDOWN = "markdown"
    PLAIN = "plain"


# File extension mappings
_EXTENSION_TO_CONTENT_TYPE = {
    # HTML
    ".html": ContentType.HTML,
    ".htm": ContentType.HTML,
    ".xhtml": ContentType.HTML,
    # Markdown
    ".md": ContentType.MARKDOWN,
    ".markdown": ContentType.MARKDOWN,
    ".mdown": ContentType.MARKDOWN,
    ".mkd": ContentType.MARKDOWN,
    # Plain text (explicit)
    ".txt": ContentType.PLAIN,
    ".text": ContentType.PLAIN,
    # Code files (treat as plain)
    ".py": ContentType.PLAIN,
    ".js": ContentType.PLAIN,
    ".ts": ContentType.PLAIN,
    ".java": ContentType.PLAIN,
    ".c": ContentType.PLAIN,
    ".cpp": ContentType.PLAIN,
    ".go": ContentType.PLAIN,
    ".rs": ContentType.PLAIN,
    ".rb": ContentType.PLAIN,
    ".php": ContentType.PLAIN,
    ".sh": ContentType.PLAIN,
    ".bash": ContentType.PLAIN,
    ".zsh": ContentType.PLAIN,
    ".sql": ContentType.PLAIN,
    ".json": ContentType.PLAIN,
    ".yaml": ContentType.PLAIN,
    ".yml": ContentType.PLAIN,
    ".xml": ContentType.PLAIN,
    ".csv": ContentType.PLAIN,
    ".tsv": ContentType.PLAIN,
}


def detect_content_type_from_extension(
    file_path: Optional[str],
) -> Optional[ContentType]:
    """
    Detect content type from file extension.

    Args:
        file_path: Path to the file (can be full path or just filename)

    Returns:
        ContentType if extension is recognized, None otherwise
    """
    if not file_path:
        return None

    try:
        extension = Path(file_path).suffix.lower()
        return _EXTENSION_TO_CONTENT_TYPE.get(extension)
    except Exception:
        return None


def detect_content_type_from_heuristics(text: str) -> Tuple[ContentType, float]:
    """
    Detect content type using content heuristics.

    Args:
        text: The text content to analyze

    Returns:
        Tuple of (ContentType, confidence_score) where confidence is 0.0-1.0
    """
    if not text or len(text) < 10:
        return ContentType.PLAIN, 0.5

    # Sample first 5000 chars for efficiency
    sample = text[:5000]

    # Check HTML first (most specific patterns)
    html_score = _calculate_html_score(sample)
    if html_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ContentType.HTML, html_score

    # Check Markdown
    markdown_score = _calculate_markdown_score(sample)
    if markdown_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ContentType.MARKDOWN, markdown_score

    # Return the higher scoring type, or PLAIN if both are low
    if html_score > markdown_score and html_score > 0.3:
        return ContentType.HTML, html_score
    elif markdown_score > 0.3:
        return ContentType.MARKDOWN, markdown_score
    else:
        return ContentType.PLAIN, 0.6


def _calculate_html_score(text: str) -> float:
    """Calculate confidence score for HTML content."""
    score = 0.0
    indicators = 0

    # Strong indicators
    if re.search(r"<!DOCTYPE\s+html", text, re.IGNORECASE):
        score += 0.4
        indicators += 1

    if re.search(r"<html[\s>]", text, re.IGNORECASE):
        score += 0.3
        indicators += 1

    # Structural tags
    structural_tags = ["<head", "<body", "<div", "<span", "<p>", "<table", "<form"]
    for tag in structural_tags:
        if tag.lower() in text.lower():
            score += 0.1
            indicators += 1
            if indicators >= 5:
                break

    # Header tags
    if re.search(r"<h[1-6][\s>]", text, re.IGNORECASE):
        score += 0.15
        indicators += 1

    # Closing tags pattern
    if re.search(r"</\w+>", text):
        score += 0.1
        indicators += 1

    return min(score, 1.0)


def _calculate_markdown_score(text: str) -> float:
    """Calculate confidence score for Markdown content."""
    score = 0.0
    indicators = 0

    # Headers (# ## ###) - strong indicator
    header_matches = len(re.findall(r"^#{1,6}\s+.+", text, re.MULTILINE))
    if header_matches >= 3:
        score += 0.35
        indicators += 1
    elif header_matches >= 1:
        score += 0.2
        indicators += 1

    # Links [text](url) - strong indicator
    link_matches = len(re.findall(r"\[.+?\]\(.+?\)", text))
    if link_matches >= 2:
        score += 0.25
        indicators += 1
    elif link_matches >= 1:
        score += 0.15
        indicators += 1

    # Code blocks ``` - strong indicator
    if re.search(r"^```", text, re.MULTILINE):
        score += 0.2
        indicators += 1

    # Inline code `code`
    if re.search(r"`[^`]+`", text):
        score += 0.1
        indicators += 1

    # Lists (-, *, +, or numbered)
    list_matches = len(re.findall(r"^[\*\-\+]\s+", text, re.MULTILINE))
    list_matches += len(re.findall(r"^\d+\.\s+", text, re.MULTILINE))
    if list_matches >= 3:
        score += 0.15
        indicators += 1
    elif list_matches >= 1:
        score += 0.08
        indicators += 1

    # Bold/italic
    if re.search(r"\*\*.+?\*\*|__.+?__", text):
        score += 0.1
        indicators += 1

    # Blockquotes
    if re.search(r"^>\s+", text, re.MULTILINE):
        score += 0.1
        indicators += 1

    return min(score, 1.0)


def detect_content_type(text: str, file_path: Optional[str] = None) -> ContentType:
    """
    使用文件扩展名（主判断）和内容启发式规则（兜底）来识别内容类型。

    策略：
    1. 如果文件扩展名存在且能识别，优先使用扩展名结果
    2. 如果没有扩展名，或扩展名无法提供有效判断，则使用内容启发式规则
    3. 只有当启发式判断置信度非常高时，才允许它覆盖扩展名给出的 plain 结果

    Args:
        text: 要分析的文本内容
        file_path: 可选的文件路径，用于基于扩展名进行检测

    Returns:
        检测得到的 ContentType
    """
    # 先尝试基于扩展名判断内容类型。
    # 这样做的原因是：扩展名判断成本最低，而且对 .html / .md 这类明确后缀通常已经足够可靠。
    # 但它也不是绝对可信，所以这里只把它当“主判断”，不是唯一判断。
    extension_type = detect_content_type_from_extension(file_path)

    # 再基于文本内容本身做启发式判断，并拿到置信度。
    # 这一步会分析文本像不像 HTML / Markdown / plain text，
    # 用来弥补“没有扩展名”或“扩展名过于泛化”的情况。
    heuristic_type, confidence = detect_content_type_from_heuristics(text)

    # 如果扩展名完全帮不上忙，就直接采用启发式结果。
    # 当前实现里 extension_type 为 None 代表：
    # - 没传 file_path
    # - 路径没有后缀
    # - 后缀不在已知映射表里
    # 这时再坚持扩展名优先就没有意义了。
    if extension_type is None:
        logger.debug(
            f"No file extension, using heuristics: {heuristic_type.value} "
            f"(confidence: {confidence:.2f})"
        )
        return heuristic_type

    # 如果扩展名只给出一个比较保守的 plain 结果，而内容启发式又非常有把握，
    # 就允许启发式覆盖扩展名。
    # 这里只允许覆盖 plain，而不轻易覆盖 html / markdown，
    # 说明当前策略更偏向“修正过于泛化的扩展名”，而不是全面推翻文件后缀判断。
    # 典型场景是：文件后缀是 .txt，但里面其实是成段的 Markdown 或 HTML 内容。
    if extension_type == ContentType.PLAIN and confidence >= HIGH_CONFIDENCE_THRESHOLD:
        logger.debug(
            f"Extension suggests plain, but heuristics override with "
            f"{heuristic_type.value} (confidence: {confidence:.2f})"
        )
        return heuristic_type

    # 除了上面那种“plain 被高置信度纠正”的情况，其他场景都信任扩展名。
    # 这体现了当前函数的整体取舍：扩展名优先，启发式兜底，只在非常明确时才纠偏。
    logger.debug(f"Using extension-based content type: {extension_type.value}")
    return extension_type


def _get_html_splitter() -> HTMLHeaderTextSplitter:
    """Get HTML header splitter configured for h1, h2, h3."""
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    return HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


def _get_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """Get Markdown header splitter configured for #, ##, ###."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )


def _get_plain_splitter() -> RecursiveCharacterTextSplitter:
    """Get plain text splitter using CHUNK_SIZE and CHUNK_OVERLAP constants."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_count,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def _apply_secondary_chunking(chunks: List[str]) -> List[str]:
    """
    Apply secondary chunking to ensure no chunk exceeds CHUNK_SIZE tokens.

    Used when primary splitters (HTML/Markdown) produce oversized chunks.
    """
    result = []
    secondary_splitter = _get_plain_splitter()

    for chunk in chunks:
        if token_count(chunk) > CHUNK_SIZE:
            # Split oversized chunk
            sub_chunks = secondary_splitter.split_text(chunk)
            result.extend(sub_chunks)
        else:
            result.append(chunk)

    return result


def chunk_text(
    text: str,
    content_type: Optional[ContentType] = None,
    file_path: Optional[str] = None,
) -> List[str]:
    """
    根据内容类型，使用对应的 splitter 将文本切分成多个 chunks。

    Args:
        text: 需要切分的文本
        content_type: 可选的显式内容类型；如果不传，会自动检测
        file_path: 可选的文件路径，用于辅助内容类型检测

    Returns:
        文本 chunk 列表；每个 chunk 的大小大致不超过 CHUNK_SIZE tokens
    """
    # 空文本或全空白文本直接返回空列表。
    # 这里尽早返回，是为了避免后面做 token 统计、类型检测和 splitter 初始化这些无意义工作。
    if not text or not text.strip():
        return []

    # 先计算整段文本的 token 数。
    # 后面的“是否需要切块”判断依赖的不是字符数，而是更接近 embedding 模型约束的 token 数。
    text_tokens = token_count(text)

    # 短文本不需要切块，直接作为一个整体返回。
    # 这样做能保留最完整的上下文，避免把本来就很短的内容切碎，反而损失语义完整性。
    if text_tokens <= CHUNK_SIZE:
        return [text]

    # 如果调用方没有显式给出内容类型，就在这里自动检测。
    # 之所以延后到这里才检测，是因为短文本已经直接返回了，没必要为它们额外做类型识别。
    if content_type is None:
        content_type = detect_content_type(text, file_path)

    # 记录当前使用的切块策略，方便后续排查“为什么某段文本被按某种方式切开”。
    logger.debug(f"Chunking text with content type: {content_type.value}")

    # 根据内容类型选择不同的 splitter。
    # 这里的核心思路不是“所有文本都用同一种固定规则切”，而是尽量利用原文本的结构信息：
    # - HTML 按标题层级切
    # - Markdown 按标题层级切
    # - plain text 用递归字符切分
    if content_type == ContentType.HTML:
        # HTML 文本优先按 h1 / h2 / h3 这类结构边界切分。
        # 这样切出来的 chunk 更接近页面的语义段落，而不是机械按长度截断。
        splitter = _get_html_splitter()
        # HTML splitter 返回的是 Document 对象，而不是纯字符串。
        # 所以这里要把 page_content 提取出来，统一成字符串列表。
        docs = splitter.split_text(text)
        chunks = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in docs
        ]
    elif content_type == ContentType.MARKDOWN:
        # Markdown 文本优先按 # / ## / ### 标题结构切分。
        # 这样能尽量保留 markdown 原有层级，对后续检索和问答通常更友好。
        splitter = _get_markdown_splitter()
        # Markdown splitter 同样返回 Document 对象，因此也要统一抽出 page_content。
        docs = splitter.split_text(text)
        chunks = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in docs
        ]
    else:
        # 普通文本没有稳定的结构标签可依赖，所以直接用递归字符切分器。
        # 它会按段落、换行、标点、空格等分隔符逐层尝试，尽量找到自然边界。
        splitter = _get_plain_splitter()
        chunks = splitter.split_text(text)

    # 对 HTML / Markdown 结果再做一次二次切块。
    # 原因是按标题切出来的块虽然结构更自然，但有可能仍然过大；
    # 二次切块负责把这些超长块再压回到 CHUNK_SIZE 约束内。
    if content_type in (ContentType.HTML, ContentType.MARKDOWN):
        chunks = _apply_secondary_chunking(chunks)

    # 清理空 chunk 和纯空白 chunk。
    # 这是为了避免后续 embedding 阶段收到没有内容的块，造成无意义向量或额外错误处理。
    chunks = [c.strip() for c in chunks if c and c.strip()]

    # 记录最终切块结果，便于观察切块前后的规模变化。
    logger.debug(f"Created {len(chunks)} chunks from {text_tokens} tokens")
    return chunks
