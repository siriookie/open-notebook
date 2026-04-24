"""
Unified embedding utilities for Open Notebook.

Provides centralized embedding generation with support for:
- Single text embedding (with automatic chunking and mean pooling for large texts)
- Batch text embedding (multiple texts with automatic batching)
- Mean pooling for combining multiple embeddings into one

All embedding operations in the application should use these functions
to ensure consistent behavior and proper handling of large content.
"""

import asyncio
import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from loguru import logger

from .chunking import CHUNK_SIZE, ContentType, chunk_text
from .token_utils import token_count


def _get_embedding_batch_size() -> int:
    """
    Read the embedding batch size from the environment.

    This is intentionally configurable because provider limits vary widely, and
    CPU-only local embedding endpoints often need smaller batches than cloud APIs.
    """
    raw = os.getenv("OPEN_NOTEBOOK_EMBEDDING_BATCH_SIZE", "50").strip()
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            "Invalid OPEN_NOTEBOOK_EMBEDDING_BATCH_SIZE='{}'; falling back to 50",
            raw,
        )
        return 50


EMBEDDING_BATCH_SIZE = _get_embedding_batch_size()
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_DELAY = 2  # seconds

# Lazy import to avoid circular dependency:
# utils -> embedding -> models -> key_provider -> provider_config -> utils
if TYPE_CHECKING:
    from open_notebook.ai.models import ModelManager


async def mean_pool_embeddings(embeddings: List[List[float]]) -> List[float]:
    """
    Combine multiple embeddings into a single embedding using mean pooling.

    Algorithm:
    1. Normalize each embedding to unit length
    2. Compute element-wise mean
    3. Normalize the result to unit length

    This approach ensures the final embedding has the same properties as
    individual embeddings (unit length) regardless of input count.

    Args:
        embeddings: List of embedding vectors (each is a list of floats)

    Returns:
        Single embedding vector (mean pooled and normalized)

    Raises:
        ValueError: If embeddings list is empty or embeddings have different dimensions
    """
    if not embeddings:
        raise ValueError("Cannot mean pool empty list of embeddings")

    if len(embeddings) == 1:
        # Single embedding - just normalize and return
        arr = np.array(embeddings[0], dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    # Convert to numpy array
    arr = np.array(embeddings, dtype=np.float64)

    # Verify all embeddings have same dimension
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    # Normalize each embedding to unit length
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)
    normalized = arr / norms

    # Compute mean
    mean = np.mean(normalized, axis=0)

    # Normalize the result
    mean_norm = np.linalg.norm(mean)
    if mean_norm > 0:
        mean = mean / mean_norm

    return mean.tolist()


async def generate_embeddings(
    texts: List[str], command_id: Optional[str] = None
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts with automatic batching and retry.

    Texts are split into batches of EMBEDDING_BATCH_SIZE to avoid exceeding
    provider payload limits. Each batch is retried up to EMBEDDING_MAX_RETRIES
    times on transient failures.

    Args:
        texts: List of text strings to embed
        command_id: Optional command ID for error logging context

    Returns:
        List of embedding vectors, one per input text

    Raises:
        ValueError: If no embedding model is configured
        RuntimeError: If embedding generation fails
    """
    # 空列表直接返回空结果。
    # 这样调用方不需要额外在外面做一次“if texts”防守，也能避免后面无意义地加载模型。
    if not texts:
        return []

    # 延迟导入 model_manager，避免 utils.embedding 和模型层之间出现循环依赖。
    # 只有真正需要发起 embedding 请求时，才把模型管理器拉进来。
    from open_notebook.ai.models import model_manager

    # 读取当前配置好的 embedding 模型。
    # 这里统一通过 model_manager 获取，而不是让每个调用点自己创建模型实例，
    # 这样整个应用的 embedding 行为才能保持一致。
    embedding_model = await model_manager.get_embedding_model()
    if not embedding_model:
        raise ValueError(
            "No embedding model configured. Please configure one in the Models section."
        )

    # 取模型名主要用于日志。
    # 这样批量 embedding 失败时，日志里能明确指出到底是哪一个模型在报错。
    model_name = getattr(embedding_model, "model_name", "unknown")

    # 准备一组用于调试的输入规模指标。
    # 这里故意做成懒计算，是因为 token_count() 对大批文本并不便宜；
    # 只有当前日志级别真的需要输出这些信息时，才去计算。
    metrics: tuple[int, int, int, int] | None = None

    def _get_size_metrics() -> tuple[int, int, int, int]:
        nonlocal metrics
        if metrics is None:
            # 同时统计 token 和字符规模，方便判断失败是“文本太多”还是“文本太长”。
            token_sizes = [token_count(t) for t in texts]
            metrics = (
                min(token_sizes),
                max(token_sizes),
                sum(token_sizes),
                sum(len(t) for t in texts),
            )
        return metrics

    # 用 lazy logging 记录输入规模。
    # 这样在 debug 级别关闭时，不会白白计算 token 统计。
    logger.opt(lazy=True).debug(
        "Generating embeddings for {} texts "
        "(tokens: min={}, max={}, total={}; chars: total={})",
        lambda: len(texts),
        lambda: _get_size_metrics()[0],
        lambda: _get_size_metrics()[1],
        lambda: _get_size_metrics()[2],
        lambda: _get_size_metrics()[3],
    )

    # all_embeddings 按输入顺序累计所有批次的结果。
    # 这里顺序很重要：调用方默认依赖“第 N 条文本对应第 N 个 embedding”。
    all_embeddings: List[List[float]] = []

    # 先根据全局批大小算出总批次数。
    # 之所以分批，是为了避免一次性请求太大，撞上 provider 的 payload / 并发限制。
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    # 逐批处理输入文本。
    # 当前实现选择串行处理批次，而不是所有批次一起并发发出去，
    # 目的是把请求规模和失败面控制得更稳。
    for batch_idx in range(total_batches):
        start = batch_idx * EMBEDDING_BATCH_SIZE
        end = start + EMBEDDING_BATCH_SIZE
        batch = texts[start:end]

        # 每个批次都带有限次重试。
        # 这类请求常见的失败原因是瞬时网络抖动、限流、provider 短暂不可用，
        # 所以没必要第一次失败就立刻让整批任务报废。
        for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
            try:
                # 真正调用 embedding 模型的批量接口。
                # 当前假设 embedding_model.aembed(batch) 返回的向量顺序与输入顺序一致。
                batch_embeddings = await embedding_model.aembed(batch)
                all_embeddings.extend(batch_embeddings)
                # 当前批次成功后，退出重试循环，继续处理下一批。
                break
            except Exception as e:
                # command_id 只是日志上下文，不参与模型调用本身。
                # 这样在后台命令链里出错时，可以把具体失败和 command 记录关联起来。
                cmd_context = f" (command: {command_id})" if command_id else ""
                if attempt < EMBEDDING_MAX_RETRIES:
                    # 还没到最后一次时，记录失败并等待后重试。
                    logger.debug(
                        f"Embedding batch {batch_idx + 1}/{total_batches} "
                        f"attempt {attempt}/{EMBEDDING_MAX_RETRIES} failed "
                        f"using model '{model_name}'{cmd_context}: {e}. Retrying..."
                    )
                    # 固定延迟后重试，避免瞬时连续重放把 provider 压得更紧。
                    await asyncio.sleep(EMBEDDING_RETRY_DELAY)
                else:
                    # 到最后一次仍失败时，不再吞错，直接把这批失败升级成 RuntimeError。
                    # 这样调用方能明确知道是 embedding 流程失败，而不是得到一份不完整结果。
                    logger.debug(
                        f"Embedding batch {batch_idx + 1}/{total_batches} "
                        f"failed after {EMBEDDING_MAX_RETRIES} attempts "
                        f"using model '{model_name}'{cmd_context}: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to generate embeddings using model '{model_name}' "
                        f"(batch {batch_idx + 1}/{total_batches}, "
                        f"{len(batch)} texts): {e}"
                    ) from e

    # 所有批次成功后，记录最终生成的 embedding 数量和批次数。
    logger.debug(f"Generated {len(all_embeddings)} embeddings in {total_batches} batch(es)")
    return all_embeddings


async def generate_embedding(
    text: str,
    content_type: Optional[ContentType] = None,
    file_path: Optional[str] = None,
    command_id: Optional[str] = None,
) -> List[float]:
    """
    Generate a single embedding for text, handling large content via chunking and mean pooling.

    For short text (<= CHUNK_SIZE tokens):
        - Embeds directly and returns the embedding

    For long text (> CHUNK_SIZE tokens):
        - Chunks the text using appropriate splitter for content type
        - Embeds all chunks in batches
        - Combines embeddings via mean pooling

    Args:
        text: The text to embed
        content_type: Optional explicit content type for chunking
        file_path: Optional file path for content type detection
        command_id: Optional command ID for error logging context

    Returns:
        Single embedding vector (list of floats)

    Raises:
        ValueError: If text is empty or no embedding model configured
        RuntimeError: If embedding generation fails
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")

    text = text.strip()
    text_tokens = token_count(text)

    # Check if chunking is needed
    if text_tokens <= CHUNK_SIZE:
        # Short text - embed directly
        logger.debug(f"Embedding short text ({text_tokens} tokens) directly")
        embeddings = await generate_embeddings([text], command_id=command_id)
        return embeddings[0]

    # Long text - chunk and mean pool
    logger.debug(f"Text exceeds chunk size ({text_tokens} tokens), chunking...")

    chunks = chunk_text(text, content_type=content_type, file_path=file_path)

    if not chunks:
        raise ValueError("Text chunking produced no chunks")

    if len(chunks) == 1:
        # Single chunk after splitting
        embeddings = await generate_embeddings(chunks, command_id=command_id)
        return embeddings[0]

    logger.debug(f"Embedding {len(chunks)} chunks and mean pooling")

    # Embed all chunks in batches
    embeddings = await generate_embeddings(chunks, command_id=command_id)

    # Mean pool to get single embedding
    pooled = await mean_pool_embeddings(embeddings)

    logger.debug(f"Mean pooled {len(embeddings)} embeddings into single vector")
    return pooled
