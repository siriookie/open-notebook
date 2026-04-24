import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.database.repository import ensure_record_id
from open_notebook.domain.notebook import Source
from open_notebook.domain.transformation import Transformation
from open_notebook.exceptions import ConfigurationError

try:
    from open_notebook.graphs.source import source_graph
    from open_notebook.graphs.transformation import graph as transform_graph
except ImportError as e:
    logger.error(f"Failed to import graphs: {e}")
    raise ValueError("graphs not available")


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class SourceProcessingInput(CommandInput):
    source_id: str
    content_state: Dict[str, Any]
    notebook_ids: List[str]
    transformations: List[str]
    embed: bool


class SourceProcessingOutput(CommandOutput):
    success: bool
    source_id: str
    embedded_chunks: int = 0
    insights_created: int = 0
    processing_time: float
    error_message: Optional[str] = None


@command(
    "process_source",
    app="open_notebook",
    retry={
        # 这里把重试次数设得较高，是为了吸收后台队列堆积、
        # SurrealDB v2 事务冲突等瞬时失败，而不是因为这条命令本身逻辑不稳定。
        "max_attempts": 15,  # Handle deep queues (workaround for SurrealDB v2 transaction conflicts)
        "wait_strategy": "exponential_jitter",
        "wait_min": 1,
        # 允许最大等待 120 秒，避免失败任务在高压时刻立即回冲，给队列和数据库留出恢复窗口。
        "wait_max": 120,  # Allow queue to drain
        # 这两类错误属于永久性问题，重试没有价值：
        # - ValueError：输入不合法、source/transformation 不存在等
        # - ConfigurationError：运行配置本身有问题
        "stop_on": [ValueError, ConfigurationError],  # Don't retry validation/config errors
        # 事务冲突重试时日志可能非常密集，降到 debug 可以减少噪音，保留真正重要的错误。
        "retry_log_level": "debug",  # Avoid log noise during transaction conflicts
    },
)
async def process_source_command(
    input_data: SourceProcessingInput,
) -> SourceProcessingOutput:
    """
    Process source content using the source_graph workflow
    """
    # 记录命令开始时间，供最终输出 processing_time。
    # 这里统计的是“整条后台命令”的端到端耗时，而不只是 graph 内某个节点的局部耗时。
    start_time = time.time()

    try:
        # 先把关键输入打到日志里。
        # 由于这是后台命令，排障时通常拿不到原始 HTTP 请求上下文，所以入口日志很重要。
        logger.info(f"Starting source processing for source: {input_data.source_id}")
        logger.info(f"Notebook IDs: {input_data.notebook_ids}")
        logger.info(f"Transformations: {input_data.transformations}")
        logger.info(f"Embed: {input_data.embed}")

        # 先把 transformation ID 解析成真正的 Transformation 对象。
        # source_graph 期望收到的是对象而不是字符串，这样后续节点可以直接使用
        # transformation 的 name/title/prompt 等字段。
        transformations = []
        for trans_id in input_data.transformations:
            logger.info(f"Loading transformation: {trans_id}")
            transformation = await Transformation.get(trans_id)
            if not transformation:
                raise ValueError(f"Transformation '{trans_id}' not found")
            transformations.append(transformation)

        logger.info(f"Loaded {len(transformations)} transformations")

        # 这条命令不会创建 Source，而是更新 API 层预先创建好的占位 source。
        # 因此 source_id 必须对应一个已经存在的数据库记录。
        source = await Source.get(input_data.source_id)
        if not source:
            raise ValueError(f"Source '{input_data.source_id}' not found")

        # 把当前后台任务的 command_id 回写到 source.command。
        # 这是前端轮询来源状态的关键锚点：来源卡片会通过 source.command 找到对应命令状态，
        # 决定展示 new / queued / running / completed / failed。
        source.command = (
            ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None
        )
        # 持久化 command 引用，确保其他读路径也能看到这个处理中的任务。
        await source.save()

        logger.info(f"Updated source {source.id} with command reference")

        # 到这里才真正进入“来源处理”主流程。
        # notebook_ids 仍然继续往下传，是为了满足 SourceState 的输入结构；
        # notebook 关联本身已经在 API 层创建，这里主要负责内容处理。
        logger.info(f"Processing source with {len(input_data.notebook_ids)} notebooks")

        # source_graph 是来源处理的核心编排：
        # 1. content_process：抽取 URL/文件/文本内容
        # 2. save_source：把 full_text / asset / 标题写回 source
        # 3. transform_content：按 transformation 生成 insights
        # 4. 如需嵌入，在 save_source 中进一步提交 embed_source 后台任务
        # 命令层只负责准备输入和接收结果，具体流程统一交给 graph 维护。
        result = await source_graph.ainvoke(
            {  # type: ignore[arg-type]
                "content_state": input_data.content_state,
                "notebook_ids": input_data.notebook_ids,  # Use notebook_ids (plural) as expected by SourceState
                "apply_transformations": transformations,
                "embed": input_data.embed,
                "source_id": input_data.source_id,  # Add the source_id to the state
            }
        )

        # graph 返回的是最终状态字典，这里取出已经被更新过的 source。
        # 它比命令开头读到的占位 source 更接近“处理完成后的真实状态”。
        processed_source = result["source"]

        # 4. Gather processing results (notebook associations handled by source_graph)
        # Note: embedding is fire-and-forget (async job), so we can't query the
        # count here — it hasn't completed yet. The embed_source_command logs
        # the actual count when it finishes.
        # 收尾阶段只统计“当前命令可以确定”的结果。
        # insight 是 graph 内同步创建的，所以这里可以立即查询；
        # embedding 则是 save_source -> source.vectorize() 里再提交一个后台命令，
        # 当前命令结束时它通常还没真正跑完，因此这里不能返回真实 embedded_chunks。
        insights_list = await processed_source.get_insights()
        insights_created = len(insights_list)

        # processing_time 统一在出口处计算，反映一次完整来源处理命令的总耗时。
        processing_time = time.time() - start_time
        embed_status = "submitted" if input_data.embed else "skipped"
        logger.info(
            f"Successfully processed source: {processed_source.id} in {processing_time:.2f}s"
        )
        logger.info(
            f"Created {insights_created} insights, embedding {embed_status}"
        )

        return SourceProcessingOutput(
            success=True,
            source_id=str(processed_source.id),
            # 这里固定返回 0 是有意设计：
            # 真正的 embedding 任务是异步 fire-and-forget，当前时刻还拿不到最终 chunk 数。
            embedded_chunks=0,
            insights_created=insights_created,
            processing_time=processing_time,
        )

    except ValueError as e:
        # ValueError 代表永久性业务失败，例如 source/transformation 不存在，
        # 或 graph 发现输入内容本身不可处理。对这类错误重试没有意义，
        # 所以直接返回失败结果，让任务以明确终态结束。
        processing_time = time.time() - start_time
        logger.error(f"Source processing failed: {e}")
        return SourceProcessingOutput(
            success=False,
            source_id=input_data.source_id,
            processing_time=processing_time,
            error_message=str(e),
        )
    except Exception as e:
        # 其他异常统一视为暂时性失败，交给 surreal-commands 的 retry 策略处理。
        # 这里故意继续 raise，而不是吞掉错误返回失败结果，
        # 因为框架只有在异常冒泡时才会认定这次执行值得重试。
        logger.debug(
            f"Transient error processing source {input_data.source_id}: {e}"
        )
        raise


# =============================================================================
# RUN TRANSFORMATION COMMAND
# =============================================================================


class RunTransformationInput(CommandInput):
    """Input for running a transformation on an existing source."""

    source_id: str
    transformation_id: str


class RunTransformationOutput(CommandOutput):
    """Output from transformation command."""

    success: bool
    source_id: str
    transformation_id: str
    processing_time: float
    error_message: Optional[str] = None


@command(
    "run_transformation",
    app="open_notebook",
    retry={
        "max_attempts": 5,
        "wait_strategy": "exponential_jitter",
        "wait_min": 1,
        "wait_max": 60,
        "stop_on": [ValueError, ConfigurationError],  # Don't retry validation/config errors
        "retry_log_level": "debug",
    },
)
async def run_transformation_command(
    input_data: RunTransformationInput,
) -> RunTransformationOutput:
    """
    Run a transformation on an existing source to generate an insight.

    This command runs the transformation graph which:
    1. Loads the source and transformation
    2. Calls the LLM to generate insight content
    3. Creates the insight via create_insight command (fire-and-forget)

    Use this command for UI-triggered insight generation to avoid blocking
    the HTTP request while the LLM processes.

    Retry Strategy:
    - Retries up to 5 times for transient failures (network, timeout, etc.)
    - Uses exponential-jitter backoff (1-60s)
    - Does NOT retry permanent failures (ValueError for validation errors)
    """
    start_time = time.time()

    try:
        logger.info(
            f"Running transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}"
        )

        # Load source
        source = await Source.get(input_data.source_id)
        if not source:
            raise ValueError(f"Source '{input_data.source_id}' not found")

        # Load transformation
        transformation = await Transformation.get(input_data.transformation_id)
        if not transformation:
            raise ValueError(
                f"Transformation '{input_data.transformation_id}' not found"
            )

        # Run transformation graph (includes LLM call + insight creation)
        await transform_graph.ainvoke(
            input=dict(source=source, transformation=transformation)
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully ran transformation {input_data.transformation_id} "
            f"on source {input_data.source_id} in {processing_time:.2f}s"
        )

        return RunTransformationOutput(
            success=True,
            source_id=input_data.source_id,
            transformation_id=input_data.transformation_id,
            processing_time=processing_time,
        )

    except ValueError as e:
        # Validation errors are permanent failures - don't retry
        processing_time = time.time() - start_time
        logger.error(
            f"Failed to run transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}: {e}"
        )
        return RunTransformationOutput(
            success=False,
            source_id=input_data.source_id,
            transformation_id=input_data.transformation_id,
            processing_time=processing_time,
            error_message=str(e),
        )
    except Exception as e:
        # Transient failure - will be retried (surreal-commands logs final failure)
        logger.debug(
            f"Transient error running transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}: {e}"
        )
        raise
