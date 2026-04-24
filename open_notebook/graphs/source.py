import operator
from typing import Any, Dict, List, Optional

from content_core import extract_content
from content_core.common import ProcessSourceState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, TypedDict

from open_notebook.ai.models import Model, ModelManager
from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph


class SourceState(TypedDict):
    content_state: ProcessSourceState
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool


class TransformationState(TypedDict):
    source: Source
    transformation: Transformation


async def content_process(state: SourceState) -> dict:
    # 这里先构造一份内容处理配置对象，作用是为后续抽取阶段提供“默认策略”。
    # 之所以在 graph 节点里集中组装，而不是把这些值散落在 API 层，
    # 是因为“来源抽取”本质上属于处理工作流的一部分，应该和 graph 一起演化。
    content_settings = ContentSettings(
        default_content_processing_engine_doc="auto",
        default_content_processing_engine_url="auto",
        default_embedding_option="ask",
        auto_delete_files="yes",
        # 代码	含义	说明
        # en	英语（English）	通用英语（默认）
        # pt	葡萄牙语（Portuguese）	巴西/葡萄牙通用
        # es	西班牙语（Spanish）	拉美/西班牙通用
        # de	德语（German）	德国/奥地利等
        # nl	荷兰语（Dutch）	荷兰、比利时
        # en-GB	英式英语（British English）	英国用法（拼写/发音差异）
        # fr	法语（French）	法国及法语区
        # hi	印地语（Hindi）	印度主要语言之一
        # ja	日语（Japanese）	日本
        youtube_preferred_languages=[
            "en",
            "pt",
            "es",
            "de",
            "nl",
            "en-GB",
            "fr",
            "hi",
            "ja",
        ],
    )
    # 取出上游传进来的 content_state。
    # 这份状态在 create_source / process_source_command 中已经被统一规整过，
    # 这里继续在其上补充处理参数，而不是重新组装一份新对象，
    # 这样可以保留 URL、file_path、content 等原始输入上下文。
    content_state: Dict[str, Any] = state["content_state"]  # type: ignore[assignment]

    # 把文档和 URL 的默认处理引擎写进 content_state。
    # content-core 后续会根据这些字段决定采用哪套抽取实现；
    # 这里统一设置为配置值或 "auto"，是为了把“处理策略选择权”下放给抽取层，
    # 而不是在 graph 里手写一堆 if/else 分支。
    content_state["url_engine"] = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    content_state["document_engine"] = (
        content_settings.default_content_processing_engine_doc or "auto"
    )
    # 强制输出为 markdown，是为了让后续保存到 Source.full_text 的内容格式尽量统一。
    # 统一格式之后，后面的 transformation、chat context、embedding chunking 都更稳定，
    # 不用分别处理 HTML、纯文本、网页结构化片段等多种输出形态。
    content_state["output_format"] = "markdown"

    # 尝试把默认 STT 模型配置注入到 content_state。
    # 这是为了让音频/视频/YouTube 这类来源在需要转写时，优先使用项目里已配置好的
    # speech-to-text 模型，而不是完全依赖 content-core 自己的默认行为。
    try:
        model_manager = ModelManager()
        # get_defaults() 会从数据库读取 DefaultModels 记录，而不是读硬编码常量；
        # 这样用户在设置页切换默认 STT 模型后，后续来源处理就能立即生效。
        defaults = await model_manager.get_defaults()
        if defaults.default_speech_to_text_model:
            # 这里再把默认模型 ID 解析成真正的 Model 记录，目的是拿到 provider 和 model name。
            # content-core 消费的是 audio_provider/audio_model，而不是 Open Notebook 自己的 model_id。
            stt_model = await Model.get(defaults.default_speech_to_text_model)
            if stt_model:
                content_state["audio_provider"] = stt_model.provider
                content_state["audio_model"] = stt_model.name
                logger.debug(
                    f"Using speech-to-text model: {stt_model.provider}/{stt_model.name}"
                )
    except Exception as e:
        # 这里故意只记 warning，不让整个节点失败。
        # 原因是“拿不到自定义 STT 模型”并不一定意味着来源无法处理：
        # 对纯文本、网页、普通文档来说根本不需要 STT；
        # 即便是音视频，content-core 也可能还有自己的默认回退路径。
        logger.warning(f"Failed to retrieve speech-to-text model configuration: {e}")
        # Continue without custom audio model (content-core will use its default)

    # 把标准化后的 content_state 交给 content-core 做真正的内容抽取。
    # 我目前无法直接展开 content-core 依赖源码，因此这里只解释当前代码已明确表达出的职责：
    # 输入是统一的处理状态，输出是包含抽取结果的 processed_state。
    processed_state = await extract_content(content_state)

    # 抽取结束后，先在 graph 边界上做一次“是否真的拿到文本内容”的判定。
    # 这样后面的 save_source、vectorize、transform_content 就都可以建立在
    # “content 一定是非空文本”这个前提上，避免在多个节点重复做空值防御。
    if not processed_state.content or not processed_state.content.strip():
        url = processed_state.url or ""
        # 对 YouTube 单独给出更有操作性的错误信息。
        # 这里之所以专门分支，是因为视频场景常见失败原因不是“链接坏了”，
        # 而是“没有字幕/没有转写模型”，直接提示用户去配置 STT 能明显降低排障成本。
        if url and ("youtube.com" in url or "youtu.be" in url):
            raise ValueError(
                "Could not extract content from this YouTube video. "
                "No transcript or subtitles are available. "
                "Try configuring a Speech-to-Text model in Settings "
                "to transcribe the audio instead."
            )
        # 非 YouTube 场景统一报“抽不到文本”。
        # 这里抛 ValueError 而不是吞掉返回空内容，是为了让上层命令把它视作永久性失败，
        # 避免无意义重试，也避免后续把空内容写入 source/full_text。
        raise ValueError(
            "Could not extract any text content from this source. "
            "The content may be empty, inaccessible, or in an unsupported format."
        )

    # 返回更新后的 content_state，供下一个 save_source 节点继续使用。
    # graph 里只把必要的增量状态往后传，保持节点之间职责清晰：
    # 这个节点只负责“把原始来源变成已抽取文本”，不负责持久化。
    return {"content_state": processed_state}


async def save_source(state: SourceState) -> dict:
    # content_process 已经把原始输入转成了统一的提取结果；
    # 这里不再重复做抽取，只消费前一节点产出的标准化状态并负责持久化。
    content_state = state["content_state"]

    # 这里必须按 source_id 取回“已经存在”的 Source，而不是新建一个：
    # 1. 上游流程/接口已经提前创建了 Source 记录，并把它的 id 放进 graph state；
    # 2. 这个函数的职责是把抽取结果写回原记录，保持同一个 source 身份不变；
    # 3. Source.get() 最终会查询数据库中的现有记录，拿不到就抛错，这样能尽早暴露
    #    “状态里的 source_id 无效”这种真正的数据一致性问题。
    source = await Source.get(state["source_id"])
    if not source:
        raise ValueError(f"Source with ID {state['source_id']} not found")

    # 用抽取后的定位信息回填 asset。
    # 这里同时保存 url 和 file_path，是因为 Source.asset 本身就承担“来源定位”的职责：
    # 无论内容是来自网页还是上传文件，后续代码都可以从同一字段读取来源元数据，
    # 不需要分别追踪不同输入通道的临时参数。
    source.asset = Asset(url=content_state.url, file_path=content_state.file_path)

    # 把抽取出的正文写入 full_text。
    # 后面的 transformation、聊天上下文和向量化都以 Source.full_text 为输入，
    # 所以必须先把最新文本持久化到 Source 上，而不是只留在 graph 的临时 state 里。
    source.full_text = content_state.content

    # 只在标题为空，或仍是占位值 "Processing..." 时，才用抽取出的标题覆盖。
    # 这样做是为了保护用户已经手动改过的标题，避免一次内容重处理把用户命名冲掉。
    # 这个行为不是猜测：tests/test_graphs.py 里有专门测试覆盖了
    # “自定义标题保留、占位标题可替换、空标题可补全”这几种情况。
    if content_state.title and (not source.title or source.title == "Processing..."):
        source.title = content_state.title

    # 先保存 Source，再决定是否提交向量化任务。
    # 原因是 Source.save() 负责把当前对象状态真正写进数据库；而 Source.vectorize()
    # 不做内联 embedding，它只是提交一个后台 embed_source 命令，命令参数只有 source_id。
    # 因此必须保证数据库里已经有最新的 full_text / asset / title，后台任务按 source_id
    # 读取时才能看到这次处理后的内容，而不是旧数据。
    await source.save()

    # Notebook 关联关系不在这里补建。
    # 当前文件里的约定很明确：关联边由 API 层更早创建，用来提升界面响应速度；
    # 如果这里再建一次，就会让“保存内容”和“建立 notebook 关联”两个职责重叠，
    # 还可能产生重复关系。

    if state["embed"]:
        # 只有显式要求 embed 时，才提交向量化后台任务。
        # 这是把“保存 source”和“是否为检索建立向量索引”解耦：有些调用方只需要保存文本，
        # 不一定希望立刻触发额外的异步作业。
        if source.full_text and source.full_text.strip():
            # 这里再次检查文本非空，不依赖调用方口头保证。
            # 直接原因是 Source.vectorize() 自身就要求 full_text 非空并会在空文本时报错；
            # 在 graph 节点里先守住这个前提，可以把“跳过 embedding”记录成 warning，
            # 而不是把整个 save_source 流程变成失败。
            logger.debug("Embedding content for vector search")
            await source.vectorize()
        else:
            logger.warning(
                f"Source {source.id} has no text content to embed, skipping vectorization"
            )

    # 返回更新后的 source，供后续条件边和 transformation 节点继续复用同一个对象。
    return {"source": source}


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
    if len(state["apply_transformations"]) == 0:
        return []

    to_apply = state["apply_transformations"]
    logger.debug(f"Applying transformations {to_apply}")

    return [
        Send(
            "transform_content",
            {
                "source": state["source"],
                "transformation": t,
            },
        )
        for t in to_apply
    ]


async def transform_content(state: TransformationState) -> Optional[dict]:
    source = state["source"]
    content = source.full_text
    if not content:
        return None
    transformation: Transformation = state["transformation"]

    logger.debug(f"Applying transformation {transformation.name}")
    result = await transform_graph.ainvoke(
        dict(input_text=content, transformation=transformation)  # type: ignore[arg-type]
    )
    await source.add_insight(transformation.title, result["output"])
    return {
        "transformation": [
            {
                "output": result["output"],
                "transformation_name": transformation.name,
            }
        ]
    }


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("transform_content", transform_content)
# Define the graph edges
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
