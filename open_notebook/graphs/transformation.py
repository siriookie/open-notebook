from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.domain.notebook import Source
from open_notebook.domain.transformation import DefaultPrompts, Transformation
from open_notebook.exceptions import OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content


class TransformationState(TypedDict):
    input_text: str
    source: Source
    transformation: Transformation
    output: str


async def run_transformation(state: dict, config: RunnableConfig) -> dict:
    # 这里既支持直接传入 source，也支持直接传入 input_text。
    # 这样同一条 transformation graph 就能复用在两种场景里：
    # 1. 面向已存在 Source 的正式流程
    # 2. 面向任意输入文本的独立转换流程
    source_obj = state.get("source")
    source: Source = source_obj if isinstance(source_obj, Source) else None  # type: ignore[assignment]

    # input_text 是可选显式输入；如果没有，后面会回退到 source.full_text。
    content = state.get("input_text")

    # transformation 至少需要一个可转换的文本来源：source 或 input_text。
    # 这里用 assert 把这个前提写死，避免在后面拼 prompt 或调用模型时才暴露空输入问题。
    assert source or content, "No content to transform"

    # transformation 对象里包含这次转换的 prompt / title / name 等核心定义。
    transformation: Transformation = state["transformation"]

    try:
        # 如果调用方没有显式传 input_text，就默认使用 source.full_text。
        # 这让“对现有 source 运行 transformation”成为最自然的默认路径，
        # 调用方不需要每次都手动把 source.full_text 再复制一份传进来。
        if not content:
            content = source.full_text

        # 先拿 transformation 自身定义的 prompt 作为基础模板。
        transformation_template_text = transformation.prompt

        # 这里还预留了全局默认 transformation instructions 的拼接入口。
        # 当前代码虽然默认值是 None，但保留这个组合位，说明设计上允许以后给所有 transformation
        # 统一加一层全局约束，而不必逐条去改 transformation.prompt。
        default_prompts: DefaultPrompts = DefaultPrompts(transformation_instructions=None)
        if default_prompts.transformation_instructions:
            transformation_template_text = f"{default_prompts.transformation_instructions}\n\n{transformation_template_text}"

        # 在模板末尾追加一个显式的 INPUT 分隔标记。
        # 这样做的目的，是把“规则说明”和“待处理内容”分开，让后面的系统提示词结构更稳定。
        transformation_template_text = f"{transformation_template_text}\n\n# INPUT"

        # 用 Prompter 渲染系统提示词。
        # 这里不是简单把 transformation.prompt 原样发给模型，而是允许模板从 state 中读取变量，
        # 这样 transformation prompt 可以按 source / transformation 等上下文动态生成。
        system_prompt = Prompter(template_text=transformation_template_text).render(
            data=state
        )

        # 把待转换内容规范成字符串。
        # 即使 content 理论上通常已经是 str，这里仍然显式转一次，是为了降低上游传入非字符串对象时的脆弱性。
        content_str = str(content) if content else ""

        # 构造最终发给模型的消息序列：
        # - SystemMessage：放规则、角色和 transformation 指令
        # - HumanMessage：放真正要处理的输入文本
        # 这里采用消息式输入，而不是把一切拼成单个字符串，是为了匹配 LangChain chat model 的调用习惯。
        payload = [SystemMessage(content=system_prompt), HumanMessage(content=content_str)]

        # 根据当前配置或 override provisioning 出实际使用的 LangChain 模型。
        # "transformation" 这个用途标签会影响模型选择和配置策略；
        # 同时这里把 max_tokens 提到 8192，说明 transformation 输出被允许比普通短回答更长。
        chain = await provision_langchain_model(
            str(payload),
            config.get("configurable", {}).get("model_id"),
            "transformation",
            max_tokens=8192,
        )

        # 真正调用模型执行 transformation。
        response = await chain.ainvoke(payload)

        # 先从模型响应里抽取出纯文本内容。
        # 有些 provider / LangChain 返回结构不是单纯 str，所以这里统一走 extract_text_content。
        response_content = extract_text_content(response.content)

        # 再清理 thinking / reasoning 痕迹。
        # 这样最终保存下来的 insight 更接近用户真正想看的结果文本，而不是模型内部推理碎片。
        cleaned_content = clean_thinking_content(response_content)

        if source:
            # 如果这次 transformation 是绑定在某个 source 上执行的，就把结果作为 insight 写回 source。
            # 这里调用的是 source.add_insight()，也就是把 transformation 输出持久化成系统中的派生内容，
            # 供后续 source 详情页、检索和进一步问答使用。
            await source.add_insight(transformation.title, cleaned_content)

        # graph 节点只返回本次转换产出的 output。
        # 是否持久化 insight 是“有 source 时的附加副作用”，但 graph 的直接输出仍然是文本结果本身。
        return {
            "output": cleaned_content,
        }
    except OpenNotebookError:
        # 已经是系统定义好的业务异常时，原样向上抛，避免丢失更准确的错误类型。
        raise
    except Exception as e:
        # 其余异常统一走分类器，转换成对用户更友好的领域错误。
        # 这样上层不会直接暴露底层 provider / LangChain / 网络层原始异常。
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


agent_state = StateGraph(TransformationState)
agent_state.add_node("agent", run_transformation)  # type: ignore[type-var]
agent_state.add_edge(START, "agent")
agent_state.add_edge("agent", END)
graph = agent_state.compile()
