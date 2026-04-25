import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.config import DATA_FOLDER
from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.podcasts.models import (
    EpisodeProfile,
    PodcastEpisode,
    SpeakerProfile,
    _resolve_model_config,
)

try:
    from podcast_creator import configure, create_podcast
except ImportError as e:
    logger.error(f"Failed to import podcast_creator: {e}")
    raise ValueError("podcast_creator library not available")


def build_episode_output_dir(data_folder: str) -> tuple[str, Path]:
    """Build a filesystem-safe output directory path for a podcast episode.

    Uses a UUID as the directory name so the path is safe regardless of
    what the user typed as episode name (spaces, special chars, etc.).

    Returns:
        A tuple of (episode_dir_name, output_dir_path).
    """
    episode_dir_name = str(uuid.uuid4())
    output_dir = Path(f"{data_folder}/podcasts/episodes/{episode_dir_name}")
    return episode_dir_name, output_dir


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class PodcastGenerationInput(CommandInput):
    episode_profile: str
    speaker_profile: str
    episode_name: str
    content: str
    briefing_suffix: Optional[str] = None


class PodcastGenerationOutput(CommandOutput):
    success: bool
    episode_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    transcript: Optional[dict] = None
    outline: Optional[dict] = None
    processing_time: float
    error_message: Optional[str] = None


@command("generate_podcast", app="open_notebook", retry={"max_attempts": 1})
async def generate_podcast_command(
    input_data: PodcastGenerationInput,
) -> PodcastGenerationOutput:
    """
    Real podcast generation using podcast-creator library with Episode Profiles
    """
    # 记录整条播客生成命令的起始时间。
    # 这里统计的是从加载 profile、创建 episode 记录，到 podcast-creator 生成完成的总耗时。
    start_time = time.time()

    try:
        # 先打入口日志，方便把这次后台播客任务和 episode_name / profile 对上。
        # 播客生成是异步后台任务，排查问题时通常先靠这组日志串起执行链路。
        logger.info(
            f"Starting podcast generation for episode: {input_data.episode_name}"
        )
        logger.info(f"Using episode profile: {input_data.episode_profile}")

        # 1. 先按名称加载 EpisodeProfile。
        # 这里用 profile name 而不是 ID，是因为前端/API 侧传递和展示的主要是 profile 名称；
        # 命令层负责把这个对用户友好的标识解析成数据库对象。
        episode_profile = await EpisodeProfile.get_by_name(input_data.episode_profile)
        if not episode_profile:
            raise ValueError(
                f"Episode profile '{input_data.episode_profile}' not found"
            )

        # 再通过 episode profile 上配置的 speaker_config 找到 SpeakerProfile。
        # 这说明播客生成不是由请求显式传入“完整 speaker 配置”，而是通过 episode profile
        # 间接绑定一套 speaker profile，从而让“单集格式”和“说话人配置”形成可复用组合。
        speaker_profile = await SpeakerProfile.get_by_name(
            episode_profile.speaker_config
        )
        if not speaker_profile:
            raise ValueError(
                f"Speaker profile '{episode_profile.speaker_config}' not found"
            )

        logger.info(f"Loaded episode profile: {episode_profile.name}")
        logger.info(f"Loaded speaker profile: {speaker_profile.name}")

        # 2. 在真正开始生成之前，先确认 profile 里关键模型字段都已经配置。
        # 这里提前失败的原因很直接：如果 outline / transcript / voice model 本身缺失，
        # 后面再跑到 podcast-creator 内部才报错只会让问题更难定位。
        if not episode_profile.outline_llm:
            raise ValueError(
                f"Episode profile '{episode_profile.name}' has no outline model configured. "
                "Please update the profile to select an outline model."
            )
        if not episode_profile.transcript_llm:
            raise ValueError(
                f"Episode profile '{episode_profile.name}' has no transcript model configured. "
                "Please update the profile to select a transcript model."
            )
        if not speaker_profile.voice_model:
            raise ValueError(
                f"Speaker profile '{speaker_profile.name}' has no voice model configured. "
                "Please update the profile to select a voice model."
            )

        # 3. 把模型注册表里的记录解析成真正可调用的 provider/model/config 三元组。
        # 这样后面传给 podcast-creator 的就不再是 Open Notebook 内部的模型 ID，
        # 而是可以直接用来调用上游 AI 服务的配置。
        outline_provider, outline_model_name, outline_config = (
            await episode_profile.resolve_outline_config()
        )
        transcript_provider, transcript_model_name, transcript_config = (
            await episode_profile.resolve_transcript_config()
        )
        tts_provider, tts_model_name, tts_config = (
            await speaker_profile.resolve_tts_config()
        )

        logger.info(
            f"Resolved models - outline: {outline_provider}/{outline_model_name}, "
            f"transcript: {transcript_provider}/{transcript_model_name}, "
            f"tts: {tts_provider}/{tts_model_name}"
        )

        # 4. 加载所有 episode / speaker profiles，而不只是当前这两个。
        # 这是因为 podcast-creator 的 configure() 接口接收的是整套 profiles 配置字典，
        # 并且它内部会对 profiles 做统一校验，不是只消费当前选中的单个 profile。
        episode_profiles = await repo_query("SELECT * FROM episode_profile")
        speaker_profiles = await repo_query("SELECT * FROM speaker_profile")

        # 把数据库返回的数组转成按名称索引的字典，适配 podcast-creator 期望的配置格式。
        # 之所以用 name 作为 key，是因为后面 create_podcast() 也是按 profile name 引用配置。
        episode_profiles_dict = {
            profile["name"]: profile for profile in episode_profiles
        }
        speaker_profiles_dict = {
            profile["name"]: profile for profile in speaker_profiles
        }

        # 5. 把解析后的 provider/model/config 注入到 profile 字典里。
        # 这里不是只补当前 episode_profile，而是补全所有 profile：
        # 因为 podcast-creator 会校验整套配置，如果有其他 profile 配置损坏，也可能拖累当前任务。
        # 对无法解析的 profile，当前策略是直接从配置字典里移除，避免整批配置校验失败。
        for ep_name in list(episode_profiles_dict.keys()):
            ep_dict = episode_profiles_dict[ep_name]
            try:
                if ep_dict.get("outline_llm"):
                    prov, model, conf = await _resolve_model_config(
                        str(ep_dict["outline_llm"])
                    )
                    ep_dict["outline_provider"] = prov
                    ep_dict["outline_model"] = model
                    ep_dict["outline_config"] = conf
                if ep_dict.get("transcript_llm"):
                    prov, model, conf = await _resolve_model_config(
                        str(ep_dict["transcript_llm"])
                    )
                    ep_dict["transcript_provider"] = prov
                    ep_dict["transcript_model"] = model
                    ep_dict["transcript_config"] = conf
            except Exception as e:
                logger.warning(
                    f"Failed to resolve models for episode profile '{ep_name}', "
                    f"removing from config to prevent validation errors: {e}"
                )
                del episode_profiles_dict[ep_name]

        # 对所有 speaker profiles 也做同样的 TTS 配置解析。
        # 这里连“当前任务未必会用到”的 profiles 也一起处理，理由同上：外部库校验的是整套配置。
        # 解析失败的 speaker profile 会被整体移除，防止它把当前任务拖垮。
        for sp_name in list(speaker_profiles_dict.keys()):
            sp_dict = speaker_profiles_dict[sp_name]
            if sp_dict.get("voice_model"):
                try:
                    prov, model, conf = await _resolve_model_config(
                        str(sp_dict["voice_model"])
                    )
                    sp_dict["tts_provider"] = prov
                    sp_dict["tts_model"] = model
                    sp_dict["tts_config"] = conf
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve TTS for speaker profile '{sp_name}', "
                        f"removing from config to prevent validation errors: {e}"
                    )
                    del speaker_profiles_dict[sp_name]
                    continue

            # 另外再处理 speaker profile 内每个 speaker 的单独 TTS 覆盖配置。
            # 这说明系统支持 profile 级默认语音模型，同时也允许单个角色覆盖默认值。
            for speaker in sp_dict.get("speakers", []):
                if speaker.get("voice_model"):
                    try:
                        prov, model, conf = await _resolve_model_config(
                            str(speaker["voice_model"])
                        )
                        speaker["tts_provider"] = prov
                        speaker["tts_model"] = model
                        speaker["tts_config"] = conf
                    except Exception as e:
                        logger.warning(
                            f"Failed to resolve per-speaker TTS for '{speaker.get('name')}': {e}"
                        )

        # 6. 生成 briefing。
        # episode_profile.default_briefing 提供这类节目的默认创作指令；
        # 如果这次请求还带了 briefing_suffix，就把它追加进去，让单次生成可以在模板基础上再加临时要求。
        briefing = episode_profile.default_briefing
        if input_data.briefing_suffix:
            briefing += f"\n\nAdditional instructions: {input_data.briefing_suffix}"

        # 在真正调用 podcast-creator 前，先创建一条 PodcastEpisode 记录。
        # 这样做的原因是：
        # 1. 后台任务开始后，前端可以立刻看到一条“生成中”的 episode 记录；
        # 2. 把 command_id 提前回写进去后，UI 才能轮询任务状态；
        # 3. 即使生成中途失败，也能留下可追踪的 episode 实体，而不是完全无记录。
        # Create the record for the episode and associate with the ongoing command
        episode = PodcastEpisode(
            name=input_data.episode_name,
            episode_profile=full_model_dump(episode_profile.model_dump()),
            speaker_profile=full_model_dump(speaker_profile.model_dump()),
            command=ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None,
            briefing=briefing,
            content=input_data.content,
            audio_file=None,
            transcript=None,
            outline=None,
        )
        await episode.save()

        # 把整理好的 profiles 配置注入 podcast-creator 的全局配置入口。
        # 当前外部库通过 configure() 接收 profile 集合，所以这里先把说话人和单集配置都装进去，
        # 后面 create_podcast() 才能按名称引用它们。
        configure("speakers_config", {"profiles": speaker_profiles_dict})
        configure("episode_config", {"profiles": episode_profiles_dict})

        logger.info("Configured podcast-creator with episode and speaker profiles")

        # 记录 briefing 长度，方便定位“为什么这次播客输入 unusually 大/小”。
        logger.info(f"Generated briefing (length: {len(briefing)} chars)")

        # 7. 为这次 episode 创建输出目录。
        # 这里故意不用用户输入的 episode_name 直接当目录名，而是用 UUID：
        # 这样可以规避空格、特殊字符、超长路径、重名覆盖等文件系统问题。
        episode_dir_name, output_dir = build_episode_output_dir(DATA_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directory: {output_dir}")

        # 8. 调用 podcast-creator 执行真正的播客生成。
        # 这里传入的核心输入包括：
        # - content：原始素材内容
        # - briefing：本次节目的创作约束
        # - speaker_config / episode_profile：按名称引用上面刚注入的配置
        # - output_dir：音频和中间产物输出目录
        # episode_name 这里传 UUID 目录名，而不是展示名，目的是让外部库生成的文件路径保持安全稳定。
        logger.info("Starting podcast generation with podcast-creator...")

        result = await create_podcast(
            content=input_data.content,
            briefing=briefing,
            episode_name=episode_dir_name,
            output_dir=str(output_dir),
            speaker_config=speaker_profile.name,
            episode_profile=episode_profile.name,
        )

        # 生成完成后，把最终产物路径和结构化结果写回 episode 记录。
        # 这一步把“后台任务的执行结果”转成系统里可查询、可展示的持久化实体字段。
        episode.audio_file = (
            str(result.get("final_output_file_path")) if result else None
        )
        episode.transcript = {
            "transcript": full_model_dump(result["transcript"]) if result else None
        }
        episode.outline = full_model_dump(result["outline"]) if result else None
        await episode.save()

        # 成功出口统一计算总耗时。
        processing_time = time.time() - start_time
        logger.info(
            f"Successfully generated podcast episode: {episode.id} in {processing_time:.2f}s"
        )

        # 返回的是后台命令摘要 + 关键产物引用，不是重新包装一整份 episode 对象。
        # 这样命令输出保持轻量，而完整详情仍然以 PodcastEpisode 持久化记录为准。
        return PodcastGenerationOutput(
            success=True,
            episode_id=str(episode.id),
            audio_file_path=str(result.get("final_output_file_path"))
            if result
            else None,
            transcript={"transcript": full_model_dump(result["transcript"])}
            if result.get("transcript")
            else None,
            outline=full_model_dump(result["outline"])
            if result.get("outline")
            else None,
            processing_time=processing_time,
        )

    except ValueError:
        # 配置缺失、profile 不存在等业务错误直接原样抛出。
        # 这类问题属于永久性失败，重试不会自动解决。
        raise

    except Exception as e:
        # 其他异常统一视为生成流程失败，记录完整日志并包装成 RuntimeError 向上抛。
        # 这样命令框架会把它视为失败任务，而不是静默吞掉。
        logger.error(f"Podcast generation failed: {e}")
        logger.exception(e)

        error_msg = str(e)
        # 对 GPT-5 系列常见的 JSON 解析失败补充一个更有操作性的提示。
        # 这是根据当前项目里遇到的实际失败模式做的定向提示，帮助用户更快切换到更适合的模型。
        if "Invalid json output" in error_msg or "Expecting value" in error_msg:
            error_msg += (
                "\n\nNOTE: This error commonly occurs with GPT-5 models that use extended thinking. "
                "The model may be putting all output inside <think> tags, leaving nothing to parse. "
                "Try using gpt-4o, gpt-4o-mini, or gpt-4-turbo instead in your episode profile."
            )

        raise RuntimeError(error_msg) from e
