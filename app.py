import streamlit as st
import google.generativeai as genai
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import concurrent.futures
import json
from datetime import datetime
import yt_dlp
from typing import Optional, Tuple, List, Dict, Any

# ========================
# 全局配置
# ========================

GEMINI_MODEL_NAME = "gemini-flash-latest"
FREE_TIER_RPM_LIMIT = 10  # 免费版典型：1 分钟 10 次 generateContent

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []


# ========================
# 页面样式
# ========================

st.set_page_config(
    page_title="AI 多实体运动 & 空中镜头分析助手",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #020617;
        color: #e5e7eb;
    }
    .stMarkdown, .stText {
        color: #e5e7eb;
    }
    .stCode {
        font-size: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        padding: 18px 24px;
        border-radius: 18px;
        margin-bottom: 16px;
        background: radial-gradient(circle at top left, #3b82f6 0, #020617 55%, #020617 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    ">
      <h1 style="margin: 0 0 8px 0; color: #e5e7eb; font-size: 1.6rem;">
        🎬 AI 多实体运动 & 空中镜头分析助手
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传视频或输入抖音 / B站 / TikTok / YouTube 链接，自动抽取关键帧，
        分析画面中<b>人物、车辆、运动物体</b>的动作关系，
        并可针对<b>无人机 / 穿越机 / 超人起飞 / 高楼跳下</b>镜头做专业飞行轨迹 + 运镜总结，
        同时生成适合 SORA / Veo 的英文提示词。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 工具函数：抽帧 & 下载
# ========================

def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 20,
    base_fps: float = 0.8,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Image.Image], float, Tuple[float, float]]:
    """
    根据视频时长，在指定时间范围内均匀抽帧。
    返回：
      images: PIL.Image 列表
      duration: 整条视频时长（秒）
      used_range: (start_used, end_used)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], 0.0, (0.0, 0.0)

    duration = total_frames / fps

    # 规范时间范围
    if start_sec is None or start_sec < 0:
        start_sec = 0.0
    if end_sec is None or end_sec <= start_sec or end_sec > duration:
        end_sec = duration

    start_frame = int(start_sec * fps)
    end_frame_excl = min(total_frames, int(end_sec * fps))
    segment_frames = end_frame_excl - start_frame

    # 如果区间非法，退回整段
    if segment_frames <= 0:
        start_sec = 0.0
        end_sec = duration
        start_frame = 0
        end_frame_excl = total_frames
        segment_frames = total_frames

    segment_duration = segment_frames / fps

    ideal_n = int(segment_duration * base_fps)
    target_n = max(min_frames, ideal_n)
    target_n = min(target_n, max_frames, segment_frames)

    if target_n <= 0:
        cap.release()
        return [], duration, (start_sec, end_sec)

    step = segment_frames / float(target_n)
    frame_indices = [start_frame + int(i * step) for i in range(target_n)]

    images: List[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
        else:
            images.append(Image.new("RGB", (200, 200), color="gray"))

    cap.release()
    return images, duration, (start_sec, end_sec)


def download_video_from_url(url: str) -> str:
    """用 yt-dlp 下载视频到临时文件，返回路径。"""
    if not url:
        raise ValueError("视频链接为空")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": tmp_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return tmp_path


def _extract_text_from_response(resp) -> str:
    """兼容不同 Gemini 返回结构，尽量拿到纯文本。"""
    text = getattr(resp, "text", None)
    if text and isinstance(text, str) and text.strip():
        return text.strip()

    try:
        texts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
        if texts:
            return " ".join(texts).strip()
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return ""


# ========================
# 单帧：多实体运动分析
# ========================

def analyze_motion_single_frame(
    img: Image.Image,
    model,
    index: int,
) -> Dict[str, Any]:
    """
    单帧分析：
    - moving_entities：包括人物 / 车辆 / 其他明显运动物体
    - 每个实体的动作简述、角色、方向、速度
    - 镜头运动 + 多实体互动 + 动作趋势
    """
    try:
        prompt = f"""
你现在是“动作设计总监 + 电影导演 + 特技协调 + 汽车特技指导”。
只专注画面中所有“在运动的东西”：人物、汽车、摩托车、自行车、球、飞行器、抛掷物等。

请对给你的这一帧画面输出一个 JSON 对象：

{{
  "index": {index},

  "scene_motion_brief_zh": "用 1 句中文概括本帧阶段的整体运动（例如：女主在屋顶奔跑的同时，红色跑车从下方街道高速掠过）",

  "moving_entities": [
    {{
      "id": "E1",
      "type": "person / car / motorbike / bicycle / object / animal / other",
      "visual_tag_zh": "简短中文标签，例如：黑衣女主、红色跑车、蓝色摩托、白色足球",
      "role_zh": "主体 / 反派 / 路人 / 被追逐目标 / 障碍物 等",
      "action_brief_zh": "用 1 句中文描述该实体此刻主要动作和方向，例如：沿着屋顶边缘向右高速奔跑",
      "screen_pos_hint_zh": "描述在画面中的大致位置，例如：画面左下 / 画面右侧偏上 / 居中偏下",
      "direction_zh": "运动或朝向的中文描述，例如：从左向右、从远处冲向镜头、向画面深处远离",
      "speed_zh": "缓慢 / 中速 / 高速 / 几乎静止"
    }}
  ],

  "camera_motion_zh": (
    "用 1 句中文描述镜头视角和机位运动，例如：肩后跟拍人物向前奔跑的中景；"
    "或：俯视跟随镜头从上方滑动观察两辆车的对向行驶；"
    "要提到：是跟拍哪一个实体、运动方向（例如从左向右滑动）、机位是高机位 / 低机位 / 平视。"
  ),

  "interaction_zh": (
    "用 1～3 句中文描述多个实体之间的关系，例如：红色跑车正在追逐前方黑色轿车；"
    "主角骑摩托从行人之间穿过；足球被球员踢向球门；"
    "如果基本没有互动，可以写“各实体之间几乎没有直接互动”。"
  ),

  "motion_trend_zh": (
    "用 1～2 句中文，从“上一瞬间 / 当前瞬间 / 下一瞬间”的角度，推测动作发展："
    "上一瞬间刚刚发生什么（起步 / 加速 / 转向），当前是哪个极值姿态或关键时刻，"
    "下一瞬间很可能发生什么（起跳、落地、碰撞、完成超车等）。"
  ),

  "motion_tags_zh": [
    "#追逐",
    "#高速漂移"
  ]
}}

要求：
1. 只输出一个 JSON 对象，不要任何解释或额外文字。
2. 所有字符串必须使用双引号，不要使用单引号。
3. JSON 中不能有注释，不能有多余的逗号。
4. 如果画面中只有一个运动实体，也要用 moving_entities 数组，只放一个对象。
"""
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start: end + 1]
        info = json.loads(json_str)

        # 兜底字段
        info["index"] = index
        info.setdefault("scene_motion_brief_zh", "")
        info.setdefault("moving_entities", [])
        info.setdefault("camera_motion_zh", "")
        info.setdefault("interaction_zh", "")
        info.setdefault("motion_trend_zh", "")
        info.setdefault("motion_tags_zh", [])

        return info

    except Exception as e:
        return {
            "index": index,
            "scene_motion_brief_zh": f"（AI 分析失败：{e}）",
            "moving_entities": [],
            "camera_motion_zh": "",
            "interaction_zh": "",
            "motion_trend_zh": "",
            "motion_tags_zh": [],
        }


def analyze_motions_concurrently(
    images: List[Image.Image],
    model,
    max_ai_frames: int,
) -> List[Dict[str, Any]]:
    """
    并发分析多帧，只跑前 max_ai_frames，其余帧做占位说明。
    """
    n = len(images)
    if n == 0:
        return []

    use_n = min(max_ai_frames, n)
    results: List[Dict[str, Any]] = [None] * n  # type: ignore

    status = st.empty()
    status.info(f"⚡ 正在对前 {use_n} 帧进行多实体运动分析（共 {n} 帧）。")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(use_n, 6)) as executor:
        future_to_index = {
            executor.submit(analyze_motion_single_frame, images[i], model, i + 1): i
            for i in range(use_n)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {
                    "index": i + 1,
                    "scene_motion_brief_zh": f"（AI 分析失败：{e}）",
                    "moving_entities": [],
                    "camera_motion_zh": "",
                    "interaction_zh": "",
                    "motion_trend_zh": "",
                    "motion_tags_zh": [],
                }

    # 后面的帧只做占位
    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_motion_brief_zh": "（本帧未做 AI 运动分析，用于节省配额，仅保留画面参考。）",
            "moving_entities": [],
            "camera_motion_zh": "",
            "interaction_zh": "",
            "motion_trend_zh": "",
            "motion_tags_zh": [],
        }

    status.empty()
    return results


# ========================
# 整段运动轨迹概括（综合版）
# ========================

def summarize_scene_motion(
    frame_infos: List[Dict[str, Any]],
    model,
    frame_range: Optional[Tuple[int, int]] = None,
) -> str:
    """
    综合场景：追逐 / 跑酷 / 车戏 等整段运动轨迹概括。
    """
    if not frame_infos:
        return "（暂无关键帧，无法概括整体运动轨迹。）"

    n = len(frame_infos)
    if frame_range is None:
        start_idx, end_idx = 1, n
    else:
        start_idx, end_idx = frame_range
        start_idx = max(1, start_idx)
        end_idx = min(n, end_idx)
        if end_idx < start_idx:
            return "（帧区间不合法，无法概括整体运动轨迹。）"

    selected = frame_infos[start_idx - 1: end_idx]

    described = []
    for info in selected:
        desc = info.get("scene_motion_brief_zh", "") or ""
        if not desc:
            continue
        if "未做 AI 运动分析" in desc or "AI 分析失败" in desc:
            continue

        idx = info.get("index", "?")
        ents = info.get("moving_entities", []) or []
        ent_briefs = []
        for e in ents:
            vt = e.get("visual_tag_zh", "") or ""
            ab = e.get("action_brief_zh", "") or ""
            if vt or ab:
                ent_briefs.append(f"{vt}：{ab}")
        ent_text = "；".join(ent_briefs) if ent_briefs else "（未提取到实体动作）"

        interaction = info.get("interaction_zh", "") or ""
        trend = info.get("motion_trend_zh", "") or ""

        described.append(
            f"第 {idx} 帧：\n"
            f"- 整体运动简述：{desc}\n"
            f"- 各实体动作：{ent_text}\n"
            f"- 实体之间互动：{interaction}\n"
            f"- 动作趋势：{trend}"
        )

    if not described:
        return "（当前选择的帧区间内没有有效的运动分析，无法生成概括。）"

    joined = "\n\n".join(described)

    prompt = f"""
你现在是动作设计总监 + 汽车特技协调 + 运动规划师。
下面是从一段视频中抽取的若干连续关键帧的“多实体运动说明”，包括人物、车辆、物体等。

=== 连续帧运动说明开始 ===
{joined}
=== 连续帧运动说明结束 ===

请严格按下面结构输出中文 + 英文分析：

【整体运动轨迹概括】
用 2-4 句中文，从宏观角度说明：
有哪些主要实体（例如：女主、红色跑车、黑色摩托、飞行无人机等），
他们在这一段里分别完成了怎样的运动路径（从哪里来、往哪里去），
整体是追逐、逃脱、超车、对撞还是配合等。

【关键事件（时间顺序）】
用 3-7 行，每行前面加 1）、2）…，
每行用中文写出一个关键“运动事件”，
例如：红色跑车从静止急加速冲出；女主从屋顶一跃而下落到车顶；两车在弯道处发生轻微碰撞等。

【运动层面的风格与节奏】
用 2-3 句中文总结这段运动的节奏感（慢 / 中 / 快）、是否有突然的爆发 / 刹停 / 漂移、
整体更偏写实动作还是夸张特效。

【SORA / Veo 用整段运动英文提示词】
用 3-6 句英文描述这一整段镜头，重点写：
1）有哪些实体，以及它们的视觉特征（例如 red sports car, female protagonist in black outfit）；
2）各实体的大致运动路径与相互关系（chasing, overtaking, collision 等）；
3）镜头视角和机位运动（low-angle tracking shot following the car, overhead shot revealing both lanes 等）；
4）整体时长与格式：最后一句类似
"8 second continuous action shot, vertical 9:16, 24fps, cinematic, highly detailed."

不要输出其他任何小节或解释。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        msg = str(e)
        if "quota" in msg or "You exceeded your current quota" in msg:
            return "整体运动轨迹概括生成失败：当前 Gemini 免费额度每分钟调用次数已用完，请减少本次分析帧数或稍后再试。"
        return f"整体运动轨迹概括生成失败：{msg}"


# ========================
# 空中 / 穿越机 / 超人专用总结
# ========================

def summarize_aerial_shot(
    frame_infos: List[Dict[str, Any]],
    model,
    frame_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    专门用来总结：无人机 / 穿越机 FPV / 超人飞行 / 高楼跳下 等“空中运动镜头”。
    返回：
      {
        "shot_category": "aerial_fpv_or_superhero",
        "aerial_summary_zh": "...",
        "aerial_meta": {...},
        "aerial_prompt_en": "..."
      }
    """
    if not frame_infos:
        return {
            "shot_category": "aerial_fpv_or_superhero",
            "aerial_summary_zh": "（暂无关键帧，无法概括空中镜头。）",
            "aerial_meta": {},
            "aerial_prompt_en": "",
        }

    n = len(frame_infos)
    if frame_range is None:
        start_idx, end_idx = 1, n
    else:
        start_idx, end_idx = frame_range
        start_idx = max(1, start_idx)
        end_idx = min(n, end_idx)
        if end_idx < start_idx:
            return {
                "shot_category": "aerial_fpv_or_superhero",
                "aerial_summary_zh": "（帧区间不合法，无法概括空中镜头。）",
                "aerial_meta": {},
                "aerial_prompt_en": "",
            }

    selected = frame_infos[start_idx - 1: end_idx]

    described = []
    for info in selected:
        desc = info.get("scene_motion_brief_zh", "") or ""
        if not desc:
            continue
        if "未做 AI 运动分析" in desc or "AI 分析失败" in desc:
            continue

        idx = info.get("index", "?")
        ents = info.get("moving_entities", []) or []
        ent_lines = []
        for e in ents:
            vt = e.get("visual_tag_zh", "") or ""
            etype = e.get("type", "") or ""
            act = e.get("action_brief_zh", "") or ""
            pos = e.get("screen_pos_hint_zh", "") or ""
            direction = e.get("direction_zh", "") or ""
            speed = e.get("speed_zh", "") or ""
            if vt or act:
                ent_lines.append(
                    f"- 实体：{vt}（类型：{etype}），动作：{act}，画面位置：{pos}，方向：{direction}，速度：{speed}"
                )

        camera = info.get("camera_motion_zh", "") or ""
        interaction = info.get("interaction_zh", "") or ""
        trend = info.get("motion_trend_zh", "") or ""

        described.append(
            f"第 {idx} 帧：\n"
            f"【整体运动简述】{desc}\n"
            f"【各实体运动】\n" + ("\n".join(ent_lines) if ent_lines else "（未提取到实体运动细节）") + "\n"
            f"【镜头运动】{camera}\n"
            f"【实体之间互动】{interaction}\n"
            f"【动作趋势】{trend}"
        )

    if not described:
        return {
            "shot_category": "aerial_fpv_or_superhero",
            "aerial_summary_zh": "（当前区间内没有有效的运动分析，无法生成空中镜头概括。）",
            "aerial_meta": {},
            "aerial_prompt_en": "",
        }

    joined = "\n\n".join(described)

    prompt = f"""
你现在是：顶级无人机 / 穿越机 FPV 机手 + 特技协调 + 电影摄影指导 + 分镜总监。

下面是一段空中/飞行相关镜头的若干关键帧的运动说明（包括人物、车辆、超人、穿越机等）：

=== 空中镜头帧级说明开始 ===
{joined}
=== 空中镜头帧级说明结束 ===

请帮我从“空中镜头设计”的角度，输出一个 JSON 对象，字段如下（所有 key 必须出现）：

{{
  "shot_category": "aerial_fpv_or_superhero",

  "aerial_summary_zh": "用 2～4 句中文，从宏观角度概括：这一段空中镜头里有哪些主体（人物/车/飞行器），它们在城市/山谷/室内外之间完成了怎样的飞行/坠落/上升路径，以及整体情绪（刺激/危险/自由/梦幻等）。",

  "aerial_meta": {{
    "flight_path_zh": "用 2～4 句中文描述飞行路径：从哪儿出发（地面/楼顶/室内），高度如何变化（贴地→半空→高空→再俯冲回地面），大致沿着什么路线（沿街道/顺着峡谷/围绕高楼转圈）。",
    "altitude_profile_zh": "专门用 1～2 句中文描述“高度曲线”：例如：一开始几乎贴地飞行，随后沿大楼外立面持续拉升到高空，再翻越楼顶俯冲回到街面高度。",
    "drone_attitude_zh": "用 2～3 句中文描述机体姿态：滚转（roll）、俯仰（pitch）、偏航（yaw）以及地平线是否长时间倾斜（典型 FPV 风格）。例如：机体长时间右侧倾约 30°，在俯冲瞬间快速由右倾切到左倾并伴随短暂翻滚。",
    "speed_profile_zh": "用 2～3 句中文描述速度曲线：起飞阶段从静止到高速、俯冲时的爆发加速、接近目标时的急剧减速、是否有慢动作片段（speed ramp）。",
    "proximity_events_zh": "用 2～4 句中文列出最重要的“擦边/穿越/险而又险”的瞬间：例如贴着楼体之间的窄缝飞过、从广告牌底下钻过去、穿破玻璃窗、高速掠过树梢，写清楚“离障碍物很近”的感觉。",
    "subject_lock_zh": "用 1～2 句中文描述镜头对主体的锁定方式：强锁人物/车辆（主体基本居中）、还是更偏向展示环境（主体只是画面中的一部分）；以及跟拍是贴背、贴侧、还是略领先半个身位。",
    "vertical_action_phases_zh": "如果有“高楼跳下 / 超人起飞 / 垂直飞行”，用 3～5 个阶段拆解：起势→腾空→自由落体或爬升→平飞/落地，并说明每个阶段镜头高度和视角如何变化。",
    "special_camera_moves_zh": "用 2～4 句中文描述特别的运镜技巧：甩镜转场（whip pan）、整圈滚转（barrel roll）、遮挡转场（贴近黑墙进入下一场景）、从跟车切换到跟人等。"
  }},

  "aerial_prompt_en": "用 3～6 句英文，写一段适合直接给 SORA / Veo 的空中镜头提示词：\\n1）先用 1 句点名主要主体和环境，例如：an FPV drone follows a black-clad female protagonist leaping off a skyscraper in a neon city...；\\n2）用 2～3 句详细描述飞行路径（height changes, path around/through buildings, near misses）和机体姿态（tilted horizon, roll, dive）；\\n3）最后一两句交代整体节奏和参数，例如：fast, immersive FPV shot with one brief slow-motion moment during the dive. 8 second continuous aerial shot, vertical 9:16, 24fps, cinematic, highly detailed."
}}

要求：
1. 只输出这一个 JSON 对象，不要任何解释或额外文字。
2. 所有字符串必须使用双引号，不要使用单引号。
3. JSON 中不能有注释，不能有多余的逗号。
"""
    try:
        resp = model.generate_content(prompt)
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start: end + 1]
        data = json.loads(json_str)

        data.setdefault("shot_category", "aerial_fpv_or_superhero")
        data.setdefault("aerial_summary_zh", "")
        data.setdefault("aerial_meta", {})
        data.setdefault("aerial_prompt_en", "")
        if not isinstance(data["aerial_meta"], dict):
            data["aerial_meta"] = {}

        return data

    except Exception as e:
        return {
            "shot_category": "aerial_fpv_or_superhero",
            "aerial_summary_zh": f"空中镜头概括失败：{e}",
            "aerial_meta": {},
            "aerial_prompt_en": "",
        }


# ========================
# 侧边栏：API Key & 参数
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 Gemini API Key")
    api_key = st.text_input(
        "输入 Google API Key",
        type="password",
        value=st.session_state["api_key"],
        help="粘贴你的 Gemini API Key（通常以 AIza 开头）",
    )
    st.session_state["api_key"] = api_key

    st.markdown("---")
    max_ai_frames = st.slider(
        "本次最多分析的关键帧数量（消耗配额）",
        min_value=4,
        max_value=20,
        value=10,
        step=1,
    )
    st.caption("建议：10 秒视频 6~10 帧就足够分析整体运动。")

    st.markdown("---")
    st.markdown("⏱ 分析时间范围（单位：秒）")
    start_sec = st.number_input(
        "从第几秒开始（含）", min_value=0.0, value=0.0, step=0.5,
        help="精确到 0.5 秒；默认 0 表示从头开始"
    )
    end_sec = st.number_input(
        "到第几秒结束（0 或 ≤开始秒 表示直到结尾）",
        min_value=0.0, value=0.0, step=0.5,
        help="例如只看 3~8 秒，就填 3 和 8；填 0 则分析到结尾"
    )

    st.markdown("---")
    shot_type = st.selectbox(
        "镜头类型 / 分析侧重点",
        [
            "默认：综合运动分析",
            "空中/穿越机/超人镜头（强化飞行轨迹）",
        ],
    )

    if not api_key:
        st.warning("🔴 还没有 Key，先去 https://ai.google.dev/ 申请一个。")
    else:
        st.success("🟢 Key 已就绪，可以分析。")


# ========================
# 初始化 Gemini 模型
# ========================

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"❌ 初始化 Gemini 模型失败：{e}")
        model = None


# ========================
# 主界面：视频来源
# ========================

source_mode = st.radio(
    "📥 选择视频来源",
    ["上传本地文件", "输入网络视频链接（抖音 / B站 / TikTok / YouTube）"],
    index=0,
)

video_url: Optional[str] = None
uploaded_file = None

if source_mode == "上传本地文件":
    uploaded_file = st.file_uploader(
        "📂 上传视频文件（建议 < 50MB）",
        type=["mp4", "mov", "m4v", "avi", "mpeg"],
    )
else:
    video_url = st.text_input(
        "🔗 输入视频链接",
        placeholder="例如：https://v.douyin.com/xxxxxx 或 https://www.douyin.com/video/xxxxxxxxx",
    )

# ========================
# 主按钮逻辑
# ========================

if st.button("🚀 开始分析视频运动与运镜"):
    if not api_key or model is None:
        st.error("请先在左侧输入有效的 Google API Key。")
    else:
        tmp_path: Optional[str] = None
        source_label = ""
        source_type = ""

        try:
            # 1. 准备视频
            if source_mode == "上传本地文件":
                source_type = "upload"
                if not uploaded_file:
                    st.error("请先上传一个视频文件。")
                    st.stop()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                source_label = uploaded_file.name
            else:
                source_type = "url"
                if not video_url:
                    st.error("请输入一个有效的视频链接。")
                    st.stop()
                st.info("🌐 正在从网络下载视频...")
                tmp_path = download_video_from_url(video_url)
                source_label = video_url

            if not tmp_path:
                st.error("视频路径异常，请重试。")
            else:
                # 2. 抽帧
                st.info("⏳ 正在抽取关键帧...")
                images, duration, used_range = extract_keyframes_dynamic(
                    tmp_path,
                    start_sec=start_sec,
                    end_sec=end_sec if end_sec > 0 else None,
                )
                start_used, end_used = used_range

                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

                if not images:
                    st.error("❌ 无法从视频中读取帧，请检查视频是否损坏或格式异常。")
                    st.stop()

                st.success(
                    f"✅ 已抽取 {len(images)} 个关键帧（视频总长约 {duration:.1f} 秒，"
                    f"本次分析区间：{start_used:.1f}–{end_used:.1f} 秒）。"
                )

                # 3. 控制本次 AI 调用次数（1 次整段总结 + 多帧分析）
                overhead_calls = 1  # 整段概括（综合 or 空中）
                max_ai_frames_safe = max(
                    1,
                    min(max_ai_frames, FREE_TIER_RPM_LIMIT - overhead_calls),
                )
                if max_ai_frames_safe < max_ai_frames:
                    st.info(
                        f"为避免触发免费额度限制，本次只对 **前 {max_ai_frames_safe} 帧** 做运动分析 "
                        f"（侧边栏设置为 {max_ai_frames} 帧）。"
                    )

                # 4. 帧级运动分析
                with st.spinner("🧠 正在分析每一帧的多实体运动..."):
                    frame_infos = analyze_motions_concurrently(
                        images, model, max_ai_frames_safe
                    )

                # 5. 整段概括：根据镜头类型选择不同策略
                motion_summary: Optional[str] = None
                aerial_result: Optional[Dict[str, Any]] = None

                if shot_type == "空中/穿越机/超人镜头（强化飞行轨迹）":
                    with st.spinner("🛩 正在整理空中/穿越机/超人镜头的飞行轨迹与机位设计..."):
                        aerial_result = summarize_aerial_shot(
                            frame_infos, model, frame_range=None
                        )
                else:
                    with st.spinner("🎮 正在整理整段运动轨迹与关键事件..."):
                        motion_summary = summarize_scene_motion(
                            frame_infos, model, frame_range=None
                        )

                # 6. 组装导出数据
                export_frames = []
                for info in frame_infos:
                    export_frames.append(
                        {
                            "index": info.get("index"),
                            "scene_motion_brief_zh": info.get("scene_motion_brief_zh", ""),
                            "moving_entities": info.get("moving_entities", []),
                            "camera_motion_zh": info.get("camera_motion_zh", ""),
                            "interaction_zh": info.get("interaction_zh", ""),
                            "motion_trend_zh": info.get("motion_trend_zh", ""),
                            "motion_tags_zh": info.get("motion_tags_zh", []),
                        }
                    )

                export_data: Dict[str, Any] = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "frame_count": len(images),
                        "max_ai_frames_this_run": max_ai_frames_safe,
                        "duration_sec_est": duration,
                        "start_sec_used": start_used,
                        "end_sec_used": end_used,
                        "source_type": source_type,
                        "source_label": source_label,
                        "shot_type": "aerial" if aerial_result else "general",
                    },
                    "frames": export_frames,
                    "overall_motion_summary": motion_summary if motion_summary else "",
                    "aerial_analysis": aerial_result if aerial_result else {},
                }

                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

                # 写入历史
                history = st.session_state["analysis_history"]
                run_id = f"run_{len(history) + 1}"
                history.append(
                    {
                        "id": run_id,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "meta": export_data["meta"],
                        "data": export_data,
                    }
                )
                st.session_state["analysis_history"] = history

                # 7. Tabs 展示
                tab_frames, tab_summary, tab_json, tab_history = st.tabs(
                    [
                        "🎞 逐帧多实体运动",
                        "🎯 整段总结（运动 / 空中镜头）",
                        "📦 JSON 导出（本次）",
                        "🕘 历史记录（本会话）",
                    ]
                )

                # --- Tab1：逐帧 ---
                with tab_frames:
                    st.markdown(
                        f"共抽取 **{len(images)}** 帧，其中前 **{min(len(images), max_ai_frames_safe)}** 帧做了 AI 运动分析。"
                    )
                    st.markdown("---")

                    for i, (img, info) in enumerate(zip(images, frame_infos)):
                        with st.container():
                            st.markdown(f"### 🎞 第 {i + 1} 帧")

                            c1, c2 = st.columns([1.2, 2])

                            with c1:
                                st.image(
                                    img,
                                    caption=f"第 {i + 1} 帧画面",
                                    use_column_width=True,
                                )

                            with c2:
                                st.markdown("**本帧整体运动简述：**")
                                st.code(
                                    info.get("scene_motion_brief_zh", ""),
                                    language="markdown",
                                )

                                entities = info.get("moving_entities", []) or []
                                if entities:
                                    st.markdown("**运动实体列表（可复制到分镜 / 提示词）：**")
                                    lines = []
                                    for e in entities:
                                        eid = e.get("id", "")
                                        etype = e.get("type", "")
                                        vtag = e.get("visual_tag_zh", "")
                                        role = e.get("role_zh", "")
                                        act = e.get("action_brief_zh", "")
                                        pos = e.get("screen_pos_hint_zh", "")
                                        direction = e.get("direction_zh", "")
                                        speed = e.get("speed_zh", "")
                                        lines.append(
                                            f"- [{eid}] ({etype}) {vtag}｜角色：{role}｜动作：{act}｜位置：{pos}｜方向：{direction}｜速度：{speed}"
                                        )
                                    st.code("\n".join(lines), language="markdown")
                                else:
                                    st.info("本帧未识别出明显的运动实体（或未做 AI 分析）。")

                                st.markdown("**镜头视角与机位运动：**")
                                st.code(
                                    info.get("camera_motion_zh")
                                    or "（暂无镜头运动描述）",
                                    language="markdown",
                                )

                                st.markdown("**实体之间互动关系：**")
                                st.code(
                                    info.get("interaction_zh")
                                    or "（暂无互动描述）",
                                    language="markdown",
                                )

                                st.markdown("**动作趋势（上一瞬间 / 下一瞬间）：**")
                                st.code(
                                    info.get("motion_trend_zh")
                                    or "（暂无动作趋势描述）",
                                    language="markdown",
                                )

                                tags = info.get("motion_tags_zh") or []
                                if tags:
                                    st.markdown("**运动相关标签：**")
                                    st.code(" ".join(tags), language="markdown")

                            st.markdown("---")

                # --- Tab2：整段总结 ---
                with tab_summary:
                    if aerial_result:
                        st.markdown("### 🛩 空中 / 穿越机 / 超人镜头概括（中文）")
                        st.code(
                            aerial_result.get("aerial_summary_zh", ""),
                            language="markdown",
                        )

                        st.markdown("### 📐 飞行路径 / 高度曲线 / 机体姿态 等细节")
                        meta = aerial_result.get("aerial_meta", {}) or {}
                        meta_text_lines = []
                        for k, v in meta.items():
                            if not v:
                                continue
                            meta_text_lines.append(f"【{k}】\n{v}\n")
                        st.code(
                            "\n".join(meta_text_lines)
                            or "（暂无更细致的空中镜头元信息）",
                            language="markdown",
                        )

                        st.markdown("### 🎥 空中镜头英文提示词（直接给 SORA / Veo）")
                        st.code(
                            aerial_result.get("aerial_prompt_en", "")
                            or "（暂无空中镜头英文提示词）",
                            language="markdown",
                        )
                    else:
                        st.markdown("### 🎮 整段运动轨迹 + 关键事件 + 英文提示词")
                        st.code(
                            motion_summary or "（暂无整段运动轨迹概括）",
                            language="markdown",
                        )

                # --- Tab3：JSON 导出 ---
                with tab_json:
                    st.markdown("### 📦 下载本次运动分析 JSON")
                    st.download_button(
                        label="⬇️ 下载 motion_analysis.json",
                        data=json_str,
                        file_name="motion_analysis.json",
                        mime="application/json",
                    )

                    with st.expander("🔍 预览部分 JSON 内容"):
                        preview = json_str[:3000] + (
                            "\n...\n" if len(json_str) > 3000 else ""
                        )
                        st.code(preview, language="json")

                # --- Tab4：历史记录 ---
                with tab_history:
                    st.markdown("### 🕘 当前会话历史记录（刷新页面会清空）")

                    history = st.session_state.get("analysis_history", [])
                    if not history:
                        st.info("当前会话还没有任何历史记录。")
                    else:
                        options = [
                            f"{len(history) - i}. {h['created_at']} | {h['meta'].get('source_label','')} | "
                            f"{h['meta'].get('frame_count',0)} 帧 | 区间 {h['meta'].get('start_sec_used',0):.1f}-{h['meta'].get('end_sec_used',0):.1f}s | "
                            f"类型：{h['meta'].get('shot_type','')}"
                            for i, h in enumerate(reversed(history))
                        ]
                        idx_display = st.selectbox(
                            "选择一条历史记录查看",
                            options=list(range(len(history))),
                            format_func=lambda i: options[i],
                        )
                        real_index = len(history) - 1 - idx_display
                        selected = history[real_index]

                        st.markdown(
                            f"**ID：** `{selected['id']}`  \n"
                            f"**时间：** {selected['created_at']}  \n"
                            f"**来源类型：** {selected['meta'].get('source_type','')}  \n"
                            f"**来源标识：** {selected['meta'].get('source_label','')}  \n"
                            f"**分析区间：** {selected['meta'].get('start_sec_used',0):.1f}–{selected['meta'].get('end_sec_used',0):.1f} 秒  \n"
                            f"**帧数：** {selected['meta'].get('frame_count',0)}  \n"
                            f"**镜头类型：** {selected['meta'].get('shot_type','')}  \n"
                            f"**模型：** {selected['meta'].get('model','')}"
                        )

                        st.markdown("#### 整段总结预览")
                        data = selected["data"]
                        if data.get("aerial_analysis"):
                            st.markdown("**空中镜头概括（中文）：**")
                            st.code(
                                data["aerial_analysis"].get("aerial_summary_zh", ""),
                                language="markdown",
                            )
                            st.markdown("**空中镜头英文提示词：**")
                            st.code(
                                data["aerial_analysis"].get("aerial_prompt_en", ""),
                                language="markdown",
                            )
                        else:
                            st.markdown("**整段运动轨迹（综合版）：**")
                            st.code(
                                data.get("overall_motion_summary", ""),
                                language="markdown",
                            )

                        frames = data.get("frames", [])
                        if frames:
                            st.markdown("#### 部分帧预览（运动实体 + 互动）")
                            for f in frames[:3]:
                                st.markdown(f"**第 {f.get('index')} 帧：**")
                                st.write(f.get("scene_motion_brief_zh", ""))
                                entities = f.get("moving_entities", []) or []
                                if entities:
                                    lines = []
                                    for e in entities:
                                        vt = e.get("visual_tag_zh", "")
                                        act = e.get("action_brief_zh", "")
                                        lines.append(f"- {vt}：{act}")
                                    st.code("\n".join(lines), language="markdown")
                                st.code(
                                    f.get("interaction_zh", ""),
                                    language="markdown",
                                )
                                st.markdown("---")

        except Exception as e:
            st.error(f"下载或解析视频时发生错误：{e}")
