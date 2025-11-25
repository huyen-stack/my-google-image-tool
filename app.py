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

GEMINI_MODEL_NAME = "gemini-flash-latest"  # 可按需替换
FREE_TIER_RPM_LIMIT = 10  # 免费版典型：1 分钟 10 次 generateContent

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "analysis_history" not in st.session_state:
    # 每条历史：
    # {
    #   "id": "run_1",
    #   "created_at": "...",
    #   "meta": {...},
    #   "data": {...}
    # }
    st.session_state["analysis_history"] = []


# ========================
# 页面样式
# ========================

st.set_page_config(
    page_title="AI 人物动作轨迹分析助手",
    page_icon="🦾",
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
        background: radial-gradient(circle at top left, #22c55e 0, #020617 55%, #020617 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    ">
      <h1 style="margin: 0 0 8px 0; color: #e5e7eb; font-size: 1.6rem;">
        🦾 AI 人物动作分析助手 · 只关注动作轨迹 + 镜头运动
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传视频或输入抖音 / B站 / TikTok / YouTube 链接，自动抽关键帧，
        <b>只分析人物动作、身体姿态、镜头运动 & 整段动作轨迹</b>，
        输出结构化 JSON + 中文动作拆解 + 英文整段动作提示词（适合 SORA / VEO）。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 工具函数
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
# 单帧：人物动作分析
# ========================

def analyze_action_single_frame(
    img: Image.Image,
    model,
    index: int,
) -> Dict[str, Any]:
    """
    单帧只分析：人物动作 + 身体姿态 + 镜头运动 + 动作趋势。
    """
    try:
        prompt = f"""
你现在是动作设计总监 + 电影导演 + 分镜统筹。
只专注于“人物动作”“身体姿态”“镜头运动”，不要展开冗长场景描述。

对给你的这一帧画面，输出一个 JSON 对象，字段如下（所有 key 必须出现）：

{{
  "index": {index},

  "scene_brief_zh": "用 1 句简短中文概括这个画面发生了什么（只点到人物正在做什么，不展开环境细节）",

  "character_action_detail_zh": (
    "用 1～3 句中文，从 头部 → 上肢 → 躯干 → 下肢 的顺序，具体描述人物此刻的身体姿态和动作："
    "1）重心在哪（前倾 / 后仰 / 蹲下 / 腾空 / 贴在物体表面等）；"
    "2）双手/手指在做什么（抓住什么、推、拉、挥动、抱头、举枪等）；"
    "3）双腿/脚的姿态（站立、迈步、腾空、跪地、脚尖朝向哪里）。"
  ),

  "face_expression_detail_zh": (
    "用 1～2 句中文写清：眉毛 / 眼睛 / 嘴角 / 下颌的状态，以及眼神的方向和情绪（紧张、专注、恐惧、轻松等）。"
  ),

  "cloth_hair_reaction_zh": (
    "用 1～2 句中文写清头发和衣服如何响应动作或风的惯性："
    "例如：长发向后甩起、衣摆被动作拖出残影、裙摆延迟摆动等。"
  ),

  "camera_movement_zh": (
    "用 1 句中文总结这一帧所在镜头的机位和运动方式："
    "例如：肩后跟拍向前冲、低机位仰拍人物落下、从右向左高速跟随横移、第一人称视角俯冲等。"
  ),

  "motion_trend_zh": (
    "用 1～2 句中文，用“上一瞬间 / 当前瞬间 / 下一瞬间”的逻辑，推测这一帧所在动作片段："
    "上一瞬间大概率是什么姿态；当前画面定格在什么极值；下一瞬间可能会发生什么（起跳 / 落地 / 转身 / 撞击等）。"
  ),

  "action_tags_zh": [
    "#短标签1",
    "#短标签2"
  ]
}}

要求：
1. 只输出一个 JSON 对象，不要任何解释或额外文字。
2. 所有字符串必须使用双引号，不要使用单引号。
3. JSON 中不能有注释，不能有多余的逗号。
"""
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start : end + 1]
        info = json.loads(json_str)

        # 兜底字段
        info["index"] = index
        info.setdefault("scene_brief_zh", "")
        info.setdefault("character_action_detail_zh", "")
        info.setdefault("face_expression_detail_zh", "")
        info.setdefault("cloth_hair_reaction_zh", "")
        info.setdefault("camera_movement_zh", "")
        info.setdefault("motion_trend_zh", "")
        info.setdefault("action_tags_zh", [])

        return info

    except Exception as e:
        return {
            "index": index,
            "scene_brief_zh": f"（AI 分析失败：{e}）",
            "character_action_detail_zh": "",
            "face_expression_detail_zh": "",
            "cloth_hair_reaction_zh": "",
            "camera_movement_zh": "",
            "motion_trend_zh": "",
            "action_tags_zh": [],
        }


def analyze_actions_concurrently(
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
    status.info(f"⚡ 正在对前 {use_n} 帧进行人物动作分析（共 {n} 帧）。")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(use_n, 6)) as executor:
        future_to_index = {
            executor.submit(analyze_action_single_frame, images[i], model, i + 1): i
            for i in range(use_n)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {
                    "index": i + 1,
                    "scene_brief_zh": f"（AI 分析失败：{e}）",
                    "character_action_detail_zh": "",
                    "face_expression_detail_zh": "",
                    "cloth_hair_reaction_zh": "",
                    "camera_movement_zh": "",
                    "motion_trend_zh": "",
                    "action_tags_zh": [],
                }

    # 后面的帧只做占位
    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_brief_zh": "（本帧未做 AI 动作分析，用于节省配额，仅保留画面参考。）",
            "character_action_detail_zh": "",
            "face_expression_detail_zh": "",
            "cloth_hair_reaction_zh": "",
            "camera_movement_zh": "",
            "motion_trend_zh": "",
            "action_tags_zh": [],
        }

    status.empty()
    return results


# ========================
# 整段人物动作概括
# ========================

def summarize_character_actions(
    frame_infos: List[Dict[str, Any]],
    model,
    frame_range: Optional[Tuple[int, int]] = None,
) -> str:
    """
    概括一段连续帧里人物的整体动作轨迹。
    frame_range:
        None: 使用所有帧
        (start, end): 使用第 start~end 帧（1-based，含端点）
    """
    if not frame_infos:
        return "（暂无关键帧，无法概括人物动作。）"

    n = len(frame_infos)
    if frame_range is None:
        start_idx, end_idx = 1, n
    else:
        start_idx, end_idx = frame_range
        start_idx = max(1, start_idx)
        end_idx = min(n, end_idx)
        if end_idx < start_idx:
            return "（帧区间不合法，无法概括人物动作。）"

    selected = frame_infos[start_idx - 1 : end_idx]

    described = []
    for info in selected:
        desc = info.get("scene_brief_zh", "") or ""
        if not desc:
            continue
        if "未做 AI 动作分析" in desc or "AI 分析失败" in desc:
            continue

        idx = info.get("index", "?")
        act = info.get("character_action_detail_zh", "") or ""
        trend = info.get("motion_trend_zh", "") or ""

        described.append(
            f"第 {idx} 帧：\n"
            f"- 画面简述：{desc}\n"
            f"- 人物动作：{act if act else '（暂无动作细节）'}\n"
            f"- 动作趋势：{trend if trend else '（暂无动作趋势）'}"
        )

    if not described:
        return "（当前选择的帧区间内没有有效的人物动作分析，无法生成概括。）"

    joined = "\n\n".join(described)

    prompt = f"""
你现在是动作设计总监 + 电影导演 + 分镜统筹。
下面是从一段视频中抽取的若干连续关键帧的人物动作说明，请你从“动作设计”的角度做整体概括。

=== 连续帧动作说明开始 ===
{joined}
=== 连续帧动作说明结束 ===

请严格按下面结构输出：

【人物动作整体概括】
用 2-4 句中文，从整体视角描述人物在这一段里完成了怎样的连续动作轨迹，
要说明起点、移动路径（例如从画面右侧高速冲到左下、腾空附着到机翼、再滑到安全位置）、
以及最后人物停留在怎样的姿态。

【动作阶段拆解】
用 3-6 行，按“起势 → 加速/腾空 → 关键动作 → 落地/收势”的顺序拆解，
每行前面加 1）、2）…，每行一句中文。

【SORA/VEO 用整段动作英文提示词】
用 2-4 句英文描述这一整段动作（人物外观可简要提一笔，重点写动作路径、镜头视角和机位运动），
最后一句写明时长，例如：
"8 second continuous action shot, vertical 9:16, 24fps, cinematic, highly detailed."

不要输出其他任何内容。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        msg = str(e)
        if "quota" in msg or "You exceeded your current quota" in msg:
            return "人物动作概括生成失败：当前 Gemini 免费额度每分钟调用次数已用完，请减少本次分析帧数或稍后再试。"
        return f"人物动作概括生成失败：{msg}"


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
    st.caption("建议：10 秒视频 6~10 帧就够分析动作连续性了。")

    st.markdown("---")
    st.markdown("⏱ 分析时间范围（单位：秒）")
    start_sec = st.number_input(
        "从第几秒开始（含）", min_value=0.0, value=0.0, step=0.5,
        help="精确到 0.5 秒；默认 0 表示从头开始"
    )
    end_sec = st.number_input(
        "到第几秒结束（0 或 ≤开始秒 表示直到结尾）",
        min_value=0.0, value=0.0, step=0.5,
        help="例如只看 3~8 秒的动作，就填 3 和 8；填 0 则分析到结尾"
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
# 主界面：视频来源选择
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

if st.button("🦾 开始分析人物动作"):
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
                overhead_calls = 1  # 整段动作概括
                max_ai_frames_safe = max(
                    1,
                    min(max_ai_frames, FREE_TIER_RPM_LIMIT - overhead_calls),
                )
                if max_ai_frames_safe < max_ai_frames:
                    st.info(
                        f"为避免触发免费额度限制，本次只对 **前 {max_ai_frames_safe} 帧** 做动作分析 "
                        f"（侧边栏设置为 {max_ai_frames} 帧）。"
                    )

                # 4. 帧级动作分析
                with st.spinner("🧠 正在分析每一帧的人物动作与姿态..."):
                    frame_infos = analyze_actions_concurrently(
                        images, model, max_ai_frames_safe
                    )

                # 5. 整段动作概括
                with st.spinner("🦾 正在整理整段人物动作轨迹..."):
                    action_summary = summarize_character_actions(
                        frame_infos, model, frame_range=None  # 默认使用所有帧
                    )

                # 6. 组装导出数据
                export_frames = []
                for info in frame_infos:
                    export_frames.append(
                        {
                            "index": info.get("index"),
                            "scene_brief_zh": info.get("scene_brief_zh", ""),
                            "character_action_detail_zh": info.get("character_action_detail_zh", ""),
                            "face_expression_detail_zh": info.get("face_expression_detail_zh", ""),
                            "cloth_hair_reaction_zh": info.get("cloth_hair_reaction_zh", ""),
                            "camera_movement_zh": info.get("camera_movement_zh", ""),
                            "motion_trend_zh": info.get("motion_trend_zh", ""),
                            "action_tags_zh": info.get("action_tags_zh", []),
                        }
                    )

                export_data = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "frame_count": len(images),
                        "max_ai_frames_this_run": max_ai_frames_safe,
                        "duration_sec_est": duration,
                        "start_sec_used": start_used,
                        "end_sec_used": end_used,
                        "source_type": source_type,
                        "source_label": source_label,
                    },
                    "frames": export_frames,
                    "character_action_summary": action_summary,
                }

                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

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

                # 7. 界面展示：帧卡片 + 整段概括 + JSON + 历史
                tab_frames, tab_summary, tab_json, tab_history = st.tabs(
                    [
                        "🎞 逐帧人物动作",
                        "🦾 整段动作轨迹概括",
                        "📦 JSON 导出（本次）",
                        "🕘 历史记录（本会话）",
                    ]
                )

                # --- Tab1：逐帧 ---
                with tab_frames:
                    st.markdown(
                        f"共抽取 **{len(images)}** 帧，其中前 **{min(len(images), max_ai_frames_safe)}** 帧做了动作分析。"
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
                                st.markdown("**画面简述（人物做了什么）：**")
                                st.code(
                                    info.get("scene_brief_zh", ""),
                                    language="markdown",
                                )

                                st.markdown("**人物动作拆解（可复制给分镜脚本）：**")
                                st.code(
                                    info.get("character_action_detail_zh")
                                    or "（暂无人物动作细节，可能未做 AI 分析）",
                                    language="markdown",
                                )

                                st.markdown("**面部与表情 / 眼神：**")
                                st.code(
                                    info.get("face_expression_detail_zh")
                                    or "（暂无面部表情描述）",
                                    language="markdown",
                                )

                                st.markdown("**服装与头发对动作/风的反应：**")
                                st.code(
                                    info.get("cloth_hair_reaction_zh")
                                    or "（暂无服装与头发反应描述）",
                                    language="markdown",
                                )

                                st.markdown("**镜头视角与运镜方式：**")
                                st.code(
                                    info.get("camera_movement_zh")
                                    or "（暂无镜头运动描述）",
                                    language="markdown",
                                )

                                st.markdown("**动作趋势（上一瞬间 / 下一瞬间）：**")
                                st.code(
                                    info.get("motion_trend_zh")
                                    or "（暂无动作趋势描述）",
                                    language="markdown",
                                )

                                tags = info.get("action_tags_zh") or []
                                if tags:
                                    st.markdown("**动作相关标签：**")
                                    st.code(" ".join(tags), language="markdown")

                            st.markdown("---")

                # --- Tab2：整段动作轨迹 ---
                with tab_summary:
                    st.markdown("### 🦾 整段人物动作轨迹（可直接丢给编导/分镜）")
                    st.code(action_summary, language="markdown")

                # --- Tab3：JSON 导出 ---
                with tab_json:
                    st.markdown("### 📦 下载本次动作分析 JSON")
                    st.download_button(
                        label="⬇️ 下载 character_actions.json",
                        data=json_str,
                        file_name="character_actions.json",
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
                            f"{h['meta'].get('frame_count',0)} 帧 | 区间 {h['meta'].get('start_sec_used',0):.1f}-{h['meta'].get('end_sec_used',0):.1f}s"
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
                            f"**模型：** {selected['meta'].get('model','')}"
                        )

                        st.markdown("#### 人物动作整体概括（该次）")
                        st.code(
                            selected["data"].get("character_action_summary", ""),
                            language="markdown",
                        )

                        frames = selected["data"].get("frames", [])
                        if frames:
                            st.markdown("#### 部分帧预览（人物动作 + 动作趋势）")
                            for f in frames[:3]:
                                st.markdown(f"**第 {f.get('index')} 帧：**")
                                st.write(f.get("scene_brief_zh", ""))
                                st.code(
                                    f.get("character_action_detail_zh", ""),
                                    language="markdown",
                                )
                                st.code(
                                    f.get("motion_trend_zh", ""),
                                    language="markdown",
                                )
                                st.markdown("---")

        except Exception as e:
            st.error(f"下载或解析视频时发生错误：{e}")
