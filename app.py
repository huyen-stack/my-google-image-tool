import streamlit as st
import requests
import base64
from io import BytesIO

# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="OpenAI 多功能图像生成器（高级版）",
    page_icon="🎨",
    layout="wide",
)

# 简单一点的全局样式
st.markdown(
    """
    <style>
    .small-text {font-size: 12px; color: #888;}
    .stButton>button {border-radius: 999px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎨 OpenAI 多功能图像生成器（高级版）")
st.caption("支持：多图生成｜画风选择｜高清尺寸｜提示词优化｜图像→文字描述")

# =========================
# Sidebar：基础设置
# =========================
with st.sidebar:
    st.header("🔑 API Key 设置")
    openai_key = st.text_input("OpenAI API Key（sk- 开头）", type="password")
    st.markdown('<div class="small-text">前往 platform.openai.com 生成，注意不要泄露。</div>', unsafe_allow_html=True)

    st.divider()
    st.header("🎨 画风选择")
    style_choice = st.selectbox(
        "风格",
        [
            "default（默认）",
            "anime（动漫）",
            "realistic（写实）",
            "cyberpunk（赛博）",
            "oil painting（油画）",
            "comic（漫画）",
        ],
    )

    st.divider()
    st.header("🖼 生成数量 & 尺寸")
    num_images = st.slider("生成图片数量", 1, 4, 1)

    size = st.selectbox(
        "尺寸（越大越高清）",
        [
            "1024x1024（标准方图）",
            "1024x1536（竖版高清）",
            "1536x1024（横版高清）",
        ],
    )

    # 真正传给 API 的 size 字符串
    size_map = {
        "1024x1024（标准方图）": "1024x1024",
        "1024x1536（竖版高清）": "1024x1536",
        "1536x1024（横版高清）": "1536x1024",
    }
    api_size = size_map[size]

# =========================
# 帮助函数：OpenAI Chat / Vision / Images
# =========================

CHAT_URL = "https://api.openai.com/v1/chat/completions"
IMAGE_URL = "https://api.openai.com/v1/images/generations"
HEADERS = lambda key: {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}


def optimize_prompt(raw_prompt: str, key: str) -> str:
    """调用 gpt-4o-mini，帮用户优化绘画提示词（英文）。"""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional text-to-image prompt engineer. "
                           "Rewrite the user's request as a single, detailed English prompt for high-quality image generation.",
            },
            {"role": "user", "content": raw_prompt},
        ],
        "temperature": 0.8,
    }
    resp = requests.post(CHAT_URL, headers=HEADERS(key), json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def describe_image(img_bytes: bytes, key: str) -> str:
    """使用 gpt-4o-mini Vision，对图片进行中文描述。"""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用简洁、自然的中文详细描述这张图片的内容和风格。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.4,
    }
    resp = requests.post(CHAT_URL, headers=HEADERS(key), json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_images(prompt: str, key: str, size: str, n: int):
    """调用 gpt-image-1 生成图片。"""
    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": n,
        "size": size,          # 例如 "1024x1024"
        # 不再传 response_format，默认返回 b64_json
    }
    resp = requests.post(IMAGE_URL, headers=HEADERS(key), json=payload)
    resp.raise_for_status()
    return resp.json()


def image_download_button(img_bytes: bytes, filename: str, key: str):
    """绘制一个 PNG 下载按钮。"""
    st.download_button(
        "📥 下载 PNG",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        key=key,
    )


# =========================
# 主界面布局
# =========================

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 文本提示词 → 生成图片")
    prompt = st.text_area("请输入你想生成的画面描述：", height=150, placeholder="例如：在木桌前微笑举杯的中国女孩，暖色调，家庭聚餐氛围……")

with col_right:
    st.subheader("🖼 图像 → 文本描述")
    uploaded_image = st.file_uploader("上传图片（PNG / JPG）", type=["png", "jpg", "jpeg"])

# 操作按钮区
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    btn_optimize = st.button("✨ 优化提示词")
with btn_col2:
    btn_generate = st.button("🎨 生成图片")

st.divider()

# =========================
# 处理：提示词优化
# =========================
if btn_optimize:
    if not openai_key:
        st.error("请先在左侧输入 OpenAI API Key。")
    elif not prompt.strip():
        st.error("请先输入原始提示词。")
    else:
        with st.spinner("✨ 正在优化提示词…"):
            try:
                optimized = optimize_prompt(prompt.strip(), openai_key)
                st.subheader("✅ 优化后的英文提示词：")
                st.write(optimized)
                st.info("你可以直接用这个英文 prompt 去生成图片，也可以再手动微调。")
                # 方便你复制
                st.code(optimized, language="markdown")
            except Exception as e:
                st.error(f"提示词优化失败：{e}")

# =========================
# 处理：图像 → 文本描述
# =========================
if uploaded_image is not None:
    if not openai_key:
        st.warning("如需图片识别，请先在左侧输入 OpenAI API Key。")
    else:
        if st.button("🧠 分析上传图片内容"):
            with st.spinner("🧠 正在理解图片内容…"):
                try:
                    img_bytes = uploaded_image.read()
                    desc = describe_image(img_bytes, openai_key)
                    st.subheader("📝 图片描述结果：")
                    st.write(desc)
                except Exception as e:
                    st.error(f"解析图片失败：{e}")

# =========================
# 处理：生成图片
# =========================
if btn_generate:
    if not openai_key:
        st.error("请先在左侧输入 OpenAI API Key。")
    elif not prompt.strip():
        st.error("请先输入提示词。")
    else:
        # 根据风格对 prompt 做增强
        style_suffix = {
            "default（默认）": "",
            "anime（动漫）": ", anime style illustration, vibrant colors, clean line art, 2D, highly detailed",
            "realistic（写实）": ", ultra realistic photography, natural lighting, 4k, shallow depth of field",
            "cyberpunk（赛博）": ", cyberpunk style, neon lights, futuristic city, high contrast, dramatic lighting",
            "oil painting（油画）": ", oil painting, rich textures, visible brush strokes, art gallery quality",
            "comic（漫画）": ", comic book style, bold outlines, halftone shading, dynamic pose",
        }
        final_prompt = prompt.strip() + style_suffix.get(style_choice, "")

        with st.spinner("🎨 正在生成图片，请稍候…"):
            try:
                data = generate_images(final_prompt, openai_key, api_size, num_images)
            except Exception as e:
                st.error(f"生成请求失败：{e}")
            else:
                if "data" not in data:
                    st.error("API 返回了异常结果：")
                    st.json(data)
                else:
                    st.success("生成完成！👇 下方是本次生成的全部图片：")
                    cols = st.columns(num_images)

                    for i, img_info in enumerate(data["data"]):
                        img_b64 = img_info["b64_json"]
                        img_bytes = base64.b64decode(img_b64)

                        with cols[i]:
                            st.image(img_bytes, caption=f"图片 {i+1}", use_container_width=True)
                            image_download_button(
                                img_bytes,
                                filename=f"openai_image_{i+1}.png",
                                key=f"download_{i}",
                            )
