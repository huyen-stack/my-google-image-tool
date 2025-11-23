import streamlit as st
import requests
import base64
from io import BytesIO

# ==== 页面配置 ====
st.set_page_config(page_title="OpenAI 多功能图像生成器", page_icon="🎨", layout="wide")

st.title("🎨 OpenAI 多功能图像生成器（高级版）")
st.caption("支持：多图生成｜画风选择｜高清超分｜提示词优化｜图像→文字描述")

# ==== Sidebar ====
with st.sidebar:
    st.header("🔑 API Key 设置")
    openai_key = st.text_input("OpenAI API Key", type="password")

    st.header("🎨 画风选择")
    style = st.selectbox(
        "风格",
        ["default（默认）", "anime（动漫）", "realistic（写实）", "cyberpunk（赛博）", "oil painting（油画）", "comic（漫画）"]
    )

    st.header("📐 生成数量")
    num_images = st.slider("生成图片数量", 1, 4, 1)

    st.header("🖼 尺寸")
    size = st.selectbox("尺寸", ["1024x1024", "512x512", "256x256"])

# ==== 提示词 ====
prompt = st.text_area("请输入你想生成的画面描述：", height=150)

# ==== 自动提示词优化 ====
def optimize_prompt(prompt, key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "你是专业提示词优化助手，帮用户优化绘画 prompt（英文）。"},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post(url, json=payload, headers=headers)
    return r.json()["choices"][0]["message"]["content"]

# ==== 图片生成 ====
def generate_images(prompt, key, size, n):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": "b64_json"
    }
    r = requests.post(url, headers=headers, json=payload)
    return r.json()

# ==== 图片下载按钮 ====
def download_button(img_bytes, filename):
    buf = BytesIO(img_bytes)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 下载 PNG</a>'
    st.markdown(href, unsafe_allow_html=True)

# ==== 图像 → 文本 描述 ====
def describe_image(img_bytes, key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    image_b64 = base64.b64encode(img_bytes).decode()

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", 
             "content": [
                 {"type": "input_text", "text": "请描述这张图片。"},
                 {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"}
             ]}
        ]
    }

    r = requests.post(url, json=payload, headers=headers)
    return r.json()["choices"][0]["message"]["content"]


# ==== 按钮区域 ====
col1, col2, col3 = st.columns(3)

with col1:
    optimize = st.button("✨ 优化提示词")
with col2:
    generate = st.button("🎨 生成图片")
with col3:
    upload_img = st.file_uploader("📤 上传图片（图像→文字）", type=["png", "jpg"])

# ==== 执行逻辑 ====

# 提示词优化
if optimize and prompt:
    if not openai_key:
        st.error("请先输入 API Key")
    else:
        st.success("正在优化提示词...")
        new_prompt = optimize_prompt(prompt, openai_key)
        st.subheader("✨ 优化后的提示词：")
        st.write(new_prompt)
        prompt = new_prompt

# 图像→文字
if upload_img:
    if not openai_key:
        st.error("请先输入 API Key")
    else:
        st.info("📡 正在分析图片内容...")
        img_bytes = upload_img.read()
        desc = describe_image(img_bytes, openai_key)
        st.subheader("📝 图片描述结果：")
        st.write(desc)

# 图像生成
if generate:
    if not openai_key:
        st.error("请输入 API Key")
    elif not prompt:
        st.error("请输入 prompt")
    else:
        final_prompt = prompt
        if style != "default（默认）":
            final_prompt += f" | style: {style}"

        st.info("🎨 正在生成图片...")

        res = generate_images(final_prompt, openai_key, size, num_images)

        if "data" not in res:
            st.error("⚠️ API 返回错误：")
            st.json(res)
        else:
            st.success("生成完成！🎉")

            cols = st.columns(num_images)
            for i, img_data in enumerate(res["data"]):
                img_b64 = img_data["b64_json"]
                img_bytes = base64.b64decode(img_b64)

                with cols[i]:
                    st.image(img_bytes, caption=f"图片 {i+1}")

                    download_button(img_bytes, f"image_{i+1}.png")
