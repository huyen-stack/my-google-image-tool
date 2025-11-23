import streamlit as st
import requests
import base64

st.set_page_config(page_title="Gemini 2.0 Flash - 文生图", page_icon="✨")

st.title("✨ Gemini 2.0 Flash 文生图 (免费版)")
st.caption("使用 Google AI Studio 免费 API Key，无需 Imagen 权限")

# --- API Key 输入 ---
with st.sidebar:
    st.header("🔑 API Key 设置")
    api_key = st.text_input("Google API Key", type="password")
    st.info("提示：Gemini Flash 文生图无需开通付费权限，完全免费。")

# --- Prompt 输入框 ---
prompt = st.text_area("请输入你想生成的画面描述：", height=150)

if st.button("🚀 开始生成"):
    if not api_key:
        st.error("请先输入 API Key")
        st.stop()

    if not prompt:
        st.error("请先输入提示词 prompt")
        st.stop()

    st.info("📡 正在调用 Gemini 2.0 Flash 文生图 API...")

    # 最新 Gemini 图像生成接口（google提供）
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateImage?key={api_key}"

    payload = {
        "prompt": {"text": prompt},
        "image": {"size": "1024x1024"}  # 输出分辨率
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()

        if "images" in data:
            img_b64 = data["images"][0]["base64"]
            img_bytes = base64.b64decode(img_b64)

            st.image(img_bytes, caption="Gemini 2.0 Flash 生成结果", use_column_width=True)
            st.success("生成成功！🎉")

        else:
            st.error("❌ API 未返回图片：")
            st.json(data)

    except Exception as e:
        st.error(f"请求失败：{e}")
