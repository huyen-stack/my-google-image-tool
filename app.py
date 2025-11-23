import streamlit as st
import requests
import base64

# --- 页面配置 ---
st.set_page_config(page_title="Google Imagen 3 - REST 超轻版", page_icon="🎨")

st.title("🎨 Google Imagen 3 - REST 直连超轻版")
st.caption("完全兼容 Streamlit Cloud，无需 google-generative-ai SDK")

# --- API Key 输入 ---
with st.sidebar:
    st.header("🔑 设置 API Key")
    api_key = st.text_input("Google API Key", type="password")
    st.info("使用 REST 模式，兼容 Imagen 3 最新接口")

# --- Prompt 输入 ---
prompt = st.text_area("请输入绘图描述（建议英文）:", height=150)

if st.button("🚀 开始生成"):
    if not api_key:
        st.error("请先输入 API Key")
        st.stop()
    if not prompt:
        st.error("请先输入 Prompt 描述")
        st.stop()

    st.info("✏️ 正在联系 Google Imagen 3 API，请稍候...")

    # 正确的 REST URL（Imagen 3）
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0:generateImage?key={api_key}"

    payload = {
        "prompt": {
            "text": prompt
        },
        "image": {
            "size": "1024x1024"
        }
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        if "images" in data:
            # 获取第一张图
            img_b64 = data["images"][0]["base64"]
            img_bytes = base64.b64decode(img_b64)
            st.image(img_bytes, caption="Google Imagen 3 生成结果", use_column_width=True)
            st.success("生成成功！")

        else:
            st.error("❌ API 未返回图片，返回内容如下：")
            st.json(data)

    except Exception as e:
        st.error(f"请求失败：{e}")
