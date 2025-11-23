import streamlit as st
import google.generative_ai as genai
import os

# --- 页面配置 ---
st.set_page_config(page_title="Google AI 绘图", page_icon="🍌")

st.title("🍌 Google AI 绘图神器")
st.caption("Powered by Imagen 3 (Nano Banana)")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("🔑 密钥设置")
    google_api_key = st.text_input("Google API Key", type="password", help="请输入你的 AIza... 开头的密钥")
    
    st.info("💡 提示：此功能需要你的 API Key 拥有 Imagen 模型权限。")

# --- 主界面 ---
prompt = st.text_area("请描述你想要的画面 (推荐用英文):", height=150, 
                     placeholder="例如: A cute cyberpunk cat sitting on a neon rooftop, cinematic lighting, 8k resolution")

if st.button("🚀 开始生成", type="primary"):
    if not prompt:
        st.warning("请先输入描述词！")
        st.stop()

    if not google_api_key:
        st.error("请先在侧边栏输入 Google API Key")
        st.stop()
        
    try:
        genai.configure(api_key=google_api_key)
        
        # 尝试调用 Imagen 3 模型
        # 如果这个 ID 报错，可以尝试换成 'imagen-2' 或 'imagen-3.0-generate-001'
        model = genai.GenerativeModel('imagen-3.0-generate-001')
        
        with st.spinner("Google AI (Nano Banana) 正在绘图..."):
            response = model.generate_content(prompt)
            
            if response.parts:
                # 获取图片数据并显示
                st.image(response.parts[0].inline_data.data, caption="Google 生成结果", use_column_width=True)
                st.success("生成成功！")
            else:
                st.error("生成失败：API 返回了空数据。")
                st.warning("可能原因：你的 API Key 暂时没有画图权限。")
                
    except Exception as e:
        st.error(f"发生错误: {e}")
