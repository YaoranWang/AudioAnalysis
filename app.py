"""
音频分析模块 - Streamlit应用主程序
"""

import streamlit as st
import os
import tempfile
from audio_utils import AudioProcessor
from text_utils import HarmfulContentDetector

# 设置页面配置
st.set_page_config(
    page_title="音频分析模块 - 有害内容检测",
    page_icon="🎵",
    layout="wide"
)

# 应用标题
st.title("🎵 音频分析模块")
st.markdown("上传音频文件，自动转录并检测有害内容")

# 初始化会话状态
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = None
if "detector" not in st.session_state:
    st.session_state.detector = None

# 侧边栏 - 模型设置
with st.sidebar:
    st.header("模型设置")

    # Whisper模型选择
    whisper_model = st.selectbox(
        "选择Whisper模型",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,  # 默认选择"base"
        help="模型越大，准确度越高，但处理速度越慢"
    )

    # 离线模式选择
    offline_mode = st.checkbox(
        "使用离线模式（无需网络连接）",
        value=True,
        help="选中此项将使用简单规则进行有害内容检测，无需下载模型"
    )

    # 有害内容检测模型选择（仅在非离线模式下显示）
    detector_model = "unitary/toxic-bert"
    if not offline_mode:
        detector_model = st.selectbox(
            "选择有害内容检测模型",
            options=["unitary/toxic-bert", "unitary/multilingual-toxic-xlm-roberta"],
            index=0,
            help="toxic-bert适用于英文，multilingual-toxic-xlm-roberta支持多语言"
        )

    # 有害内容阈值设置
    threshold = st.slider(
        "有害内容判定阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="分数高于此阈值的内容将被判定为有害"
    )

    # 语言选择
    language_option = st.radio(
        "语言选择",
        options=["自动检测", "中文", "英文"],
        index=0,
        help="选择音频语言或自动检测"
    )

    # 语言代码映射
    language_map = {
        "自动检测": None,
        "中文": "zh",
        "英文": "en"
    }

    selected_language = language_map[language_option]

    # 加载模型按钮
    if st.button("加载模型"):
        with st.spinner("正在加载模型，请稍候..."):
            # 加载音频处理器
            st.session_state.audio_processor = AudioProcessor(model_name=whisper_model)

            # 加载有害内容检测器
            st.session_state.detector = HarmfulContentDetector(
                model_name=detector_model,
                offline_mode=offline_mode
            )

            st.success("模型加载完成！")

# 主界面
st.header("1. 上传音频文件")
uploaded_file = st.file_uploader("选择音频文件", type=["wav", "mp3"], help="支持WAV和MP3格式")

# 处理上传的音频文件
if uploaded_file is not None:
    # 检查模型是否已加载
    if st.session_state.audio_processor is None or st.session_state.detector is None:
        st.warning("请先在侧边栏点击'加载模型'按钮加载模型")
    else:
        # 显示音频播放器
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        # 转录按钮
        if st.button("开始转录和分析"):
            # 保存上传的文件到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                # 转录音频
                with st.spinner("正在转录音频..."):
                    transcription = st.session_state.audio_processor.transcribe(
                        temp_file_path,
                        language=selected_language
                    )
                    st.session_state.transcription = transcription

                # 检测有害内容
                with st.spinner("正在检测有害内容..."):
                    detection_result = st.session_state.detector.detect(
                        transcription["text"],
                        threshold=threshold
                    )
                    st.session_state.detection_result = detection_result

                st.success("处理完成！")
            finally:
                # 删除临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# 显示转录结果
if st.session_state.transcription:
    st.header("2. 音频转录结果")

    # 显示检测到的语言
    detected_language = st.session_state.transcription["language"]
    supported_languages = st.session_state.audio_processor.get_supported_languages()
    language_name = supported_languages.get(detected_language, detected_language)

    st.info(f"检测到的语言: {language_name}")

    # 显示转录文本
    st.subheader("转录文本")
    st.write(st.session_state.transcription["text"])

    # 显示分段信息（可折叠）
    with st.expander("查看详细分段信息"):
        for i, segment in enumerate(st.session_state.transcription["segments"]):
            st.markdown(f"**片段 {i+1}** ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}")

# 显示有害内容检测结果
if st.session_state.detection_result:
    st.header("3. 有害内容检测结果")

    # 显示是否有害
    is_harmful = st.session_state.detection_result["is_harmful"]
    if is_harmful:
        st.error("⚠️ 检测到有害内容")
    else:
        st.success("✅ 未检测到有害内容")

    # 显示详细分类结果
    st.subheader("详细分类结果")
    result_df = st.session_state.detector.get_result_dataframe(st.session_state.detection_result)
    st.dataframe(result_df, use_container_width=True)

    # 显示有害类别详情（如果有）
    harmful_categories = st.session_state.detection_result["harmful_categories"]
    if harmful_categories:
        st.subheader("有害内容详情")
        for category, info in harmful_categories.items():
            st.markdown(f"- **{info['description']}**: {info['score']:.4f}")

# 页脚
st.markdown("---")
st.markdown("音频分析模块 - 有害内容检测 | 支持中文和英文")
