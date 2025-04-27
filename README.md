# 音频分析模块 (Audio Analysis Module)

这是一个音频分析模块，可以将上传的音频文件转录为文本，并检测文本中的有害内容。

## 功能特点

- 支持上传 `.wav` 和 `.mp3` 格式的音频文件
- 使用 OpenAI Whisper 模型进行语音转文字
- 支持中文和英文音频的转录
- 使用 Huggingface 模型进行有害内容检测
- 提供详细的有害内容分类结果
- 简洁直观的 Streamlit 用户界面

## 安装说明

### 系统要求

- Python 3.8 或更高版本
- 足够的磁盘空间用于存储模型（根据选择的模型大小，可能需要几百MB到几GB）
- 推荐使用支持CUDA的GPU以加速处理（非必须）

### 安装步骤

1. 克隆或下载本项目到本地

2. 创建并激活虚拟环境（可选但推荐）

```bash
# 使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用说明

1. 启动应用

```bash
streamlit run app.py
```

2. 在浏览器中打开应用（通常会自动打开，默认地址为 http://localhost:8501）

3. 使用流程：
   - 在侧边栏中选择模型和设置
   - 点击"加载模型"按钮
   - 上传音频文件（.wav 或 .mp3 格式）
   - 点击"开始转录和分析"按钮
   - 查看转录结果和有害内容检测结果

## 支持的语言

当前版本支持以下语言的音频转录和有害内容检测：

- 中文 (Chinese)
- 英文 (English)

## 未来工作 (Future Work)

计划在未来版本中添加对以下联合国官方语言的支持：

- 法语 (French)
- 西班牙语 (Spanish)
- 阿拉伯语 (Arabic)
- 俄语 (Russian)

代码中已预留这些语言的接口，但当前版本尚未实现完整的有害内容检测功能。

## 项目结构

```
.
├── app.py              # Streamlit应用主程序
├── audio_utils.py      # 音频处理模块
├── text_utils.py       # 文本处理和有害内容检测模块
├── requirements.txt    # 项目依赖
├── README.md           # 项目文档
└── samples/            # 示例音频文件目录（可选）
```

## 注意事项

- 首次运行时，程序会自动下载所需的模型，这可能需要一些时间
- 大型音频文件的处理可能需要较长时间，特别是在没有GPU的情况下
- 有害内容检测的准确性取决于所选模型和设置的阈值
