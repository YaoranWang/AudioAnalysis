"""
音频处理模块 - 负责音频文件的处理和转录功能
"""

import os
import tempfile
import whisper
import torch
from pydub import AudioSegment

class AudioProcessor:
    """音频处理类，提供音频转文字功能"""
    
    def __init__(self, model_name="base"):
        """
        初始化音频处理器
        
        参数:
            model_name (str): Whisper模型名称，可选值包括 "tiny", "base", "small", "medium", "large"
        """
        # 检查CUDA是否可用，如果可用则使用GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载Whisper模型
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"已加载Whisper {model_name}模型")
        
    def convert_audio_format(self, audio_file, target_format="wav"):
        """
        转换音频格式为模型所需格式
        
        参数:
            audio_file: 上传的音频文件
            target_format: 目标格式，默认为wav
            
        返回:
            临时文件路径
        """
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_format}")
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 读取上传的音频文件
        audio = AudioSegment.from_file(audio_file)
        
        # 导出为目标格式
        audio.export(temp_file_path, format=target_format)
        
        return temp_file_path
    
    def transcribe(self, audio_file, language=None):
        """
        将音频转换为文字
        
        参数:
            audio_file: 音频文件路径
            language: 音频语言代码 (如 "en", "zh", "fr", "es", "ar", "ru")，
                     如果为None则自动检测
        
        返回:
            转录文本和检测到的语言
        """
        # 转换音频格式
        if not audio_file.endswith('.wav'):
            audio_file = self.convert_audio_format(audio_file)
            temp_created = True
        else:
            temp_created = False
            
        try:
            # 设置转录选项
            options = {}
            if language:
                options["language"] = language
            
            # 执行转录
            result = self.model.transcribe(audio_file, **options)
            
            # 返回转录文本和检测到的语言
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
        finally:
            # 如果创建了临时文件，则删除
            if temp_created and os.path.exists(audio_file):
                os.remove(audio_file)
    
    def get_supported_languages(self):
        """
        获取Whisper支持的语言列表
        
        返回:
            支持的语言字典 {语言代码: 语言名称}
        """
        # Whisper支持的主要语言
        languages = {
            "en": "English (英语)",
            "zh": "Chinese (中文)",
            "fr": "French (法语) - 接口已预留",
            "es": "Spanish (西班牙语) - 接口已预留",
            "ar": "Arabic (阿拉伯语) - 接口已预留",
            "ru": "Russian (俄语) - 接口已预留"
        }
        
        return languages
