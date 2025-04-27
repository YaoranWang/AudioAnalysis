"""
文本处理模块 - 负责文本分析和有害内容检测
"""

import os
import pandas as pd
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class HarmfulContentDetector:
    """有害内容检测类，提供文本有害内容分析功能"""

    def __init__(self, model_name="unitary/toxic-bert", offline_mode=True):
        """
        初始化有害内容检测器

        参数:
            model_name (str): Huggingface模型名称，默认使用unitary/toxic-bert
            offline_mode (bool): 是否使用离线模式，默认为True
        """
        # 定义有害内容类别及其中文解释
        self.category_descriptions = {
            "toxicity": "有毒/有害言论",
            "severe_toxicity": "严重有害言论",
            "obscene": "淫秽内容",
            "threat": "威胁性言论",
            "insult": "侮辱性言论",
            "identity_attack": "身份攻击",
            "sexual_explicit": "露骨色情内容"
        }

        if offline_mode:
            print("使用简单规则检测模式（离线模式）")
            self.detector = None
            self.offline_mode = True
        else:
            try:
                # 尝试加载有害内容检测模型
                print(f"尝试加载有害内容检测模型: {model_name}")
                self.detector = pipeline(
                    "text-classification",
                    model=model_name,
                    return_all_scores=True
                )
                self.offline_mode = False
                print(f"已成功加载有害内容检测模型: {model_name}")
            except Exception as e:
                print(f"无法加载模型，错误: {str(e)}")
                print("切换到简单规则检测模式（离线模式）")
                self.detector = None
                self.offline_mode = True

    def detect(self, text, threshold=0.5):
        """
        检测文本中的有害内容

        参数:
            text (str): 需要检测的文本
            threshold (float): 判定为有害的阈值，默认0.5

        返回:
            检测结果字典，包含是否有害和详细分类
        """
        if self.offline_mode:
            return self._detect_offline(text, threshold)
        else:
            return self._detect_with_model(text, threshold)

    def _detect_with_model(self, text, threshold=0.5):
        """使用Huggingface模型进行检测"""
        # 检测文本
        results = self.detector(text)[0]

        # 整理结果
        scores = {item['label']: item['score'] for item in results}

        # 判断是否包含有害内容
        harmful_categories = {
            category: {
                "score": score,
                "description": self.category_descriptions.get(category, category)
            }
            for category, score in scores.items()
            if score >= threshold
        }

        is_harmful = len(harmful_categories) > 0

        # 返回结果
        return {
            "is_harmful": is_harmful,
            "harmful_categories": harmful_categories,
            "all_scores": scores
        }

    def _detect_offline(self, text, threshold=0.5):
        """使用简单规则进行离线检测"""
        # 定义有害词汇列表（简单示例，实际应用中可以扩展）
        harmful_words = {
            "toxicity": ["toxic", "poison", "有毒", "毒害", "恶毒"],
            "severe_toxicity": ["fuck", "shit", "damn", "操", "妈的", "他妈的"],
            "obscene": ["obscene", "淫秽", "下流", "色情"],
            "threat": ["kill", "murder", "die", "杀", "死", "打死", "威胁"],
            "insult": ["stupid", "idiot", "dumb", "蠢", "笨", "白痴", "傻"],
            "identity_attack": ["nigger", "chink", "negro", "黑鬼", "支那", "犹太猪"],
            "sexual_explicit": ["penis", "vagina", "sex", "阴茎", "阴道", "性交"]
        }

        # 将文本转为小写以进行不区分大小写的匹配
        text_lower = text.lower()

        # 检查每个类别的有害词汇
        scores = {}
        harmful_categories = {}

        for category, words in harmful_words.items():
            # 计算匹配的有害词数量
            matches = sum(1 for word in words if word.lower() in text_lower)

            # 根据匹配数量计算分数（简单实现）
            score = min(matches * 0.2, 1.0)  # 每个匹配增加0.2分，最高1.0
            scores[category] = score

            # 如果分数超过阈值，标记为有害
            if score >= threshold:
                harmful_categories[category] = {
                    "score": score,
                    "description": self.category_descriptions.get(category, category)
                }

        is_harmful = len(harmful_categories) > 0

        # 返回结果
        return {
            "is_harmful": is_harmful,
            "harmful_categories": harmful_categories,
            "all_scores": scores
        }

    def get_result_dataframe(self, detection_result):
        """
        将检测结果转换为DataFrame格式，方便在Streamlit中展示

        参数:
            detection_result: detect()方法返回的结果

        返回:
            pandas DataFrame
        """
        # 提取所有分数
        scores = detection_result["all_scores"]

        # 创建DataFrame
        df = pd.DataFrame({
            "类别": [self.category_descriptions.get(category, category) for category in scores.keys()],
            "分数": list(scores.values()),
            "是否有害": ["是" if score >= 0.5 else "否" for score in scores.values()]
        })

        # 按分数降序排序
        df = df.sort_values("分数", ascending=False).reset_index(drop=True)

        return df
