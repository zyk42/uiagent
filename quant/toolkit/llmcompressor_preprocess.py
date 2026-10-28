"""
llmcompressor 自定义数据集 preprocessing 函数。

用法 (oneshot 参数):
    dataset_path="/root/quant_calib/calib_512_llmcompressor.jsonl",
    dataset="json",
    preprocessing_func="/root/quant_calib/llmcompressor_preprocess.py:preprocess",

输入样本字段: {"text": <chat_template 已应用的字符串>, "image_path": <图片绝对路径>}
输出: processor 的 kwargs —— {"text": str, "images": [PIL.Image]}
"""

from PIL import Image


def preprocess(sample):
    img = Image.open(sample["image_path"]).convert("RGB")
    return {"text": sample["text"], "images": [img]}
