#!/usr/bin/env python3
"""
Qwen3-VL-30B-A3B 量化脚本 (llmcompressor oneshot)

用法: python run_quant.py <method>
  method: gptq-w4a16 | awq-w4a16 | nvfp4-w4a4 | mxfp4-w4a4 | smoothquant-w8a8

ignore 列表与已有 quant/w4a16-x 保持一致:
  - lm_head
  - model.visual.*  (整个 visual tower, 含 merger / deepstack_merger_list)
  - model.language_model.layers.*.mlp.gate  (MoE router)
"""
import sys

import orjson
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL = "/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct"
OUT_BASE = "/mnt/lisiqi23/internship/zhangyikang/uiagent/quant"
CALIB_JSONL = "/root/quant_calib/calib_pc_512.jsonl"

# dense qwen3_vl: 无 MoE gate; 忽略 visual tower + lm_head
IGNORE = ["lm_head", "re:model\\.visual\\..*"]


def ignore_skip_layers(n):
    """基础 ignore + 跳过语言模型前 n 层不量化（浅层保护）"""
    return IGNORE + [f"re:model\\.language_model\\.layers\\.{i}\\..*" for i in range(n)]


class CalibDataset(Dataset):
    """逐条懒加载: processor(text, images) 直接产出带 batch 维的张量。

    绕过 HF datasets 的 arrow 物化 (512 张高清截图的 pixel_values ~35GB 会
    触发 pyarrow 2GB offset 上限)。"""

    def __init__(self, jsonl_path, processor):
        self.rows = [orjson.loads(l) for l in open(jsonl_path, "rb")]
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        out = self.processor(text=[row["text"]], images=[img], return_tensors="pt")
        batch = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(torch.bfloat16)  # 与模型 dtype 一致, 省一半内存
            batch[k] = v
        return batch

RECIPES = {
    "gptq-w4a16": lambda: GPTQModifier(
        targets="Linear", scheme="W4A16", ignore=IGNORE,
    ),
    "awq-w4a16": lambda: [
        # dense 模型默认不做激活缓存 offload, 必须显式设 cpu
        # 另配 63 条 <=3072 token 减小子集 (calib_pc_awq_64.jsonl) 控制缓存规模
        AWQModifier(ignore=IGNORE, duo_scaling=True, offload_device=torch.device("cpu")),
        GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE),
    ],
    "nvfp4-w4a4": lambda: QuantizationModifier(
        targets="Linear", scheme="NVFP4", ignore=IGNORE,
    ),
    # GPTQ 优化的 NVFP4 (权重经 GPTQ 补偿, 比纯 RTN 的 nvfp4-w4a4 精度更好)
    "gptq-nvfp4": lambda: GPTQModifier(
        targets="Linear", scheme="NVFP4", ignore=IGNORE,
    ),
    # GPTQ+NVFP4 浅层保护变体: 前 N 层保持 BF16
    "gptq-nvfp4-skip3": lambda: GPTQModifier(
        targets="Linear", scheme="NVFP4", ignore=ignore_skip_layers(3),
    ),
    "gptq-nvfp4-skip6": lambda: GPTQModifier(
        targets="Linear", scheme="NVFP4", ignore=ignore_skip_layers(6),
    ),
    "gptq-nvfp4-skip12": lambda: GPTQModifier(
        targets="Linear", scheme="NVFP4", ignore=ignore_skip_layers(12),
    ),
    "gptq-nvfp4-skip24": lambda: GPTQModifier(
        targets="Linear", scheme="NVFP4", ignore=ignore_skip_layers(24),
    ),
    "mxfp4-w4a4": lambda: QuantizationModifier(
        targets="Linear", scheme="MXFP4", ignore=IGNORE,
    ),
    "smoothquant-w8a8": lambda: [
        SmoothQuantModifier(smoothing_strength=0.5, ignore=IGNORE),
        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=IGNORE),
    ],
    # FP8 动态量化: 无需校准数据, 精度最接近 BF16, Hopper 内核成熟
    "fp8-w8a8": lambda: QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE,
    ),
}

# AWQ 用 basic pipeline, 需要把模型直接放上 GPU
PIPELINE_OVERRIDE = {}
DEVICEMAP_OVERRIDE = {}


def main():
    method = sys.argv[1]
    assert method in RECIPES, f"unknown method {method}, choose from {list(RECIPES)}"
    recipe = RECIPES[method]()
    out_dir = f"{OUT_BASE}/{method}"

    # AWQ 激活缓存 ~500MB/样本(长序列), 必须用减小子集, 否则 CPU 内存被撑爆
    calib_jsonl = "/root/quant_calib/calib_pc_awq_64.jsonl" if method == "awq-w4a16" else CALIB_JSONL
    n_samples = sum(1 for _ in open(calib_jsonl, "rb"))
    print(f"method={method}  calib={calib_jsonl} ({n_samples} samples)", flush=True)

    # llmcompressor 的 initialize_model_from_path 硬编码 AutoModelForCausalLM,
    # 不支持 VLM, 需自行加载后传入
    print(f"loading model from {MODEL} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype="auto", trust_remote_code=True,
        device_map=DEVICEMAP_OVERRIDE.get(method),  # basic pipeline 需模型在 GPU
    )
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("model loaded, start oneshot ...", flush=True)

    print("model loaded, building calibration dataloader ...", flush=True)
    calib_loader = DataLoader(
        CalibDataset(calib_jsonl, processor),
        batch_size=None,  # 样本已带 batch 维, 逐条产出
        shuffle=False,
        num_workers=0,
    )

    oneshot(
        model=model,
        processor=processor,
        recipe=recipe,
        dataset=calib_loader,
        max_seq_length=8192,  # 校准集已全部 <=8186; eager 注意力 O(L^2) 瞬存 ~34GB @8192
        num_calibration_samples=n_samples,
        pipeline=PIPELINE_OVERRIDE.get(method, "independent"),
        output_dir=out_dir,
    )
    print(f"DONE {method} -> {out_dir}")


if __name__ == "__main__":
    main()
