#!/usr/bin/env python3
"""
将 calib_512.jsonl (regen 格式) 转为 llmcompressor 自定义数据集格式。

输出每条记录: {"text": <chat_template 应用后的完整字符串>, "image_path": <图片路径>}
配合 llmcompressor_preprocess.py:preprocess 使用 (加载 PIL 图片 -> {"text", "images"})。

消息结构 = [system] + conversations(历史, 最后一条 user 的 <image> 换成 image part)
           + [assistant: ground_truth]
"""
import orjson
from transformers import AutoProcessor

MODEL = '/mnt/xiaoai-guiagent/guiagent/training-out/shot-navigation-rl-merge/20260806zwzv5/checkpoints/RL/final'
SRC = '/root/quant_calib/calib_512.jsonl'
DST = '/root/quant_calib/calib_512_llmcompressor.jsonl'

processor = AutoProcessor.from_pretrained(MODEL)

n = 0
warns = 0
with open(SRC, 'rb') as f, open(DST, 'wb') as out:
    for line in f:
        d = orjson.loads(line)
        # conv[-1] 是旧轨迹响应, 输入截止到 conv[-2] (带图的 user), ground_truth 是 regen 目标
        conv = d['conversations'][:-1]

        # 定位 <image>: 应恰好出现 1 次, 且在最后一条 user 消息
        img_cnt = sum(m['content'].count('<image>') for m in conv)
        if img_cnt != 1 or '<image>' not in conv[-1]['content'] or conv[-1]['role'] != 'user':
            warns += 1
            print(f"WARN record {d['id']}: image tag count={img_cnt}, last role={conv[-1]['role']}")
            continue

        messages = []
        if d.get('system'):
            messages.append({"role": "system", "content": d['system']})
        for m in conv:
            if m is conv[-1]:
                # 把 <image> 替换为 {"type": "image"} content part
                parts = []
                segs = m['content'].split('<image>')
                for i, seg in enumerate(segs):
                    if seg:
                        parts.append({"type": "text", "text": seg})
                    if i < len(segs) - 1:
                        parts.append({"type": "image"})
                messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": m['role'], "content": m['content']})
        messages.append({"role": "assistant", "content": d['ground_truth']})

        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        out.write(orjson.dumps({"text": text, "image_path": d['image']}) + b'\n')
        n += 1

print(f"converted {n} records -> {DST}  (skipped {warns})")
