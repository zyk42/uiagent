# GUI-Owl-1.5-32B 量化一致性评估集

> 用途：衡量**量化模型与 BF16 原版模型输出的一致性**，用于量化方法选型。
> 生成日期：2026-10-21 · 维护：zhangyikang1

## 文件

| 文件 | 说明 |
|---|---|
| `eval_pc_360.jsonl` | 评估集，360 条（win / mac / ubuntu 各 120） |
| `images/` | 360 张截图（jsonl 中 `image_path` 指向此处，自包含） |

## 数据格式（每行一个 JSON）

```jsonc
{
  "sample_id": "20240930..._s5_0",   // 源数据 ID
  "platform": "win",                 // win | mac | ubuntu
  "prompt": "### Background ###...", // PC-Agent 完整决策提示词（含任务指令+历史）
  "image_path": "images/<sid>.png",  // 当前截图
  "ref_output": "```json\n{...}\n```",  // ★ BF16 原模型的参照输出（temperature=0）
  "ref_thought": "...",              // 解析出的 Thought
  "ref_action":  "Tap (182, 744)",   // 解析出的 Action（核心对比字段）
  "ref_summary": "...",
  "gt_action":   "Tap (195, 743)"    // AgentNet 人工 GT（对照参考，非评估基准）
}
```

## 设计要点

1. **参照答案 = BF16 模型的确定性输出**（temperature=0 蒸馏结果），来自
   `pc_agent_distill/distill_32b_dedup.jsonl`（win/mac）与 `distill_ubuntu_32b_dedup.jsonl`。
   评估的是"量化相对原版的行为漂移"，不是任务正确率。
2. **与校准集零重叠**（量化校准时没见过这些样本，避免高估一致性）。
3. **三平台均衡 + 平台内按动作类型分层 + task 去重**：Tap/Type/Stop/Shortcut/Press/
   Right Tap/Double Tap/Drag/Scroll 等 17 类动作按真实分布覆盖。
4. 样本已通过 `parse_ok` + 三字段（Thought/Action/Summary）齐全过滤。

## 评估方法

对每个被测模型起 sglang 服务，按蒸馏 pipeline 完全相同的请求格式逐条请求：

```python
messages = [
    {"role": "system", "content": "You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction."},
    {"role": "user", "content": [
        {"type": "text", "text": record["prompt"]},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of_image}"}},
    ]},
]
# temperature=0, max_tokens=2048
```

解析模型输出（`json.loads(out.split('```json')[-1].split('```')[0])`），与 `ref_*` 对比。

## 建议指标

| 指标 | 定义 | 说明 |
|---|---|---|
| **Action 完全一致率** | `model_action == ref_action` 占比 | 核心指标 |
| 动作类型一致率 | 忽略参数只比类型（Tap/Type/...） | 看决策层面漂移 |
| 坐标偏差 | Tap 类坐标归一化欧氏距离中位数 | 看定位精度漂移 |
| JSON 可解析率 | 输出能解析出三字段的比例 | 量化是否破坏格式 |
| （参考）vs 人工 GT 匹配率 | `model_action == gt_action` | 量化后绝对水平 |

注意：BF16 自身 temperature=0 的输出是确定性的，因此 BF16 重跑应得到 100% 一致——可作为评估管线自检。

## 关联

- 校准集：`/root/quant_calib/calib_pc_512.jsonl`（v2 三平台，与本评估集互斥）
- 量化产物与速度对比：`../quant/README.md`
- 压测集：`../press/press_pc_512.jsonl`
- 生成脚本：`/root/quant_calib/make_eval_set_v2.py`
