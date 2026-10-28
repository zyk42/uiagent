# eval/results 索引(一致性评估归档)

> 更新: 2026-10-28 · 评估方法: `../量化一致性评估细则.md` · 结果分析: `../评估结果-第一轮.md`
> 参照: `../refs_cloudml_bf16.jsonl`(sglang 栈 BF16)、`../refs_cloudml_bf16_vllm.jsonl`(vllm 栈 BF16)

## 目录组织

- **模型名目录**(推荐查阅): `<模型>[-形态][-vllm]/`, 内含 `details.jsonl`(逐条输出) + `score_vs_*_ref.json`(细则打分)
- **`s-<部署ID>/` 目录**: 各次运行的原始归档, 与模型名目录内容一致, 保留备查
- 无后缀 = sglang 栈;`-vllm` = vLLM 栈

## sglang 栈一致性(vs sglang-BF16 参照, tp4 形态)

| 模型 | L0 / L1 / L2 | 目录 |
|---|---|---|
| BF16 自检(噪声地板) | 100(参照自身); 重跑间 71~80 | `s-...-n4r7y/` |
| fp8-w8a8 | 47.9 / 74.6 / 86.6 | `s-...-c100y/` |
| awq-w4a16 | 37.0 / 67.0 / 79.8 | `s-...-hvaf6/` |
| gptq-nvfp4 | 33.3 / 61.8 / 81.2 | `s-...-pttfp/` |
| gptq-w4a16 | 32.5 / 61.5 / 75.8 | `s-...-0gxyg/` |
| nvfp4-w4a4 | 27.9 / 58.1 / 79.5 | `s-...-1692i/` |
| gptq-nvfp4-skip3 | ❌ 乱码(混合精度不兼容) | `gptq-nvfp4-skip3/` |
| skip6/12/24 | 未测(同路径判断) | `gptq-nvfp4-skip*/` |
| smoothquant-w8a8 | 不可部署 | `smoothquant-w8a8/` |

## vLLM 栈一致性(vs vllm-BF16 参照; 引擎漂移基线 L0=62.1)

| 模型 | L0 / L1 / L2(vllm 参照) | 目录 |
|---|---|---|
| bf16-vllm | 100(vllm 参照自身); vs sglang 参照 62.1 / 79.2 / 87.5 | `bf16-vllm/` |
| awq-w4a16(-dp2tp2)-vllm | 37.4 / 66.3 / 80.6(38.3 / 63.4 / 79.7) | `awq-w4a16*-vllm/` |
| gptq-w4a16(-dp2tp2)-vllm | 35.1 / 64.0 / 77.4(32.9 / 63.7 / 76.3) | `gptq-w4a16*-vllm/` |
| gptq-nvfp4(-dp2tp2)-vllm | 28.3 / 60.9 / 79.7(31.4 / 60.3 / 81.4) | `gptq-nvfp4*-vllm/` |
| nvfp4-w4a4(-dp2tp2)-vllm | 24.3 / 54.6 / 77.7(26.3 / 52.3 / 75.4) | `nvfp4-w4a4*-vllm/` |
| skip3/6/12/24-vllm | 见各自目录 score_vs_vllm_ref.json | `gptq-nvfp4-skip*-vllm/` |
| 4bit dp1tp4-vllm ×4 | 未测(vllm dp=4 部署失败) | — |

## 备注

- skip 系列(skip3/6/12/24)在 vllm 栈输出正常、可评估; sglang 栈乱码不可用
- 一致性主结论(两轮一致): **awq-w4a16 / fp8-w8a8 第一梯队, nvfp4-w4a4 最弱**
- 细则打分器: `/root/compare_actions.py`(可离线重打分); 推理脚本: `/root/eval_consistency.py`
