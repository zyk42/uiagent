# GUI-Owl-1.5-32B-Instruct 量化产物说明

> 基座模型：`/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct`
> （Qwen3-VL-32B dense，64 层；BF16 原版 64GB）
> 量化工具：llmcompressor 0.14.0（compressed-tensors 0.19.0）
> 生成日期：2026-10-14 ~ 2026-10-20

## 产物总览

| 目录 | 方法 | 格式 | 大小 | 校准集 | sglang@H20 可部署 |
|---|---|---|---|---|---|
| **`gptq-w4a16/`** | GPTQ W4A16 | pack-quantized, int4 g128 sym | 20GB | v2 三平台 ✅ | ✅ |
| **`fp8-w8a8/`** | **FP8 动态量化 W8A8（推荐）** | float-quantized, fp8 channel-wise W + dynamic per-token A | 34GB | 无需校准（datafree） | ✅ |
| `awq-w4a16/` | AWQ(平滑)+GPTQ W4A16 | pack-quantized, int4 g128 sym | 20GB | v1 (win/mac) | ✅ |
| **`nvfp4-w4a4/`** | NVFP4 W4A4 (RTN) | nvfp4-pack-quantized, fp4 g16 | 21GB | v2 三平台 | ❌ 需 Blackwell |
| **`gptq-nvfp4/`** | **GPTQ+NVFP4** W4A4 | nvfp4-pack-quantized, fp4 g16 | 21GB | v2 三平台 | ❌ 需 Blackwell |
| **`gptq-nvfp4-skip3/`** | GPTQ+NVFP4, 前 3 层保持 BF16 | 同上 | 23GB | v2 三平台 | ❌ 需 Blackwell |
| **`gptq-nvfp4-skip6/`** | GPTQ+NVFP4, 前 6 层保持 BF16 | 同上 | 25GB | v2 三平台 | ❌ 需 Blackwell |
| **`gptq-nvfp4-skip12/`** | GPTQ+NVFP4, 前 12 层保持 BF16 | 同上 | 29GB | v2 三平台 | ❌ 需 Blackwell |
| **`gptq-nvfp4-skip24/`** | GPTQ+NVFP4, 前 24 层保持 BF16 | 同上 | 37GB | v2 三平台 | ❌ 需 Blackwell |
| `mxfp4-w4a4/` | MXFP4 W4A4 (RTN) | mxfp4-pack-quantized, fp4 g32 | 20GB | v1 (win/mac) | ❌ dense 无内核 |

> 注：smoothquant-w8a8 / gptq-w4a16-v1-winmac / nvfp4-w4a4-v1-winmac 已于 2026-10-27 删除。

**加粗**为推荐使用的新版本（v2 三平台校准集）。所有产物均 ignore 了 visual tower 与 lm_head（语言模型 Linear 层量化）。

## 部署限制说明（H20-3e 实测）

> ⚠️ **2026-10-27 更新**：换装 **vLLM 0.32.0** 后，NVFP4 系列在 H20 上**可以部署**——vLLM 走 `HummingNvFp4LinearKernel`（humming-kernels 支持 Hopper 的 FP4 GEMM），已实测 gptq-nvfp4 正常推理。以下 sglang 结论仅作历史参考（sglang 已从环境卸载）。

**sglang 0.5.20（已废弃）**：

- **NVFP4 W4A4**（含 GPTQ 版及 skip 变体）：~~需 Blackwell~~ → **vLLM 0.32 + H20 可部署**（HummingNvFp4 内核）；sglang 不支持（其 NVFP4 路径仅认 Blackwell）
- **MXFP4 W4A4**：sglang/vLLM 的 MXFP4 均仅支持 MoE 层，dense 模型无部署路径

## 速度对比（H20-3e 单卡实测，v1 校准集产物，同内核的 v2 版速度相同）

| 模型 | TTFT 2k/4k/8k tok | Decode | 16 并发吞吐 | 显存 |
|---|---|---|---|---|
| BF16 基线 | 1.16 / 1.84 / 3.09s | 49.3 tok/s | 44.5 tok/s | 64GB |
| GPTQ/AWQ W4A16 | 1.71 / 2.79 / 4.70s | **73.7 tok/s** | ~36-38 tok/s | **20GB** |
| SmoothQuant W8A8 | **0.68 / 1.09 / 1.80s** | 33.0 tok/s | **72.5 tok/s** | 34GB |

- 单实例 GUI Agent（decode 为主）：W4A16 最快
- 批量并发（蒸馏/评测）：W8A8 吞吐最高

## 校准集与评估集

| 文件 | 说明 |
|---|---|
| `/root/quant_calib/calib_pc_512.jsonl` | **v2 校准集**：win 170 + mac 171 + ubuntu 171，17 类动作全覆盖（稀有类保底 3 条），≤8192 token，与评估集零重叠 |
| `/root/quant_calib/calib_pc_512_v1_winmac.jsonl.bak` | v1 校准集（仅 win/mac，已废弃） |
| `/root/quant_calib/calib_pc_awq_64.jsonl` | AWQ 专用减小子集（44 条 ≤3072 token，AWQ 激活缓存内存敏感） |
| `/root/quant_calib/eval_pc_360.jsonl` | **一致性评估集**：win/mac/ubuntu 各 120 条，参照答案 = BF16 模型蒸馏输出（temperature=0），与所有校准集零重叠 |

数据来源链：AgentNet 原始轨迹 → `convert_to_pcagent.py` 转 PC-Agent 格式 → BF16 模型蒸馏输出作为参照/校准目标（on-policy）。

## 复现 / 重新量化

```bash
cd /root/quant_calib
# 单个方法（method: gptq-w4a16 | awq-w4a16 | nvfp4-w4a4 | mxfp4-w4a4 | smoothquant-w8a8 | gptq-nvfp4）
setsid nohup bash run_<method>.sh > /dev/null 2>&1 < /dev/null &

# 查看进度
tr '\r' '\n' < logs/<method>.log | grep Calibrating | tail -1
```

核心脚本：`run_quant.py`（量化主程序）、`make_calib_v2.py`（校准集构建）、`make_eval_set_v2.py`（评估集构建）、`bench_all.sh`（测速）、`smoke_test.sh`（部署验证）。

## 浅层保护变体（2026-10-26，GPTQ+NVFP4）

语言模型前 N 层对 FP4 量化最敏感，`gptq-nvfp4-skip{3,6,12,24}` 系列将前 N 层保持 BF16、其余层 GPTQ+NVFP4：

| 变体 | 保护 Linear 数 | 大小 | 相对全量化的增量 |
|---|---|---|---|
| gptq-nvfp4（基线） | 0 | 21GB | — |
| skip3 | 21（3层×7） | 23GB | +2GB |
| skip6 | 42 | 25GB | +4GB |
| skip12 | 84 | 29GB | +8GB |
| skip24 | 168 | 37GB | +16GB |

选型建议：用 `eval/` 下的一致性评估集对比各版与 BF16 的 Action 一致率，取满足精度要求的最小版本。

## FP8 说明（2026-10-21 新增）

`fp8-w8a8/` 使用 `FP8_DYNAMIC` preset（权重 per-channel 静态 FP8 + 激活 per-token 动态 FP8）：
**无需校准数据、精度最接近 BF16、Hopper 内核最成熟**，是 H20 上的推荐 FP8 方案。
（备选项：`FP8` 静态激活——需校准且精度更脆；`FP8_BLOCK` 块级——DeepSeek 风格。）
sglang 部署已实测通过。生成耗时 ~10 分钟（纯权重量化，无校准前向）。

## 部署命令（可部署的模型）

```bash
python -m sglang.launch_server \
  --model-path /mnt/lisiqi23/internship/zhangyikang/uiagent/quant/gptq-w4a16 \
  --trust-remote-code --host 0.0.0.0 --port 30000 --mem-fraction-static 0.85
```

请求格式与蒸馏 pipeline 一致（system + user[text, image]，temperature=0，max_tokens=2048，见 `pc_agent_distill/蒸馏pipeline说明.md`）。
