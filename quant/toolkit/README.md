# 量化工具包（toolkit）

> 从 dev 机 `/root/quant_calib` 同步（2026-10-27）。所有路径已重写为本目录，开箱即用。
> 主模型：`/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct`

## 数据

| 文件 | 说明 |
|---|---|
| `calib_pc_512.jsonl` | 量化校准集 v2：512 条，win/mac/ubuntu 三平台，17 类动作全覆盖，≤8192 token，与评估集零重叠。图片在 `pc_images_v2/` |
| `calib_pc_awq_64.jsonl` | AWQ 专用减小子集（44 条 ≤3072 token，AWQ 激活缓存内存敏感） |
| `pc_images_v2/` | 校准集引用的截图（518 张） |

评估集在 `../../eval/`（eval_pc_360.jsonl），压测集在 `../../press/`（press_pc_512.jsonl）。

## 量化主程序

```bash
python run_quant.py <method>   # gptq-w4a16 | awq-w4a16 | nvfp4-w4a4 | mxfp4-w4a4
                               # | smoothquant-w8a8 | fp8-w8a8 | gptq-nvfp4
                               # | gptq-nvfp4-skip3/6/12/24
```

- 产物输出到 `run_quant.py` 内 `OUT_BASE`（当前指向本卷 `quant/`）
- 校准集路径在同文件 `CALIB_JSONL`
- `run_<method>.sh` 为带环境变量的包装脚本（`setsid nohup bash run_<m>.sh > /dev/null 2>&1 < /dev/null &` 后台脱离终端）

## 部署与测试

| 脚本 | 用途 |
|---|---|
| `deploy_gptq_w4a16_dflash.sh` | **vLLM 部署**：gptq-w4a16 + DFlash2 投机解码（实测 decode 4.2×） |
| `bench_decode.py` / `bench_one.py` / `bench_all.sh` | decode 速度 A/B / 单模型全档测速 / 批量 |
| `smoke_test.sh` | 起服务→发请求→关服务 冒烟 |

## 数据集构建脚本（可复现）

`make_calib_v2.py`（三平台校准集）、`make_calib_awq.py`（AWQ 子集）、`make_eval_set_v2.py`（评估集）、`make_press_set.py`（压测集）、`repair_images.py`（图片修复）。

## 环境记录

`pip_snapshot_before_vllm.txt`（装 vllm 前的 pip freeze）、`bench_results.txt`（sglang 时代四模型测速）、`smoke_results.txt`（sglang 时代部署验证）。

注意：部分脚本内引用 dev 机绝对路径（如 `/root/quant_calib/logs`），换机使用时自行调整。
