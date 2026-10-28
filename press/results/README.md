# press/results 索引(压测结果归档)

> 更新: 2026-10-28 · 方法见 `../README.md` · 计划见 `../部署以及压测.md` · 汇总报告 `../压测结果-第一轮.md`

## 目录命名

- 无后缀 = **sglang 栈**(镜像 `sglang0.5.20-dflash2`);`-vllm` 后缀 = **vLLM 栈**(镜像 `vllm-nvfp4-llmcompressor`)
- 形态后缀: 无 = tp4 dp1(4 卡);`-dp2tp2` = tp2 dp2;`-dp1tp4` = tp1 dp4
- 每个目录: `summary.json`(聚合)+ `<tag>-rpsN.jsonl`(逐条明细)

## sglang 栈(第一轮, 完整)

| 目录 | RPS4 吞吐 | 备注 |
|---|---|---|
| bf16 | 146 tok/s | 基线 |
| fp8-w8a8 / -dp2tp2 | 158 / 219 | |
| gptq-w4a16 / -dp2tp2 / -dp1tp4 | 168 / 237 / 293 | |
| awq-w4a16 / -dp2tp2 / -dp1tp4 | 168 / 238 / 295 | |
| nvfp4-w4a4 / -dp2tp2 / -dp1tp4 | 229 / 369 / **474** | |
| gptq-nvfp4 / -dp2tp2 / -dp1tp4 | 226 / 370 / 456 | |
| gptq-nvfp4-skip3 | 423 | ⚠️ 输出乱码(混合精度路径不兼容), 速度仅作栈参考 |
| gptq-nvfp4-skip6/12/24 | — | sglang 栈未测(skip3 乱码后决策停止) |
| smoothquant-w8a8 | — | 不可部署(int8 内核无 sm120) |

## vLLM 栈(第二轮)

| 目录 | RPS4 吞吐 | 备注 |
|---|---|---|
| bf16-vllm | 187 tok/s | ✅ 可信(比 sglang 146 快 28%) |
| gptq-nvfp4-skip3/6/12/24-vllm | 285 / 277 / 261 / 242 | ✅ skip 系列在 vllm 下输出正常 |
| *-dp2tp2-vllm (4bit×4) | 135~193 | ✅ 但比 sglang 同形态慢 ~40% |
| *-vllm tp4 (4bit×4) | 5~21 ⚠️ | **异常值**, 疑似落在故障节点, 勿作结论 |
| fp8-w8a8-vllm | — | unusable(fp8 路径 5090 退化 200 倍, 已弃) |
| *-dp1tp4-vllm (4bit×4) | — | deploy_failed(vllm dp=4 crash loop, 待查) |
| smoothquant-*-vllm | — | dropped(双栈出局) |

## 已知坑位(复跑前必读)

1. **vllm dp=4 起不来**: 4 个 dp1tp4 均 crash loop(2026-10-27), pod 日志需控制台查看
2. **vllm 4bit tp4 四组数字异常低**(5~21 tok/s, 大量请求超时), 与同镜像 skip 系列(240+)矛盾, 疑似故障节点; 结论使用前需复测
3. 服务单服务部署数上限 10, 自动化循环记得删除前 `wait --for stopped` 并校验删除成功

## vLLM + DFlash2 投机解码(第三轮)

命名: `<模型>-spec-vllm`(draft: `pc_agent_distill/dflash2/draft-part07-final-hf`,16 tokens)。全部 9 组有效:

| 目录 | RPS4 吞吐 | RPS1 E2E |
|---|---|---|
| bf16-spec-vllm | 184 tok/s | 1.79s |
| gptq-w4a16-spec-vllm | 190 | 1.32s |
| awq-w4a16-spec-vllm | 188 | 1.31s |
| nvfp4-w4a4-spec-vllm | **302** | 0.94s |
| gptq-nvfp4-spec-vllm | 300 | **0.86s** |
| skip3/6/12/24-spec-vllm | ~280 / 288 / 272 / 248 | 0.94~1.18s |

(skip3 首轮部署失败后于 10-28 重试成功)
