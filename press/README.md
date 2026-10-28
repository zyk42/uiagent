# uiagent 模型服务压测指南

> 维护: zhangyikang1 · 更新: 2026-10-21
> 压测脚本: `/root/bench.py`(dev 机) · 历史报告: `/root/bench-report.md`

## 1. 压测集格式

压测对象是 OpenAI 兼容的 `/v1/chat/completions` 接口(多模态 GUI agent 模型),数据集为 **JSONL,每行一条完整请求**,两种可用格式:

### 1.1 原始 chat 格式(推荐,自写脚本用)

每行一个 JSON 对象,`messages` 为 OpenAI 多模态消息数组,图片用 **base64 data URL** 内联:

```json
{"model": "base_model", "temperature": 0, "messages": [
  {"role": "system", "content": "# Role\nYou are a GUI interaction agent..."},
  {"role": "user", "content": "任务指令文本"},
  {"role": "assistant", "content": "历史轮输出"},
  {"role": "user", "content": [
    {"type": "text", "text": "当前指令"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
  ]}
]}
```

要点:

- 图片必须内联 base64(`data:image/jpeg;base64,...`),脚本直接把 `messages` 原样 POST 给服务,不做任何改写,**负载与线上一致**。
- 示例数据: `/root/压测数据.jsonl`(500 条,平均 prompt ≈3700 tokens,其中图片 ≈2550 tokens)。
- PC agent 数据: `press_pc_512.jsonl`(512 条,win/mac/ubuntu 三平台均衡,prompt 3757±244 tokens,completion 118±15 tokens,窄带选样降低测量误差;vLLM custom 格式见 `press_pc_512_prompt.jsonl`,元信息见 `press_pc_512.meta.jsonl`)。
- `model` / `temperature` 字段会被脚本覆盖(见 §3),写什么都行。

### 1.2 vLLM custom 格式(cml 内置 benchmark 用)

cml 的 `cml service benchmark` 底层是 vLLM `bench serve` 的 `custom` 数据集,要求每行**必须有 `prompt` 列**。两种写法:

纯文本:

```json
{"prompt": "一段纯文本 prompt"}
```

带图(`prompt` 写成 OpenAI content parts 数组,已实测可用):

```json
{"prompt": [
  {"type": "text", "text": "system + 历史 + 指令的全部文本拼接"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
]}
```

注意: 此格式把多轮对话**拍平成一条 user 消息**,chat template 结构和线上略有差异(token 数基本一致,语义结构有差)。转换脚本逻辑见 §4.2;已转换好的示例: `/mnt/lisiqi23/internship/zhangyikang/press-500-prompt.jsonl`。

### 1.3 采样与规模建议

- 条数 ≥ 最大档位的 `rps × duration`(如 RPS 4 × 60s = 240 条),建议 500+ 条以覆盖多样性。
- 必须来自**真实线上分布**(真实截图尺寸、真实指令长度),否则 TTFT/吞吐会失真。
- 各对比实验必须用**同一份数据、同一顺序**(脚本按 `records[i % N]` 取,天然保证)。

## 2. 压测原理

### 2.1 开环固定速率(open-loop fixed-RPS)

脚本按**固定到达速率**发请求: 第 i 条在 `t0 + i/rps` 时刻发出,不管服务端是否处理得完。

- 测的是"给定负载水平下,延迟表现如何",不是"服务最多能跑多快"。
- 判断饱和的信号: 实际吞吐(输出 tok/s)不再随 RPS 增长、TTFT/E2E 开始非线性上涨、墙钟时间 > 发送窗口(请求积压)。
- 对比"闭环固定并发"模式: 闭环是回来一个补一个,永远打不爆服务;要测容量极限必须用开环。

### 2.2 采集指标(逐条记录)

每条请求一行 jsonl,字段:

| 字段 | 含义 | 测量方式 |
|---|---|---|
| `ttft` | 首 token 延迟 | 发送 → 收到第一个非空 content chunk(流式) |
| `e2e` | 端到端延迟 | 发送 → 流结束 |
| `tpot` | decode 阶段每 token 平均耗时 | `(e2e 中 decode 段) / (completion_tokens - 1)` |
| `prompt_tokens` / `image_tokens` / `cached_tokens` | 输入构成 | 从流式响应的 `usage` 读取(`stream_options.include_usage=true`) |
| `completion_tokens` | 输出长度 | 同上 |
| `queue_delay` | 客户端调度偏差 | 实际发送时刻 - 计划发送时刻(应 ≈0,验证客户端没成为瓶颈) |
| `http_status` / `finish_reason` / `error` | 状态 | — |

聚合: mean / p50 / p90 / p95、成功率、实测 RPS、输入/输出 token 吞吐。

### 2.3 可比性守则

1. **同数据同顺序**: 所有被测服务跑同一份 jsonl 的同一切片。
2. **temperature=0**: 输出尽量确定,减少服务间输出长度差异。
3. **不截断输出**: 不加 `ignore_eos`,让模型自然停止——截断会扭曲投机解码等有状态优化的行为;用 `max_tokens=512` 做上限兜底即可。实测同数据下各服务输出长度分布一致(均值 243~245 tokens)。
4. **p95 而不是 p99**: 60s 档位样本量 60~240 条,p99 只有 1~2 个样本,噪声大;p95 更稳。
5. **每档 60s**: 比 30s 显著降低随机误差;档间冷却 15s 让队列排空。

## 3. 压测流程(以 CloudML 模型服务为例)

### 3.1 部署服务

用 cml CLI 在服务下创建部署(参考 `/root/bench-report.md` 里的四轮配置):

```bash
cml service deployment create --filename deploy.yaml \
  --service-id s-20260902173040-8ly5q --workspace-id 10037 --dry-run -o json   # 先预演
cml service deployment create --filename deploy.yaml \
  --service-id s-20260902173040-8ly5q --workspace-id 10037                    # 确认后执行
cml service deployment wait --deployment-id <dep-id> --workspace-id 10037 --for running
```

部署 YAML 关键项: `queueId`(队列)→ `resourceConfig`(`numPerInstance` = 每实例 GPU 数)→ `imageConfig.imageCommand`(SGLang 启动命令,dp/tp 在这里配)。

注意: deployment 级 `trafficPorts` **不要重复声明服务级 HTTP 端口**(会报端口冲突),整个不写即可继承。

### 3.2 等端点就绪

平台显示 running ≠ 服务可调用(SGLang 还在加载模型)。以真实请求探活:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  http://<deployment-id>.kscn-tj5-cloudml.xiaomi.srv/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"base_model_sglang","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
```

返回 200 才可开测(`/tmp/wait-and-bench.sh` 就是这个探活+自动开测的循环)。

### 3.3 运行压测

```bash
# 编辑 /root/bench.py 顶部: DEPS(部署名→URL)、RPS_LEVELS、DURATION、MAX_TOKENS
nohup python3 /root/bench.py > /root/bench-run.log 2>&1 &
```

脚本行为: 多个部署并行、每个部署内部按 RPS 档位串行升档;每档输出 `<outdir>/<dep>-rps<N>.jsonl`(逐条)+ 最后 `summary.json`(聚合)。

### 3.4 收尾

```bash
# 停掉部署(保留配置) / 删除部署(释放且不可恢复)
cml service deployment stop   --deployment-id <dep-id> --workspace-id 10037
cml service deployment delete --deployment-id <dep-id> --workspace-id 10037 --wait
```

## 4. 工具附录

### 4.1 自写脚本 vs cml 内置 benchmark

| | 自写脚本 `/root/bench.py` | `cml service benchmark` |
|---|---|---|
| 数据格式 | 原始 OpenAI messages(§1.1),不改写 | vLLM custom(§1.2),多轮被拍平 |
| 逐条记录 | ✅ 每请求一行 jsonl | ❌ 只有聚合指标(report)+ 日志 |
| 限流 | 无 | 创建接口有用户级限频,同部署串行 |
| 适用 | 多模态/精确对比/要逐条明细 | 纯文本快速冒烟 |

### 4.2 chat 格式 → vLLM custom 格式转换

```python
import json

def convert(rec):
    parts = []
    for m in rec["messages"]:
        c = m["content"]
        if isinstance(c, str):
            if c.strip():
                parts.append({"type": "text", "text": c})
        else:
            for p in c:
                if p["type"] == "text" and p["text"].strip():
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image_url":
                    parts.append(p)
    return {"prompt": parts}

with open("in.jsonl") as f, open("out.jsonl", "w") as g:
    for line in f:
        g.write(json.dumps(convert(json.loads(line)), ensure_ascii=False) + "\n")
```

cml benchmark 的数据集需放 JuiceFS(`datasetSource: {type: JuiceFS, volume, juiceFsCluster, path}`),本 dev 机已挂载 `xiaoai-guiagent`(cluster `ak-cloudml`)和 `lisiqi23`(cluster `tj5-common`),直接 `cp` 进 `/mnt/<volume>/` 即可。

## 5. 历史实验索引(详见 /root/bench-report.md)

| 轮次 | 变量 | 结论 |
|---|---|---|
| 1 | int4 + DFLASH,dp4/dp2tp2/dp1tp4 | 4 卡下 dp4 最优 |
| 2 | int4 + DSPARK | 饱和场景 DSPARK 优于 DFLASH(吞吐 +6%) |
| 3 | int4 无投机 | 投机解码带来 1.3x~2.7x E2E 加速 |
| 4 | bf16 原模型 tp4 无投机 | 量化开销小(~3%),差距主要来自投机解码 |
