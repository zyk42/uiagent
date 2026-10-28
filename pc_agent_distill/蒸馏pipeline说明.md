# GUI-Owl-32B 蒸馏 Pipeline 说明

> 数据流向总览：
> ```
> AgentNet 原始轨迹 (agentnet_win_mac_18k.jsonl)
>   │  ① convert_to_pcagent.py：转成 PC-Agent prompt 格式
>   ▼
> agentnet_win_mac_pcagent.jsonl  (350,013 条样本, 13G)
>   │  ② distill_32b.py：封装成 OpenAI 请求发给 GUI-Owl-32B (sglang dp=8)
>   ▼
> distill_32b_raw.jsonl  (模型原始输出，边跑边写)
>   │  ③ 后处理：去重 / 解析 / join 回完整样本
>   ▼
> 最终 SFT 训练数据
> ```

---

## 一、数据是怎么封装并发送给模型的

### 1.1 输入样本（agentnet_win_mac_pcagent.jsonl）

每条样本关键字段：

```jsonc
{
  "sample_id": "20240930..._s0_0",
  "prompt": "### Background ###\nThis image is a computer screenshot ... ",  // 完整 PC-Agent 决策提示词
  "image_tar": ["shard_0007.tar", 123456, 375231],  // 截图在 tar 分片里的位置
  "image_width": 1280, "image_height": 720,          // 图片真实尺寸
  "action": "Tap (163, 271)",                        // AgentNet GT（仅作对照，不发给模型）
  "messages": [...]                                   // 预构造的消息（见下）
}
```

`prompt` 由 PC-Agent 源码 `get_action_prompt()` 生成，结构：

```
### Background ###            截图尺寸 + 用户指令
### Screenshot information ### （本数据为空——无 OCR/a11y 标注，走纯视觉 + [0,999] 归一化坐标模式）
### History operations ###    前序步骤 [Operation; Action; Reflection]（按轨迹累加）
### Last operation ###        （仅当上一步被标注错误时注入纠错提示）
### Task requirements ###     12 个动作的定义（Tap/Double Tap/Right Tap/Type/Shortcut/Press/Select/...）
### Output format ###         要求输出 {"Thought","Action","Summary"} JSON
```

### 1.2 图片读取（tar + 索引随机访问）

散文件不存在（JuiceFS 小文件创建太慢），图片全部在 tar 分片中：

```python
shard, off, size = record["image_tar"]
with open(f"win_mac_tar/{shard}", "rb") as f:
    f.seek(off)
    png_bytes = f.read(size)          # 原始 PNG 字节
img_b64 = base64.b64encode(png_bytes) # 转 base64 随请求发送
```

### 1.3 请求构造（distill_32b.py）

OpenAI 兼容接口，`POST http://localhost:30000/v1/chat/completions`：

```jsonc
{
  "model": "default",
  "max_tokens": 2048,        // 不截断（实测平均输出 130 tokens，最大 ~246）
  "temperature": 0.0,        // 确定性输出，蒸馏标签可复现
  "messages": [
    {"role": "system",
     "content": "You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction."},
    {"role": "user", "content": [
      {"type": "text", "text": "<完整 prompt>"},
      {"type": "image_url",
       "image_url": {"url": "data:image/png;base64,<base64编码的截图>"}}
    ]}
  ]
}
```

注意点：
- **没有 tools / function calling 参数**——与 PC-Agent 原生请求方式完全一致，动作空间纯靠 prompt 定义
- system prompt 来自 PC-Agent 源码 `chat.py: init_action_chat()` 原文
- 图片以 data URL（base64）内联，PC-Agent 原版也是这么做的（区别仅是它把图缩到 0.5 倍，我们用原图）
- 上下文 32768（服务启动参数 `--context-length 32768`），实际每条请求 prompt ~2000 tokens（含图片 ~900 vision tokens）+ 输出 ≤2048，余量充足

### 1.4 并发与容错

- asyncio + `Semaphore(256)` 并发；外层每 2000 条一个 gather 批次
- 每条失败重试 3 次（指数退避 2s/4s/6s），仍失败记 `error` 字段继续
- **断点续跑**：启动时读已有输出文件里的 sample_id 集合，跳过已完成
- 进度：每 5 秒打印 `done/total | req/s | parse_ok | fail | ETA` 到日志，并写 `distill_32b_status.json`

### 1.5 服务端

```
python3 -m sglang.launch_server \
  --model-path .../GUI-Owl-1.5-32B-Instruct \
  --dp-size 8 --port 30000 \
  --mem-fraction-static 0.85 --context-length 32768
```

8 张 H20-3e 各放一个副本，sglang 内部路由做负载均衡。实测吞吐 ~4-5 req/s。

---

## 二、蒸馏输出的后处理

### 2.1 原始输出文件（distill_32b_raw.jsonl）

每条一行：

```jsonc
{
  "sample_id": "...",            // 与输入样本一一对应（join 键）
  "model_output": "{...}",       // 模型原始文本输出
  "model_thought": "...",        // 解析出的 Thought
  "model_action": "Tap (52, 278)",   // 解析出的 Action
  "model_summary": "...",        // 解析出的 Summary
  "parse_ok": true,              // JSON 是否解析成功
  "gt_action": "Tap (48, 289)",  // AgentNet GT 动作（评估对照）
  "prompt_tokens": 1863,
  "completion_tokens": 88,
  "latency_s": 23.2
}
```

**解析逻辑**（与 PC-Agent `run.py` 一致）：

```python
obj = json.loads(model_output.split('```json')[-1].split('```')[0].strip().strip('`'))
# 模型实际输出通常不带 ```json 围栏，此时 split 原样返回，直接 json.loads
```

### 2.2 去重

蒸馏中途曾因重启遗留导致双进程写同一文件，产生约 40% 重复行（内容基本相同，temperature=0）。
**跑完后必须按 sample_id 去重**：

```python
seen = set()
with open('distill_32b_raw.jsonl') as fin, open('distill_32b_dedup.jsonl', 'w') as fout:
    for line in fin:
        sid = json.loads(line)['sample_id']
        if sid in seen:
            continue
        seen.add(sid)
        fout.write(line)
# 校验：len(seen) 应等于 350,013
```

去重后顺手校验完整性：unique 数 = 350,013 即为全覆盖；缺的 sample_id 可收集后用同一脚本补跑（断点续跑机制会自动只跑缺的）。

### 2.3 组装最终 SFT 数据

蒸馏文件只存了模型输出，训练样本需要 join 回完整输入：

```python
# 1. 载入蒸馏输出（去重后）
distill = {}
with open('distill_32b_dedup.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['parse_ok']:
            distill[r['sample_id']] = r

# 2. join 回完整样本，替换 assistant 回答
with open('agentnet_win_mac_pcagent.jsonl') as fin, \
     open('sft_guiowl32b_pcagent.jsonl', 'w') as fout:
    for line in fin:
        d = json.loads(line)
        r = distill.get(d['sample_id'])
        if r is None:
            continue
        # 用模型输出替换 GT 回答；保留模型原始文本作为 assistant content
        resp = r['model_output']
        if not resp.strip().startswith('```'):
            resp = '```json\n' + resp.strip() + '\n```'   # 统一带围栏，对齐 PC-Agent 解析习惯
        d['messages'][-1] = {"role": "assistant", "content": resp}
        d['response'] = resp
        d['thought'] = r['model_thought']
        d['action'] = r['model_action']
        d['summary'] = r['model_summary']
        fout.write(json.dumps(d, ensure_ascii=False) + '\n')
```

### 2.4 建议的质量过滤（可选）

join 后的样本仍带有全部元信息，可按需过滤：

| 过滤条件 | 字段 | 说明 |
|---|---|---|
| 去掉解析失败 | `parse_ok == true` | 当前失败率 ~2.5% |
| 去掉失败轨迹 | `task_completed == true` | AgentNet 原始标注，GT 都没完成的轨迹蒸馏答案参考价值低 |
| 去掉错误步骤 | `this_step_correct == true` | AgentNet 逐步标注 |
| 去掉 fail 终止 | `terminated_with_fail == false` | 86 条 |
| 扩展动作 | `action` 含 Drag/Scroll/HScroll/Middle Tap | 约 10%，PC-Agent 原生动作空间没有，看训练目标决定是否保留 |
| 去掉续拆子步 | `is_continuation == false` | 多行代码拆出的后续原子步，thought 是模板生成的 |

### 2.5 失败样本补跑

`distill_32b_raw.jsonl` 里带 `error` 字段的行（重试 3 次仍失败，目前 ~75 条），
直接在蒸馏跑完后再执行一次 `python3 distill_32b.py` 即可——断点续跑只处理缺失的 sample_id；
若 error 行已写入文件，需要先删掉这些行再补跑：

```python
with open('distill_32b_raw.jsonl') as fin, open('clean.jsonl', 'w') as fout:
    for line in fin:
        if 'error' not in json.loads(line):
            fout.write(line)
```

---

## 三、文件清单

| 路径（均在 `/mnt/lisiqi23/internship/zhangyikang/uiagent/` 下） | 说明 |
|---|---|
| `pc_agent_distill/agentnet_win_mac_pcagent.jsonl` | 转换后的完整样本（prompt + GT + messages + image_tar） |
| `pc_agent_distill/win_mac_tar/shard_*.tar` + `tar_index.json` | 349,169 张截图 + 随机访问索引 |
| `pc_agent_distill/distill_32b_raw.jsonl` | 蒸馏原始输出（运行中，含重复） |
| `pc_agent_distill/distill_32b.log` / `distill_32b_status.json` | 进度日志 / 状态 |
| `pc_agent_distill/distill_32b_snapshot_20261014.jsonl` | 10-14 上午的去重快照（95,426 条，分析用） |
| `pc_agent_distill/convert_to_pcagent.py` | ① AgentNet→PC-Agent 格式转换 |
| `pc_agent_distill/distill_32b.py` | ② 蒸馏请求脚本 |
| `tools/multizip.py` `scan_index.py` `pack_to_tar.py` | 分卷 zip 修复/扫描/打包工具 |
| `logs/sglang_32b_dp8.log` | sglang 服务日志 |
