# PC-Agent 对话历史构建机制

> 分析对象：`/root/MobileAgent-main/PC-Agent/`（新版，`run.py` + `PCAgent/`）
> 核心结论：**PC-Agent 不回传任何历史截图。每个模块的调用都是全新构建的单轮对话，
> 历史信息全部以纯文本摘要形式拼进 prompt。**

---

## 一、每次调用发几张图？

| 模块 | 图片 | 代码位置 | 说明 |
|---|---|---|---|
| 决策 Action | **1 张**：当前步的 SOM 标注截图 | `run.py:656-660` | `init_action_chat()` 全新初始化 |
| 反思 Reflection | **2 张**：操作前 + 操作后截图 | `run.py:862-863` | 用于对比操作效果 |
| 进度 Planning | **1 张**：当前步截图 | `run.py:897` | 验证真实完成度 |
| 记忆 Memory | 复用决策 chat（默认关闭） | `run.py:832-836` | `memory_switch=False` |

决策模块的典型代码：

```python
# run.py 每步循环内
prompt_action = get_action_prompt(sub_instruction, perception_infos, width, height,
                                  thought_history, summary_history, action_history,
                                  reflection_history, ...)
chat_action = init_action_chat()                     # ← 每步都是空对话重新开始！
chat_action = add_response("user", prompt_action, chat_action, [screenshot_som_file])
output_action = inference_chat(chat_action, vl_model_version, API_url, token)
```

`init_action_chat()`（`PCAgent/chat.py:12`）只放一条 system prompt，之后立刻 append 当前步的
user 消息——**没有任何 assistant/user 的多轮累积**。

---

## 二、历史信息的传递方式：文本摘要

历史通过四个 Python 列表在循环中累加（`run.py:888-892`）：

```python
thought_history.append(thought)        # 模型当步的 Thought
summary_history.append(summary)        # 当步的 Summary（一句话操作描述）
action_history.append(action)          # 当步的 Action 字符串
reflection_history.append(reflection_thought)  # 反思模块的文本结论
```

下一步的 prompt 里，`get_action_prompt()` 把它们格式化成（`prompt_qwen.py:158-166`）：

```
### History operations ###
Before arriving at the current screenshot, you have completed the following operations:
Step-1: [Operation: Click on cell C2, which contains...; Action: Tap (163, 271); Reflection: The action has successfully selected cell C2...]
Step-2: [Operation: Click the Home tab; Action: Tap (313, 82); Reflection: ...]
```

注意细节：
- `Operation` 取的是 `summary.split(' to ')[0]`——只保留摘要的动作部分
- 只有当反思开启时（`reflection_history` 非空）才带 Reflection 字段
- 上一步被判定失败时，还会在 `### Last operation ###` 段落单独注入纠错提示

---

## 三、子任务级别的历史清空

`--clear_history_each_subtask 1`（默认开，`run.py:631-635`）：

```python
if args.clear_history_each_subtask == 1:
    thought_history = []
    summary_history = []
    action_history = []
    reflection_history = []
```

每开始一个新子任务，四个历史列表全部清零。因为 PC-Agent 是层级结构（Manager 拆子任务 →
逐个子任务执行），子任务之间靠 `answer_dict` 传参数（`Tell` 动作写入），而不是靠历史。

跨子任务保留的只有：
- `answer_dict`（子任务间传递的参数字典）
- `completed_requirements`（Planning 产出的进度文本，**会被累积**）

---

## 四、消息结构（chat.py）

自定义的简单结构，非 OpenAI 多轮：

```python
operation_history = [
    ["system", [{"type": "text", "text": "You are a helpful AI PC operating assistant. ..."}]],
    ["user",   [{"type": "text", "text": "<prompt>"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]],
]
```

`add_response()` 每次 deepcopy 后 append。虽然技术上支持多轮（决策后会把 assistant 输出
append 回去再接 Memory 提问），但默认配置下 Memory 关闭，**每个模块实际就是
system + 单条 user 的单轮请求**。

---

## 五、为什么不传历史截图

这是一个明确的成本/效果取舍：

1. **token 成本**：一张截图约 900~2000 vision tokens。若回传全部历史，第 20 步要携带
   ~20 张图 ≈ 2-4 万 token，每步成本线性膨胀；而文本历史每步只有几十个 token。
2. **注意力稀释**：长串历史截图会稀释模型对当前截图的关注，且 GUI 任务里真正重要的
   几乎总是"当前画面 + 我刚做了什么"。
3. **信息损失由其他模块补偿**：
   - Reflection 用"前后双图"验证上一步效果 → 纠错信号进入文本历史
   - Planning 每步用当前图验证真实进度 → `completed_requirements` 累积
   - SOM 标注 + OCR/a11y 文本让单张图的信息量足够高

代价也很明显：如果文本摘要写错了（比如 Summary 声称成功但实际失败），模型没有画面
可以纠正它——这正是 Reflection/Planning 两个验证模块存在的原因。

---

## 六、对蒸馏数据的映射

转换后的训练数据完全复刻了这个形态：

```jsonc
// agentnet_win_mac_pcagent.jsonl 每条样本
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI PC operating assistant. ..."},
    {"role": "user", "content": [
      {"type": "text", "text": "<含文本历史的 prompt>"},
      {"type": "image", "image": "<仅当前步一张截图>"}
    ]},
    {"role": "assistant", "content": "<```json {Thought, Action, Summary} ```>"}
  ]
}
```

- **单图**：只带当前步截图，与 PC-Agent 线上推理形态一致
- **文本历史**：从 AgentNet 轨迹逐步累加生成（Step-i: [Operation; Action; Reflection]）
- Reflection 字段用 AgentNet 的逐步事后标注填充
- 每个轨迹（≈子任务）开始时历史清空，与 `clear_history_each_subtask=1` 行为一致

所以训练出来的模型，期望的推理上下文就是"当前一张截图 + prompt 里的文本历史"，
部署到 PC-Agent 管线时无需任何适配。
