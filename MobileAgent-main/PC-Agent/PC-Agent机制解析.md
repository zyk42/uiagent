# PC-Agent 机制解析文档

> 代码位置：`PC-Agent/`。仓库中包含两个版本：
> - `PCAgent/` + `run.py`：**新版（2025-02 更新，论文 arXiv:2502.14282）**，本文档主要分析对象。
> - `PCAgent_v1/` + `run_v1.py`：旧版（2024-08），文末附差异对比。

---

## 1. 总体架构

PC-Agent 是一个**层级化多智能体协作框架**，用一句话概括：

```
用户指令
  → [任务分解 Manager] 拆成子任务序列（支持子任务间参数传递）
    → 对每个子任务循环执行：
        [感知 Perception] 截图 + OCR + 无障碍树(a11y) → 标注图(SOM) + 元素坐标列表
        [决策 Decision]   VLM 输出一个动作（JSON 格式）
        [执行 Execution]  字符串解析动作 → pyautogui/pynput 执行
        [反思 Reflection] 对比操作前后两张截图，判断操作是否符合预期（A/B/C/D）
        [进度 Planning]   更新"已完成内容"，供下一步决策参考
```

**关键点：PC-Agent 没有使用 OpenAI 的 function calling / tool use API**。它的"function call"是**纯 prompt 工程 + 字符串解析**：

1. 在 prompt 里用自然语言定义一组"动作空间"（Action Space）；
2. 要求模型以固定格式（JSON）输出 `Thought / Action / Summary` 三个字段；
3. `run.py` 用字符串匹配（`"Tap" in action` 等）解析出动作名和参数；
4. 分发到对应的 Python 函数（`tap()`、`open_app()`、`shortcut()`…）执行真实的鼠标键盘操作。

---

## 2. 运行入口与主循环（run.py）

### 2.1 启动方式

```bash
# Windows
python run.py --instruction="Open Chrome and search the PC-Agent paper." --mac 0
# Mac
python run.py --instruction="..." --mac 1
```

关键命令行参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--instruction` | "" | 用户指令 |
| `--simple` | 1 | 1=跳过任务分解，整个指令作为单个子任务 |
| `--use_som` | 1 | 是否给截图画上编号标注（Set-of-Mark） |
| `--use_a11y` | 1 | 1=用系统无障碍树获取元素；0=用 GroundingDINO 图标检测模型 |
| `--use_perception_info` | 1 | 0=纯截图，不注入 OCR/a11y 文本信息 |
| `--disable_reflection` | 1 | 0=开启反思 Agent |
| `--clear_history_each_subtask` | 1 | 每个子任务开始时清空操作历史 |
| `--num_step_limit` | 20 | 总步数上限 |
| `--ratio` | 1.0 | 截图像素/逻辑坐标比例（Mac Retina 为 2） |
| `--ocr_api` | 1 | 1=用阿里云 OCR API；0=本地 modelscope OCR 模型 |

模型配置在 `config.json`：

```json
{
  "vl_model_name": "gpt-4o",   // 多模态模型：决策/反思/进度都用它
  "llm_model_name": "gpt-4o",  // 文本模型：只用于任务分解
  "token": "sk-...",
  "url": "https://api.openai.com/v1",
  "OCR_ACCESS_KEY_ID": "...",  // ocr_api=1 时需要
  "OCR_ACCESS_KEY_SECRET": "..."
}
```

### 2.2 主循环流程（run.py 的执行顺序）

```
1. 读 config.json 拿模型名/token
2. 若 simple=0：调用 LLM 做任务分解 → subtask_dict（最多重试 5 次，含格式校验）
3. 遍历每个子任务：
   a. 若子任务是"格式化字符串"（含 {key} 占位符），从 answer_dict 里取前面子任务
      产出的值做参数替换
   b. while 循环（每步）：
      - get_perception_infos(): 截图 → OCR + a11y → 画 SOM 标注图 → perception_infos 列表
      - get_action_prompt(...) 拼决策提示词 → VLM 推理
      - 从输出中提取 ```json 块 → 解析 Thought/Action/Summary
      - 坐标反归一化（模型输出 0-1000 归一化坐标 → 乘回宽高）
      - 字符串匹配分发执行动作（见 §4）
      - 若是 Tell/Stop → 结束该子任务
      - 重新截图、重新感知
      - 若开启反思：get_reflect_prompt + 前后两张截图 → VLM 判 A/B/C/D
      - get_process_prompt → VLM 更新"已完成内容" completed_requirements
      - 历史列表追加 thought/summary/action/reflection
```

注意一个设计细节：**每一步的决策 chat 都是全新初始化的**（`init_action_chat()`），历史不是通过多轮对话累积，而是把 `thought_history / summary_history / action_history / reflection_history` 以**纯文本形式拼进 prompt** 的 `### History operations ###` 段落。这样每步都是单轮调用，上下文长度可控。

---

## 3. 提示词设计（PCAgent/prompt_qwen.py）

所有提示词都是**代码里硬编码的英文模板**，由若干函数按需拼装。共 6 类：

### 3.1 任务分解提示词 `get_subtask_prompt(instruction)`

- 用 **文本模型**（llm_model_name）调用，不截图。
- 核心设计：要求把复合指令拆成 4 种类型的子任务：
  1. **常规字符串**：`"在系统设置中，打开深色模式"`
  2. **含字典输出的字符串**：结果要以 dict 传给后续子任务，如 `"查看Paul的邮件，输出 {'contact': 'Paul', 'mail_content': '...'}"`
  3. **格式化字符串**：用前序子任务的 dict key 补全，如 `"将 {mail_content} 发送给 Joey"`
  4. **既是格式化又含字典输出**：如 `"搜索 {question}，输出 {'info': '...'}"`
- 约束：每个子任务必须能**独立执行**（不能出现"该文档"这种指代）、必须指明应用（"在 Chrome 中"）、格式化 key 必须在前序子任务的 dict 中有对应输出。
- 输出格式为 `{"subtask 1": ..., "subtask 2": ...}` 的 JSON。
- `run.py` 侧的配套机制：`check_subtask_dict()` 校验 key 匹配；`answer_dict` 作为"通信枢纽"——子任务用 `Tell ({'key': 'value'})` 动作把结果写入 `answer_dict`，后续子任务的 `{key}` 占位符通过 `format_map` 替换。

### 3.2 决策提示词 `get_action_prompt(...)`（核心）

按段落拼装：

| 段落 | 内容 |
|---|---|
| `### Background ###` | 说明这是带编号标注的电脑截图，宽 w 高 h 像素，用户指令是 xxx；附 `add_info`（人工注入的操作知识，如"浏览器搜索点顶部搜索栏"） |
| `### Screenshot information ###` | 感知结果列表，每行 `[x, y]; mark number: N icon: xxx` 或 `[x, y]; text: xxx`（坐标是元素中心点像素坐标） |
| `### History operations ###` | 每步一条 `Step-i: [Operation: ...; Action: ...; Reflection: ...]` |
| `### Progress ###` | Planning Agent 产出的"已完成内容" |
| `### Memory ###` | 记忆模块内容（默认关闭） |
| `### Last operation ###` | 仅当上一步反思失败（error_flag=True）时注入，提示"上一步未达预期，需要修正" |
| `### Task requirements ###` | **动作空间定义**（见 §4）+ 坐标格式说明 |
| `### Output format ###` | 要求输出 JSON：`{"Thought": ..., "Action": ..., "Summary": ...}`，一次只许一个动作 |

坐标约定有个分支逻辑：有 perception 信息时用**像素坐标**（与 Screenshot information 一致）；`use_perception_info=0` 时让模型输出 **[0,999] 归一化坐标**，`run.py` 再换算回像素。

`Stop` 动作的描述被刻意强化过："只有当你能从**当前截图**验证所有要求都**真正完成**时才能 Stop，不能只做完了操作就停"——这是为了防止模型"自以为完成"。

### 3.3 反思提示词 `get_reflect_prompt(...)`

- 输入：**操作前截图 + 操作后截图** + 前后的 perception 信息 + 本次操作的 thought/action。
- 要求模型验证两件事：操作结果是否符合预期；**指令的真实目标是否达成**（不只是动作执行成功）。
- 输出 `### Thought ###` + `### Answer ###`，答案是四选一：
  - **A**：符合预期且目标达成
  - **B**：操作到了错误页面，需要纠正
  - **C**：操作没有产生任何变化
  - **D**：看似完成，但截图显示真实目标未达成（如点错位置、选错项）
- `run.py` 中 B 或 D → `error_flag = True`，下一步的 action prompt 会带上"上一步失败"的提示。

### 3.4 进度提示词 `get_process_prompt(...)`（Planning Agent）

- 每步结束后调用，传入当前截图 + 历史操作 + 上次的"已完成内容"。
- 要求**对照当前截图验证**真实完成度（示例特意举了"播放《稻香》必须确认屏幕上真的在播，而不是点了就完事"），输出 `### Completed contents ###` 更新进度。
- 该进度会注入下一步的 action prompt 的 `### Progress ###`。

### 3.5 记忆提示词 `get_memory_prompt(insight)`（默认关闭）

问模型"当前页面上有没有与指令/重要内容相关的内容值得记录"，输出追加进 `memory` 字符串，注入后续 action prompt 的 `### Memory ###`。

### 3.6 文本选择提示词 `get_select_prompt(content)`

服务于 `Select (content)` 动作：让 VLM 输出指定文本范围的**首行和末行**内容（`<first>...</first><last>...</last>`），然后 `run.py` 用 OpenOCR 在截图里定位这两行的坐标，执行一次**鼠标拖拽**完成选中。

### 3.7 System Prompt（PCAgent/chat.py）

非常简短，按角色区分：

- 决策：`"You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction."`
- 反思/记忆：`"You are a helpful AI PC operating assistant."`
- 任务分解：`"You are a helpful AI assistant."`

消息结构是 OpenAI 多模态格式：`content = [{"type":"text",...}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]`，截图缩放 0.5 后 base64 编码（`api.py: resize_encode_image`）。

---

## 4. "Function Call" 机制：动作空间与执行分发

### 4.1 动作空间（在 prompt 中自然语言定义）

新版动作空间（`get_action_prompt` 中列出）：

| 动作 | 参数 | 语义 | 对应 Python 函数 |
|---|---|---|---|
| `Tap (x, y)` | 像素坐标 | 单击 | `tap(x, y, 1)` → pynput 左键 |
| `Double Tap (x, y)` | 坐标 | 双击（打开文件） | `tap(x, y, 2)` |
| `Triple Tap (x, y)` | 坐标 | 三击（选段落） | `tap(x, y, 3)` |
| `Right Tap (x, y)` | 坐标 | 右键（新建文件菜单） | （代码中未实现对应分支） |
| `Shortcut (key1, key2)` | 组合键 | command/ctrl + s/a/n/t/c/v 等 | `shortcut()` → pyautogui keyDown/Up |
| `Press (key)` | 单键 | enter/backspace/方向键 | `presskey()` |
| `Type (x, y), (text)` | 坐标+文本 | 点击输入框、打字、回车 | `tap_type_enter()`（中文走剪贴板粘贴） |
| `Select (content)` | 文本描述 | 选中文档中某段文字 | `select()`：VLM 定首末行 + OCR 定位 + 拖拽 |
| `Replace (x, y), (text)` | 坐标+文本 | 全选替换内容 | `replace()`：双击 + ctrl+a + 输入 |
| `Append (x, y), (text)` | 坐标+文本 | 在指定位置后追加 | `append()` |
| `Tell (answer)` | dict | 子任务产出结果，写入 answer_dict | 解析后存入 `answer_dict` |
| `Stop` | 无 | 子任务完成 | break 跳出循环 |

### 4.2 解析与分发（run.py 698-828 行）

不是结构化 tool call，而是一串 `elif "xxx" in action:` 字符串匹配，用 `split("(")` / `split(")")` 抠参数。模型输出示例：

```json
{"Thought": "I need to click the search bar...", "Action": "Tap (960, 540)", "Summary": "Click the search bar to start searching"}
```

提取方式：`json.loads(output.split('```json')[-1].split('```')[0])`。

### 4.3 平台抽象

- Mac：`pymac.py` 的 `MacOSACI`（AXUIElement 无障碍 API），ctrl 键映射为 `command`，搜索快捷键 `command+space`，`ratio=2`（Retina）。
- Windows：`pywin.py` 的 `WindowsACI`（pywinauto + win32），搜索快捷键 `win+s`，`ratio=1`。
- 两个 ACI 类主要提供 `linearize_and_annotate_tree()`（遍历无障碍树拿到元素的位置/大小/文本/角色）和 `filter_ocr_elements()`（用 IoU 去掉与 a11y 元素重复的 OCR 框）。
- 真实输入操作统一由 `pyautogui` + `pynput` + `pyperclip` 完成。

### 4.4 感知管线（`get_perception_infos`）

```
pyautogui 截图
  ├─ 文本：OCR（阿里云 API 或本地 modelscope 模型）→ merge_boxes_and_texts 合并
  │        → 过滤超长文本(text_len_thre=1000)
  └─ 图标：use_a11y=1 → 系统无障碍树拿交互元素（默认）
            use_a11y=0 → GroundingDINO 检测 "icon"
  → a11y 与 OCR 结果去重（IoU）
  → 在截图上画彩色编号框 = SOM 图（Set-of-Mark）
  → 每个元素取中心点坐标，生成 perception_infos:
      {"text": "mark number: 5 icon: 微信", "coordinates": [x, y]}
```

这份列表既是 prompt 的 `### Screenshot information ###`，也存盘为 `perception_infos_stepN.json`。

---

## 5. 模型调用层（PCAgent/api.py + chat.py）

- 走 **OpenAI 兼容协议**（`openai.OpenAI(base_url=url, api_key=token)` + `chat.completions.create`），因此 GPT-4o、Qwen-VL（百炼 dashscope 兼容端点）、豆包等都能接。
- 每次调用重试 5 次，间隔 2 秒。
- 没有 streaming、没有 tools 参数、没有 temperature 显式设置。
- 图片统一缩放到 0.5 倍再 base64，省 token。
- 对话历史用自定义的 `[[role, content], ...]` 列表维护（`chat.py`），但**决策环节每步重建**，只把 assistant 的上一步输出加回当前 chat 用于记忆模块（默认关闭时其实没用上）。

---

## 6. 新版（PCAgent/）与旧版（PCAgent_v1/）的主要差异

| 维度 | v1 (run_v1.py) | v2 (run.py) |
|---|---|---|
| 决策输出格式 | 纯文本段落 `### Thought ### / ### Action ### / ### Operation ###` | JSON `{"Thought","Action","Summary"}`（```json 代码块） |
| 动作空间 | Open App / Tap / Double Tap / Triple Tap / Shortcut / Press / Type / Tell / Stop | 去掉 Open App（提示词里不再列出），新增 Right Tap / Select / Replace / Append |
| 图标感知 | GroundingDINO 图标检测 + OCR + icon caption | 默认改用系统**无障碍树**（a11y），GroundingDINO 为备选 |
| 反思输出 | 类似 A/B/C 判定 | A/B/C/D 四类，显式区分"动作成功但目标未达成"(D) |
| Stop 条件 | "历史操作完成即可 Stop" | 必须**从当前截图验证**目标达成才允许 Stop |
| 进度管理 | 有 process prompt | 加强：传入当前截图做验证式 Planning |

---

## 7. 代码中值得注意的坑（阅读源码时发现）

1. **坐标反归一化条件恒为真**（run.py:678）：
   ```python
   if "qwen3-vl" or "doubao" in vl_model_version.lower():
   ```
   `"qwen3-vl"` 是非空字符串恒真，所以**任何模型**都会走归一化坐标还原（x*width/1000）。如果模型输出的是像素坐标会被错误缩放。
2. **TapIdx 分支用错索引**（run.py:726-730）：`coordinate = perception_infos[i]['coordinates']` 用的是外层子任务循环变量 `i`，而不是解析出的 `idx`。`Triple Tap` / `Triple TapIdx` 两个分支的 idx/coordinate 解析也写反了。
3. **Right Tap** 在 prompt 里定义了，但执行分发里没有对应分支。
4. 反思判定 `"D" in reflect` 是子串匹配，理论上可能被误触发（实际输出格式固定为单个字母，风险不大）。

---

## 8. 复现运行要点

```bash
conda create --name pcagent python=3.10
pip install -r requirements.txt          # Windows
# pip install -r requirements_mac.txt    # Mac
git clone https://github.com/Topdu/OpenOCR.git && pip install openocr-python
# 编辑 config.json 填入 vl_model_name / llm_model_name / token / url
python run.py --instruction="打开Chrome搜索PC-Agent论文" --mac 0 --simple 1
```

运行产物保存在 `task_1/`（可用 `--screenshot_root` 改）：每步的原始截图、SOM 标注图、`perception_infos_stepN.json`（感知结果）、`output_for_save.json`（每步的 action/reflect/planning 原始输出）、`instruction.json`（子任务分解结果）。

评测集在 `PC-Eval.json`（也上传在 HuggingFace `StarBottle/PC-Eval`）。
