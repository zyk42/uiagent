# AgentNet (win/mac) → PC-Agent 格式蒸馏数据

从 `AgentNet/agentnet_win_mac_18k.jsonl`（OpenCUA, 17,625 条 Windows/macOS 轨迹）
转换而来，用于蒸馏 **PC-Agent 风格的回答格式**（目标模型：GUI-Owl-1.5-8B-Instruct）。

## 文件

| 文件 | 说明 |
|---|---|
| `agentnet_win_mac_pcagent.jsonl` | **350,013 个训练样本**（每行一个原子动作，~13G） |
| `convert_to_pcagent.py` | 转换脚本（可复用，`--input` 换成 ubuntu 的 jsonl 即可转换 Ubuntu 子集） |
| `stats.json` | 动作统计 |

## 样本格式（每行一个 JSON）

```jsonc
{
  "sample_id": "{task_id}_s{step}_{sub}",
  "task_id": "...", "system": "Windows", "domain": "Office Tools",
  "instruction": "...",                // 用于构造 prompt 的指令
  "natural_language_task": "...",      // 口语化指令（备用）
  "step_index": 0, "sub_index": 0,     // sub>0 表示原 step 的多行代码被拆成多个原子动作
  "is_continuation": false,
  "image": "win_mac_images/<uuid>.png", // 逻辑文件名（散文件不存在，请用 image_tar 读取）
  "image_tar": ["shard_0007.tar", 123456, 375231],  // [shard, offset, size]，直接随机读取
  "image_width": 1280, "image_height": 720,  // 图片真实像素尺寸（注意：98.5% 的任务图片
                                             // 被 AgentNet 缩放过，与 meta 屏幕分辨率不同！
                                             // prompt 和 action_pixel 均按真实尺寸生成）
  "coord_format": "normalized_0_999",
  "prompt":   "...",   // PC-Agent get_action_prompt() 原文（无感知信息模式）
  "response": "```json\n{\"Thought\": ..., \"Action\": ..., \"Summary\": ...}\n```",
  "thought": "...", "action": "Tap (163, 271)", "summary": "...",
  "action_pixel": "Tap (235, 244)",    // 像素坐标版本（若部署时用 perception 模式）
  "observation": "...",                // 原始观察文本（未放进 response，可自行拼接）
  "raw_code": "pyautogui.click(x=0.1632, y=0.2711)",
  "reflection_of_this_step": "...",    // 对本步动作的事后评判
  "this_step_correct": true,           // 本步动作是否正确（可用于过滤/构造正负样本）
  "this_step_redundant": false,
  "task_completed": false,             // 整条轨迹最终是否成功
  "terminated_with_fail": false,
  "alignment_score": 7, "efficiency_score": 6, "task_difficulty": 3,
  "messages": [ ... ]                  // OpenAI 多模态 chat 格式（system/user+image/assistant），可直接用于 SFT
}
```

## Prompt 构造方式

直接用 PC-Agent 源码里的 `get_action_prompt()`（`/root/MobileAgent-main/PC-Agent/PCAgent/prompt_qwen.py`）生成，参数约定：

- `clickable_infos=[]`（AgentNet 无 OCR/a11y 标注）→ 走 PC-Agent 的**无感知信息分支**，
  prompt 中明确要求输出 **[0,999] 归一化坐标**，response 中的坐标也是 [0,999]。
  这与 PC-Agent `run.py` 执行时的反归一化（`x * width / 1000`）完全兼容。
- `### History operations ###`：由前序原子动作累加（含 Reflection，与 PC-Agent 行为一致）。
- 上一步被标注为错误（`last_step_correct=false`）时，注入 PC-Agent 的
  `### Last operation ###` 纠错段落（`error_flag` 分支）。
- `add_info` / `Progress` / `Memory` 为空（AgentNet 无对应标注）。
- system prompt 与 PC-Agent 决策模块一致：
  "You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction."

## 动作映射表（AgentNet code → PC-Agent Action）

| AgentNet (pyautogui) | PC-Agent Action | 备注 |
|---|---|---|
| `click(x, y)` | `Tap (x, y)` | 235,648 |
| `doubleClick` / `clicks=2` | `Double Tap (x, y)` | 8,168 |
| `rightClick` / `button='right'` | `Right Tap (x, y)` | 4,273 |
| `tripleClick` / `clicks=3` | `Triple Tap (x, y)` | 105 |
| `write(message=...)` | `Type (x, y), ("text")` | 30,943；坐标取最近一次 click/moveTo 位置，无则退化为 `Type ("text")`；同行后续 `press('enter')` 被合并 |
| `press(key)` | `Press (key)` | `presses=N>1` 时输出 `Press (key) [xN]`（扩展语法） |
| `hotkey([k1, k2])` | `Shortcut (k1, k2)` | `cmd`→`command`；单键→`Press`；≥3 键→`Shortcut (k1, k2, k3)`（扩展） |
| `moveTo` + `dragTo` | `Drag (x1, y1), (x2, y2)` | **扩展动作**，PC-Agent prompt 未定义（其内部有 `drag()` 函数） |
| `scroll(n)` | `Scroll (x, y), (n)` | **扩展动作**；正=向上 |
| `hscroll(n)` | `HScroll (x, y), (n)` | **扩展动作** |
| `middleClick` | `Middle Tap (x, y)` | **扩展动作**，仅 42 例 |
| `computer.terminate(status='success')` | `Stop` | 17,478 例 |
| `computer.terminate(status='fail')` | `Stop` + `terminated_with_fail=true` | 86 例，建议过滤 |
| `moveTo`（单独出现） | 丢弃（悬停无 PC-Agent 对应动作） | 计入 stats 的 moveTo |

## 已做的清洗

1. thought/summary 中的 ` ``` ` 代码围栏替换为 `'''`，避免破坏 PC-Agent 的 ` ```json ` 提取逻辑（已验证 5000 随机样本 100% 可解析）。
2. 空文本 `write(message='')` 丢弃（175 例）。
3. 多行代码拆分为多个原子动作样本（`sub_index` 递增）：第一个带原始 thought，
   后续动作用模板化 thought（"Following the plan, ..."）。
4. 坐标全部校验在 [0,999] 范围内。

## 使用建议

- **质量过滤**：`task_completed=true` 保留成功轨迹；`this_step_correct=false`（约占比可在
  统计中查）的步可作为反思类负样本，或过滤掉只留正确动作。
- **像素坐标**：若蒸馏后部署时开 PC-Agent 的 perception 模式（默认），训练时可把
  response 里的坐标换成 `action_pixel` 版本。
- **截图已打包为 tar 分片**：`win_mac_tar/shard_0000~0031.tar`（共 349,169 张，126GB，0 失败）+
  `win_mac_tar/tar_index.json`（`文件名 → [shard, offset, size]`，随机读取）。
  之所以不用散文件：JuiceFS 创建小文件 ~1s/个，35 万个 PNG 要 1 小时+；tar 分片是顺序读写，2 分钟完成。
  ```python
  import json
  idx = json.load(open('win_mac_tar/tar_index.json'))
  shard, off, size = idx['<uuid>.png']
  with open(f'win_mac_tar/{shard}', 'rb') as f:
      f.seek(off); png_bytes = f.read(size)
  ```
  原始分卷 zip（`AgentNet/win_mac_images/images.z01~z23 + images.zip`）中央目录的偏移量是坏的
  （官方 `zip -s 0` 合并命令也未验证可用）；扫描重建的索引在 `AgentNet/win_mac_images_index.json`。
- **Ubuntu 子集**：`python3 convert_to_pcagent.py --input agentnet_ubuntu_5k.jsonl --image_prefix ubuntu_images`

## 重新生成

```bash
python3 convert_to_pcagent.py \
  --agentnet_dir /mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet \
  --out_dir /mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill
```
