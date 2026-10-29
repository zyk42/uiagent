# eval_pc_360_v2 评估集(自稳定 + GT 一致版)

> 生成日期: 2026-10-29 · 替代/补充: `eval_pc_360.jsonl`(v1)
> 设计依据: `评估集设计指南.md` · 打分细则: `量化一致性评估细则.md`

## 文件

| 文件 | 说明 |
|---|---|
| `eval_pc_360_v2.jsonl` | 终版评估集,360 条(win/mac/ubuntu 各 120) |
| `images_v2/` | 360 张截图(jsonl 中 `image_path` 指向,自包含) |

## 与 v1 的差异(为什么重建)

v1 的实证问题: 34% 样本 BF16 同栈三次运行不一致(多解/歧义),参照与人工 GT 匹配率仅 7.5%,L0 被噪声淹没。v2 的构建漏斗:

1. **候选池 720**: 从 `pc_agent_distill` 的 43 万样本中,选 rollout 输出与人工 GT **执行等价**(细则 L1: 坐标类 ε=10,其余规范化精确)的样本,win/mac/ubuntu 各 240,每 task 最多 2 条
2. **四轮推理过滤**: 已有 H20 蒸馏(pass1) + CloudML vLLM BF16 新跑 3 轮(pass2/3/4),**仅保留 4 轮规范化后 action 完全一致**的样本 → 稳定池 420/720(58%)
3. **终选 360**: 稳定池中按平台各 120;不分层抽样,稀有动作(Scroll/Drag/Right Tap/Press/Shortcut/Double Tap)保底全收

## 质量指标(实测)

| 指标 | v2 | v1(对照) |
|---|---|---|
| BF16 自重跑 L0(噪声地板) | **95.8%** | 57~59% |
| BF16 自重跑 L1 / L2 | 98.6% / 98.9% | 79.7 / 89.9 |
| 可解析率 | 99.4% | 98.9% |
| 参照与人工 GT 一致率 | **39.2%**(字符串级;L1 口径更高) | 7.5% |

动作分布: Tap 185 · Stop 89 · Type 37 · Shortcut 20 · Press 12 · Double Tap 10 · Scroll 3 · Drag 2 · Right Tap 2
(全类型覆盖;Tap 占比高是真实分布,但所有样本已经过了唯一解过滤)

## 使用注意

1. **参照答案 = pass2(vLLM CloudML 栈)输出**,且该输出同时满足: 4 轮跨栈一致 + 与 GT 执行等价
2. v2 的噪声地板(L0≈96%)意味着: 量化模型 L0 低于 ~90% 即可判定为真实漂移,无需再大幅折算
3. 剩余 4% 噪声来自并发组批的不可消除抖动;要再压可以用并发=1 跑参照
4. 稀有动作(Scroll/Drag/Right Tap 各 2~3 条)只做覆盖性检查,不做分层统计推断
5. 评估时 `EVAL_FILE` 环境变量指向本文件即可复用 `/root/eval_consistency.py`:
   `EVAL_FILE=.../eval_pc_360_v2.jsonl python3 /root/eval_consistency.py <url> 8`

## 复现材料(dev 机 /root 下)

- 候选池: `eval_v2_candidates.jsonl`(720 条, 含 tar 图片引用)
- 四轮 rollout: `eval_v2_rollouts/pass{2,3,4}.jsonl` + 候选池内 pass1 字段
- 稳定池/终选 ID: `eval_v2_stable.json` / `eval_v2_final_ids.json`
- rollout 脚本: `rollout_candidates.py`(注意: 图片 tar 分片两个目录同名, 必须按 platform 选目录)
