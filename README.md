# uiagent

GUI Agent 工作区:PC-Agent 数据蒸馏管道、DFlash2 投机解码草稿模型训练、量化与评估压测。

> **数据与模型不在本仓库**(体积原因),位置与获取方式见 [数据与模型说明.md](数据与模型说明.md)。

## 目录

| 目录 | 内容 |
|---|---|
| `pc_agent_distill/` | AgentNet → PC-Agent 格式转换 + GUI-Owl-32B 两遍蒸馏 + DFlash2 训练数据管道(脚本/文档/配置) |
| `pc_agent_distill/dflash2/` | DFlash2 draft 训练全链路脚本(分片链、precache、评测、自动化管道) |
| `MobileAgent-main/` | MobileAgent/PC-Agent 源码(参考实现,含 PC-Agent 机制解析) |
| `tools/` | tar 分片打包/随机读、分卷 zip 修复扫描等数据工具 |
| `quant/` | GUI-Owl 量化工具包与说明(AWQ/FP8/GPTQ-NVFP4 等) |
| `eval/` | 量化模型一致性评估(方案/细则/结果/脚本) |
| `press/` | 部署与压测(方案/结果/脚本) |
| `logs/` | 蒸馏/服务日志 |

## 相关资源(私有)

- 模型仓库(ModelScope):`vivalavida21k/dflash2-draft-guiowl-1.5-32b` — DFlash2 draft 权重
- 数据仓库(ModelScope):`vivalavida21k/pc-agent-guiowl32b-distill` — 两遍蒸馏全量数据 + 截图 tar
- DFlash2 训练复现包(内网):`/mnt/lisiqi23/dflash2_env/`
