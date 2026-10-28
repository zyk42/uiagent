#!/bin/bash
# DFlash2 GUI-Owl-32B demo: single shard (part-00) training launcher.
# 4x capture servers (GPU 0-3) + DP4 trainer (GPU 4-7), managed-local mode.
# part-00 的 tokenize cache 已生成,启动时直接命中。
set -e

D=/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill/dflash2

# 让 capture server 保留客户端发来的 input_ids(图片 token 已用 target 自带的
# AutoProcessor 展开),避免服务端 decode/retokenize 造成 token 数漂移。
export SGLANG_MM_AVOID_RETOKENIZE=1

mkdir -p "$D/demo-train-part00"
exec specforge train --config "$D/train-demo-part00.yaml" 2>&1 | tee -a "$D/demo-train-part00/train.log"
