#!/bin/bash
# GUI-Owl-1.5-32B 部署脚本: gptq-w4a16 + DFlash2 投机解码 (vLLM 0.32)
# 实测: decode 53.9 -> 225.4 tok/s (4.2x), 平均接受长度 4.6/16
# 用法: bash deploy_gptq_w4a16_dflash.sh   (建议: setsid nohup bash ... > /dev/null 2>&1 < /dev/null &)
# 停止: pkill -f "vllm serve"

TARGET_MODEL=${TARGET_MODEL:-/mnt/lisiqi23/internship/zhangyikang/uiagent/quant/gptq-w4a16}
DRAFT_MODEL=${DRAFT_MODEL:-/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill/dflash2/draft-part07-final-hf}
PORT=${PORT:-30000}
MAX_LEN=${MAX_LEN:-12288}

vllm serve "$TARGET_MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":16}"
