#!/bin/bash
# 依次运行剩余量化方法; 单个失败不阻断后续
# GPTQ 已在独立进程运行, 本脚本先等它结束
cd /root/quant_calib

echo "[orch] waiting for running gptq-w4a16 (if any)..."
while pgrep -f "run_quant.py gptq-w4a16" > /dev/null; do sleep 60; done
echo "[orch] gptq-w4a16 process exited at $(date)"

for m in awq-w4a16 nvfp4-w4a4 mxfp4-w4a4 smoothquant-w8a8; do
    echo "[orch] ==== starting $m at $(date) ===="
    python run_quant.py "$m" > "logs/$m.log" 2>&1
    rc=$?
    echo "[orch] ==== $m exited rc=$rc at $(date) ===="
done
echo "[orch] ALL DONE at $(date)"
