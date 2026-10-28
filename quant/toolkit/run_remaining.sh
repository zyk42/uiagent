#!/bin/bash
# 等当前 AWQ 跑完后, 依次运行剩余量化方法; 单个失败不阻断后续
cd /root/quant_calib

echo "[orch] waiting for running awq-w4a16 ..."
while pgrep -f "run_quant.py awq-w4a16" > /dev/null; do sleep 60; done
echo "[orch] awq-w4a16 process exited at $(date)"
if tail -c 2000 logs/awq-w4a16.log | grep -q "DONE awq-w4a16"; then
    echo "[orch] awq-w4a16 SUCCEEDED"
else
    echo "[orch] awq-w4a16 FAILED (no DONE marker), see logs/awq-w4a16.log"
fi

for m in nvfp4-w4a4 mxfp4-w4a4 smoothquant-w8a8; do
    echo "[orch] ==== starting $m at $(date) ===="
    python run_quant.py "$m" > "logs/$m.log" 2>&1
    rc=$?
    if tail -c 2000 "logs/$m.log" | grep -q "DONE $m"; then
        echo "[orch] ==== $m SUCCEEDED at $(date) ===="
    else
        echo "[orch] ==== $m FAILED rc=$rc at $(date) ===="
    fi
done
echo "[orch] ALL DONE at $(date)"
