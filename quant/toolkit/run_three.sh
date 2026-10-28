#!/bin/bash
cd /root/quant_calib
for m in gptq-w4a16 nvfp4-w4a4 gptq-nvfp4; do
    echo "[orch] ==== starting $m at $(date) ===="
    python run_quant.py "$m" > "logs/$m.log" 2>&1
    if tail -c 2000 "logs/$m.log" | grep -q "DONE $m"; then
        echo "[orch] ==== $m SUCCEEDED at $(date) ===="
    else
        echo "[orch] ==== $m FAILED at $(date) ===="
    fi
done
echo "[orch] ALL DONE at $(date)"
