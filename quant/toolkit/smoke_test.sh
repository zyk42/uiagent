#!/bin/bash
# 对 5 个量化模型逐个做 sglang 冒烟测试: 起服务 -> 健康检查 -> 发一条请求 -> 关服务
cd /root/quant_calib
QBASE=/mnt/lisiqi23/internship/zhangyikang/uiagent/quant
RESULT=/root/quant_calib/smoke_results.txt
> $RESULT

for m in gptq-w4a16 awq-w4a16 mxfp4-w4a4 nvfp4-w4a4 smoothquant-w8a8; do
    echo "=== $m ===" | tee -a $RESULT
    python -m sglang.launch_server --model-path "$QBASE/$m" --trust-remote-code \
        --host 127.0.0.1 --port 30000 --mem-fraction-static 0.85 --disable-cuda-graph \
        > logs/smoke_$m.log 2>&1 &
    SPID=$!
    ok=0
    for i in $(seq 1 60); do
        sleep 10
        if ! kill -0 $SPID 2>/dev/null; then
            echo "SERVER_DIED" | tee -a $RESULT; break
        fi
        if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then ok=1; break; fi
    done
    if [ "$ok" = "1" ]; then
        resp=$(curl -s --max-time 120 http://127.0.0.1:30000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"default","messages":[{"role":"user","content":"hi, one word"}],"max_tokens":16,"temperature":0}')
        echo "GEN: $(echo "$resp" | head -c 300)" | tee -a $RESULT
    fi
    kill $SPID 2>/dev/null; sleep 8; kill -9 $SPID 2>/dev/null
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    sleep 5
    echo "$m done at $(date)" | tee -a $RESULT
done
echo "SMOKE ALL DONE" | tee -a $RESULT
