#!/bin/bash
# 依次部署 4 个模型 (3 量化 + BF16 参照) 并测速
cd /root/quant_calib
QBASE=/mnt/lisiqi23/internship/zhangyikang/uiagent
RESULT=bench_results.txt
> $RESULT

run_one() {
    local name=$1 path=$2
    echo "=== $name ===" | tee -a $RESULT
    python -m sglang.launch_server --model-path "$path" --trust-remote-code \
        --host 127.0.0.1 --port 30000 --mem-fraction-static 0.85 --disable-radix-cache \
        > logs/bench_$name.log 2>&1 &
    local SPID=$!
    local ok=0
    for i in $(seq 1 90); do
        sleep 10
        if ! kill -0 $SPID 2>/dev/null; then echo "SERVER_DIED" | tee -a $RESULT; break; fi
        if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then ok=1; break; fi
    done
    if [ "$ok" = "1" ]; then
        python3 bench_one.py "$name" 2>&1 | tee -a $RESULT
    fi
    kill $SPID 2>/dev/null; sleep 8; kill -9 $SPID 2>/dev/null
    sleep 5
}

run_one gptq-w4a16        "$QBASE/quant/gptq-w4a16"
run_one awq-w4a16         "$QBASE/quant/awq-w4a16"
run_one smoothquant-w8a8  "$QBASE/quant/smoothquant-w8a8"
run_one bf16-baseline     "$QBASE/GUI-Owl-1.5-32B-Instruct"
echo "BENCH ALL DONE" | tee -a $RESULT
