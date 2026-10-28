#!/bin/bash
# 量化: mxfp4-w4a4  (GUI-Owl-1.5-32B-Instruct)
# 用法: bash run_mxfp4-w4a4.sh    (建议: nohup bash run_mxfp4-w4a4.sh > logs/mxfp4-w4a4.out 2>&1 &)
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py mxfp4-w4a4 2>&1 | tee logs/mxfp4-w4a4.log
