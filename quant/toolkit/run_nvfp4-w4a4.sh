#!/bin/bash
# 量化: nvfp4-w4a4  (GUI-Owl-1.5-32B-Instruct)
# 用法: bash run_nvfp4-w4a4.sh    (建议: nohup bash run_nvfp4-w4a4.sh > logs/nvfp4-w4a4.out 2>&1 &)
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py nvfp4-w4a4 2>&1 | tee logs/nvfp4-w4a4.log
