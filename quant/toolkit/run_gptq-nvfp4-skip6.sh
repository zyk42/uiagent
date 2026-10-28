#!/bin/bash
set -e
cd /root/quant_calib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
python run_quant.py gptq-nvfp4-skip6 2>&1 | tee logs/gptq-nvfp4-skip6.log
