export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=4,5,6,7 python verifiers/inference/vllm_server.py --model 'google/gemma-3-12b-it' \
    --served-model-name opp_and_env_model \
    --max-num-batched-tokens 16384 --max-num-seqs 256 --max-model-len 4096 \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --gpu-memory-utilization 0.95 --enable-prefix-caching \
    --host 0.0.0.0 --port 8010 