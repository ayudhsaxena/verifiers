export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=7 python verifiers/inference/vllm_server.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 1 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000