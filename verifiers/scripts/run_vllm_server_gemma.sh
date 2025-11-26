export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=3 python verifiers/inference/vllm_server.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --served-model-name inference_policy_model \
    --max-num-batched-tokens 16384 --max-num-seqs 256 --max-model-len 4096 \
    --tensor-parallel-size 1 --dtype bfloat16 \
    --gpu-memory-utilization 0.95 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000 