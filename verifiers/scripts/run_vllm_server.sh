export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=0 python verifiers/inference/vllm_server.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --served-model-name qwen_base_model \
    --tensor-parallel-size 1 --dtype bfloat16 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000 