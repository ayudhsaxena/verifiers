# Route judge rubric to vLLM OpenAI-compatible endpoint
export OPENAI_BASE_URL=http://localhost:8010/v1
export OPENAI_API_KEY=EMPTY
# Ensure judge model matches the vLLM served model name
export JUDGE_MODEL=opp_and_env_model

# LiteLLM httpx connection pool settings for high concurrency
# Default httpx has only 100 connections which bottlenecks parallel rollouts
export LITELLM_MAX_CONNECTIONS=1024
export LITELLM_MAX_KEEPALIVE=256
export LITELLM_TIMEOUT=600.0
export LITELLM_LOG_STDOUT=1

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
            --config-file configs/zero3.yaml \
            --num_processes=3 \
            verifiers/examples/sotopia_train_modified.py \
            --dataset_name=sotopia \
            --max_steps=400 \
            --run_name=punish-gibberish-lora-6-turns-mental_state_prediction-sotopia-qwen-2.5-7B-instruct-player-0 \
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=12 \
            --num_generations=12 \
            --train_player_id=0 \
            --gradient_accumulation_steps=6 \
            --max_turns=6 \
            --vllm_server_port=8000 \
            --evaluator_model=custom/opp_and_env_model@http://localhost:8010/v1 \
            --environment_model=custom/opp_and_env_model@http://localhost:8010/v1 \
            --num_iterations=1 \
            --served_model_name=inference_policy_model \
            --use_lora \
            --lora_r=16 \
            --lora_alpha=32
