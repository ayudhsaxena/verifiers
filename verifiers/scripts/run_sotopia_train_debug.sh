# Route judge rubric to vLLM OpenAI-compatible endpoint
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
# Ensure judge model matches the vLLM served model name
export JUDGE_MODEL=inference_policy_model

# Enable debugpy for multi-process debugging (will only attach to rank 0)
export DEBUGPY_ENABLE=1
export DEBUGPY_PORT=5678

# Change to the verifiers directory (parent of this script)
cd "$(dirname "$0")/../.." || exit 1

CUDA_VISIBLE_DEVICES=0 accelerate launch \
            --config-file configs/zero3.yaml \
            --num_processes=1 \
            verifiers/examples/sotopia_train_modified.py \
            --dataset_name=sotopia \
            --max_steps=400 \
            --run_name=mental_state_prediction-sotopia-qwen-2.5-7B-instruct-player-0-debug \
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=12 \
            --num_generations=12 \
            --train_player_id=0 \
            --gradient_accumulation_steps=1 \
            --max_turns=10 \
            --vllm_server_port=8000 \
            --evaluator_model=custom/inference_policy_model@http://localhost:8000/v1 \
            --environment_model=custom/inference_policy_model@http://localhost:8000/v1 \
            --num_iterations=1 \
            --served_model_name=inference_policy_model