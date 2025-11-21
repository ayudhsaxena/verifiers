# Route judge rubric to vLLM OpenAI-compatible endpoint
export OPENAI_BASE_URL=http://localhost:8010/v1
export OPENAI_API_KEY=EMPTY
# Ensure judge model matches the vLLM served model name
export JUDGE_MODEL=opp_and_env_model

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
            --config-file configs/zero3.yaml \
            --num_processes=6 \
            verifiers/examples/sotopia_train_modified.py \
            --dataset_name=sotopia \
            --max_steps=400 \
            --run_name=modified_advantages-6-turns-mental_state_prediction-sotopia-qwen-2.5-7B-instruct-player-0 \
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=8 \
            --num_generations=12 \
            --train_player_id=0 \
            --gradient_accumulation_steps=4 \
            --max_turns=6 \
            --vllm_server_port=8000 \
            --evaluator_model=custom/opp_and_env_model@http://localhost:8010/v1 \
            --environment_model=custom/opp_and_env_model@http://localhost:8010/v1 \
            --num_iterations=1 \
            --served_model_name=inference_policy_model \