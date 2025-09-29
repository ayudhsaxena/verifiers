CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
            --config-file configs/zero3.yaml \
            --num_processes=7 \
            verifiers/examples/sotopia_train_modified.py \
            --dataset_name=sotopia \
            --max_steps=3000 \
            --run_name=longer-v2_10_turns_modified-sotopia-qwen-2.5-7B-instruct-player-0 \
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=6 \
            --num_generations=14 \
            --train_player_id=0 \
            --gradient_accumulation_steps=4 \
            --max_turns=10 \
            --vllm_server_port=8000 \
            --num_iterations=2 \