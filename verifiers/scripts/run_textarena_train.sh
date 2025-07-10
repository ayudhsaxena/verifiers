CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
            --config-file configs/zero3.yaml \
            --num_processes=7 \
            verifiers/examples/textarena_truth_n_deception_modified.py \
            --env_id=DontSayIt-v0 \
            --dataset_name=textarena \
            --max_steps=400 \
            --run_name=modified_dontsayit-qwen-2.5-7B-instruct-player-1_dr_grpo\
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=8 \
            --num_generations=14 \
            --train_player_id=1 \
            --gradient_accumulation_steps=4 \
