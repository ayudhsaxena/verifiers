accelerate launch \
            --config-file configs/zero3.yaml \
            --num-processes 7 \
            verifiers/examples/textarena_truth_n_deception.py \
            --env_id=DontSayIt-v0 \
            --dataset_name=textarena \
            --max_steps=350 \
            --run_name=dontsayit-player-0-qwen-2.5-7B-instruct-advantage-unnormalized \
            --seed=11 \
            --resume_training_from_last_checkpoint \
            --per_device_train_batch_size=4 \
            --num_generations=14 \
            --train_player_id=0 \
            --eval_dataset=saintlyk1d/dont-say-it-prompts-player0-test-set-variant-C
# accelerate launch \
#             --config-file configs/zero3.yaml \
#             --num-processes 7 \
#             verifiers/examples/textarena_truth_n_deception.py \
#             --env_id=TruthAndDeception-v0 \
#             --dataset_name=textarena \
#             --max_steps=350 \
#             --run_name=truth-and-deception-qwen-2.5-7B-instruct-deceiver-advantage-unnormalized_v2 \
#             --seed=0 \
#             --resume_training_from_last_checkpoint \
#             --per_device_train_batch_size=8 \
#             --num_generations=14 \
#             --train_player_id=0 \
#             --eval_dataset=saintlyk1d/truthdeception-deceiver-prompts_6_turns_test_set