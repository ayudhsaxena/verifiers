import verifiers as vf
from verifiers.prompts import TEXTARENA_PROMPT, TEXTARENA_PROMPT_V2
import torch
import os
import argparse
from datetime import datetime
from verifiers.parsers import XMLParser


OUTPUT_DIR = "outputs"

def setup_environment():
    """Setup environment variables"""
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1" 
    os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "3600"  
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest checkpoint from the given directory.
    Checkpoints are named like 'checkpoint-50', 'checkpoint-100', etc.
    
    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        return None
    
    # Sort by the numeric part of the checkpoint name
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    
    # Return the full path to the latest checkpoint
    return os.path.join(checkpoint_dir, checkpoint_dirs[-1])

def main(args):
    """Main training function"""
    setup_environment()

    checkpoint_dir = os.path.join(OUTPUT_DIR, args.run_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args_file = os.path.join(checkpoint_dir, "args.json")
    with open(args_file, "w") as f:
        import json
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_file}")

    model, tokenizer = vf.get_model_and_tokenizer(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.system_prompt_version == "v1":
        system_prompt = TEXTARENA_PROMPT
        answer_tag = "response"
        stop_tag = "</response>"
        xml_parser = XMLParser(fields=["reasoning", answer_tag], xml_reward_weight=args.xml_reward_weight,
                               format_reward_weight=args.format_reward_weight)
    elif args.system_prompt_version == "v2":
        system_prompt = TEXTARENA_PROMPT_V2
        answer_tag = "answer"
        stop_tag = "</answer>"
        xml_parser = XMLParser(fields=["think", answer_tag], xml_reward_weight=args.xml_reward_weight,
                               format_reward_weight=args.format_reward_weight)
        

    vf_env = vf.TextArenaEnv(
        dataset=args.dataset_name,
        env_id=args.env_id,
        system_prompt=system_prompt,
        xml_parser=xml_parser,
        answer_tag=answer_tag,
        sampling_args={
            "stop": [stop_tag],
            "include_stop_str_in_output": True
        },
        train_player_id=args.train_player_id,
        eval_dataset=args.eval_dataset,
    )
    dataset = vf_env.get_dataset()
    rubric = vf_env.get_rubric()
    eval_dataset = vf_env.get_eval_dataset()

    if args.resume_training_from_last_checkpoint:
        if args.run_name is None:
            raise ValueError("Please provide a run name to resume training.")
        
        if os.path.exists(checkpoint_dir):
           
            checkpoint_dir = get_latest_checkpoint(checkpoint_dir)
            if checkpoint_dir is None:
                print(f"No checkpoints found in {args.run_name}. Starting a new training run.")
            else:
                print(f"Resuming training from the last checkpoint: {checkpoint_dir}")
                args.resume_from_checkpoint = checkpoint_dir
        else:
            print(f"No checkpoint found at {checkpoint_dir}. Starting a new training run.")
            args.resume_from_checkpoint = None
    

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = "textarena_" + args.model_name.split("/")[-1].lower() + "_" + vf_env.env_id + "_" + timestamp
    else:
        run_name = args.run_name

    print(f"Run name: {run_name}")
    print(f"Model name: {args.model_name}")
    print(f"Env ID: {args.env_id}")
    print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    
    training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8, output_dir=OUTPUT_DIR)

    # rollouts per prompt
    training_args.num_generations = args.num_generations
    # minibatch size per GPU ( bs 12 * 7 gpus / 21 rollouts -> 4 prompts per batch)
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    # batches to accumulate (4 prompts * 2 -> 8 prompts per global batch)
    training_args.gradient_accumulation_steps = 4
    # steps per global batch (1 on-policy, 1 off-policy)
    training_args.num_iterations = 2
    training_args.max_steps = args.max_steps
    training_args.beta = 0.02
    training_args.save_steps = 50
    training_args.seed = args.seed
    training_args.torch_empty_cache_steps = 100 
    
    # Evaluation
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 16
    # Set eval batch size larger than train batch size for optimal GPU utilization (e.g., double)
    training_args.per_device_eval_batch_size = 16
    
    # Configure resuming from checkpoint
    if args.resume_from_checkpoint:
        print(f"Attempting to load checkpoint from {args.resume_from_checkpoint}")
        training_args.resume_from_checkpoint = args.resume_from_checkpoint

    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rubric, 
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )

    trainer.train(resume_from_checkpoint=True if args.resume_from_checkpoint else False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for the TextArena environment")
    parser.add_argument("--env_id", type=str, default="TruthAndDeception-v0", 
                        help="Environment ID for TextArena")
    parser.add_argument("--max_steps", type=int, default=200, 
                        help="Maximum number of training steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Path to the model checkpoint to use")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run for logging purposes")
    parser.add_argument("--resume_training_from_last_checkpoint", action="store_true", default=True,
                        help="Resume training from the last checkpoint")
    parser.add_argument("--per_device_train_batch_size", type=int, default=12,
                        help="Batch size per device during training")
    parser.add_argument("--num_generations", type=int, default=21,
                        help="Number of generations per prompt")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--xml_reward_weight", type=float, default=1.0,
                        help="Weight for XML format reward")
    parser.add_argument("--format_reward_weight", type=float, default=1.0,
                        help="Weight for format reward")
    parser.add_argument("--system_prompt_version", type=str, default="v1",
                        help="Version of the system prompt to use")
    parser.add_argument("--train_player_id", type=int, default=0,
                        help="Player id to train")
    parser.add_argument("--dataset_name", type=str, default="textarena")
    parser.add_argument("--eval_dataset", type=str, default=None)
    
    args = parser.parse_args()
    main(args)