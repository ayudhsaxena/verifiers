import verifiers as vf
from verifiers.prompts.system_prompts import MODIFIED_SOTOPIA_PROMPT
import os
import argparse
from datetime import datetime
from verifiers.parsers.xml_parser import XMLParser
from verifiers.utils.data_utils import load_example_dataset


OUTPUT_DIR = "outputs"

def setup_environment():
    """Setup environment variables"""
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1" 
    os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "3600"  
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
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

    os.makedirs(checkpoint_dir, exist_ok=True)
    args_file = os.path.join(checkpoint_dir, "args.json")
    with open(args_file, "w") as f:
        import json
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_file}")

    model, tokenizer = vf.get_model_and_tokenizer(args.model_name) #type: ignore
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    system_prompt = MODIFIED_SOTOPIA_PROMPT
    think_tag = "think"
    answer_tag = "response"
    prediction_tag = "prediction"
    stop_tag = "</response>"
    xml_parser = XMLParser(fields=[prediction_tag, think_tag, answer_tag], answer_field=answer_tag)
        
    dataset = load_example_dataset(name=args.dataset_name,n=2000, split='train')
    eval_dataset = load_example_dataset(name=args.dataset_name,n=50, split='test')
    
    vf_env = vf.ModifiedSotopiaEnv( #type: ignore
        system_prompt=system_prompt,
        parser=xml_parser,
        answer_tag=answer_tag,
        think_tag=think_tag,
        prediction_tag=prediction_tag,
        sampling_args={
            "stop": [stop_tag],
        },
        train_player_id=args.train_player_id,
        max_turns=20,
        evaluator_model=args.evaluator_model,
        dataset=dataset,
        eval_dataset=eval_dataset,
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
        run_name = "sotopia_modified_" + args.model_name.split("/")[-1].lower() + "_" + timestamp
    else:
        run_name = args.run_name

    print(f"Run name: {run_name}")
    print(f"Model name: {args.model_name}")
    print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    
    training_args=vf.grpo_defaults(run_name=run_name, output_dir=OUTPUT_DIR) #type: ignore

    training_args.num_generations = args.num_generations
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.max_steps = args.max_steps
    training_args.save_steps = 500
    training_args.seed = args.seed
    training_args.torch_empty_cache_steps = 50
    training_args.max_completion_length = 4096
    # training_args.loss_type = "dr_grpo"
    # training_args.scale_rewards = False
    
    # Evaluation
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    # Set eval batch size larger than train batch size for optimal GPU utilization (e.g., double)
    training_args.per_device_eval_batch_size = 16
    
    # Configure resuming from checkpoint
    if args.resume_from_checkpoint:
        print(f"Attempting to load checkpoint from {args.resume_from_checkpoint}")
        training_args.resume_from_checkpoint = args.resume_from_checkpoint

    trainer = vf.GRPOTrainer( #type: ignore
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=True if args.resume_from_checkpoint else False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for the Modified Sotopia environment")
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
    parser.add_argument("--train_player_id", type=int, default=0,
                        help="Player id to train")
    parser.add_argument("--dataset_name", type=str, default="sotopia")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4o-mini",
                        help="Model to use for LLM-based evaluation (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    main(args) 