import random
import json
from typing import List, Dict

from datasets import Dataset, load_dataset # type: ignore

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def preprocess_dataset(dataset_name: str = "gsm8k", 
                       split: str = "train",
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       fewshot_prob: float = 1.0,
                       env_id: str = None,
                       player_id: int = 0) -> Dataset:
    if dataset_name == "gsm8k":
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["question"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_hash_answer(x["answer"])
        })
        return dataset
    elif dataset_name == "math":
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["problem"], system_prompt, few_shot, fewshot_prob),
            "answer": extract_boxed_answer(x["solution"])
        })
        return dataset
    elif dataset_name == "openbookqa":
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split] # type: ignore
        
        def format_question(example):
            choices_texts = example['choices']['text']
            choices_labels = example['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {example['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return question
        
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(format_question(x), str(system_prompt) + "\n\nReturn only the letter of the correct answer.", few_shot, fewshot_prob),
            "answer": x["answerKey"]
        })
        return dataset
    elif dataset_name == "saintlyk1d/dont-say-it-prompts-player1-basic-variant-B":
        dataset: Dataset = load_dataset("saintlyk1d/dont-say-it-prompts-player1-basic-variant-B")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
        })
        return dataset
    elif dataset_name == "saintlyk1d/truthdeception-deceiver-prompts_6_turns_test_set":
        dataset: Dataset = load_dataset("saintlyk1d/truthdeception-deceiver-prompts_6_turns_test_set")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
            "metadata" : {}
        })
        return dataset        
    elif dataset_name == "saintlyk1d/dont-say-it-prompts-player1-test-set-variant-C":
        dataset: Dataset = load_dataset("saintlyk1d/dont-say-it-prompts-player1-test-set-variant-C")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
            "metadata" :{"opponent_word": x["opponent_word"]},
            
        })
        return dataset     
    elif dataset_name == "saintlyk1d/dont-say-it-prompts-player0-test-set-variant-C":
        dataset: Dataset = load_dataset("saintlyk1d/dont-say-it-prompts-player0-test-set-variant-C")[split]
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
            "metadata" : {"opponent_word": x["opponent_word"]},
        })
        return dataset                                                      
    elif dataset_name == "textarena" :
        if env_id == "TruthAndDeception-v0":
            dataset: Dataset = load_dataset("saintlyk1d/truthdeception-deceiver-prompts")[split]
        elif env_id == "TruthAndDeception-v0-long":
            dataset: Dataset = load_dataset("saintlyk1d/truthdeception-deceiver-prompts_12_turns")[split]
        elif env_id == "DontSayIt-v0":
            if player_id == 0:
                dataset: Dataset = load_dataset("saintlyk1d/dont-say-it-prompts-player0-basic-variant-C")[split]
            else:
                dataset: Dataset = load_dataset("saintlyk1d/dont-say-it-prompts-player1-basic-variant-C")[split]
        else:
            raise ValueError(f"env_id {env_id} not supported for preprocess_dataset.")
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
            "metadata" : get_metadata(x, env_id),
        })
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for preprocess_dataset.")
    
def get_metadata(input: Dict, env_id: str) -> List[str]:
    if env_id == "DontSayIt-v0":
        return {
            "opponent_word": input["opponent_word"],
        }
    else:
        return {}