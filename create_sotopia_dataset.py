import os
import sys
from datasets import Dataset, Features, Value
from huggingface_hub import whoami

# Add the sotopia path to sys.path to import EnvironmentProfile
sys.path.append('/data/user_data/ayudhs/textarena/sotopia')

# Set up Redis connection (if needed)
os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"

# Import EnvironmentProfile
from sotopia.database.persistent_profile import EnvironmentProfile

def create_sotopia_environment_pk_dataset():
    """
    Create a Hugging Face dataset containing Sotopia environment profile "pk" values.
    """
    print("Fetching all environment profiles from Sotopia database...")
    
    try:
        # Get all environment profiles
        all_env_profiles = EnvironmentProfile.all()
        print(f"Found {len(all_env_profiles)} environment profiles")
        
        # Extract pk values and create dataset
        pk_data = []
        for profile in all_env_profiles:
            if profile.pk:  # Only include profiles with valid pk
                pk_data.append({
                    "pk": profile.pk,
                    "codename": profile.codename,
                    "source": profile.source,
                    "scenario": profile.scenario,
                    "relationship": str(profile.relationship),
                    "tag": profile.tag
                })
        
        print(f"Created dataset with {len(pk_data)} valid environment profiles")
        
        # Create train/test split
        import random
        random.seed(42)  # For reproducibility
        
        # Randomly select 50 environments for test split
        test_indices = random.sample(range(len(pk_data)), min(50, len(pk_data)))
        train_data = [pk_data[i] for i in range(len(pk_data)) if i not in test_indices]
        test_data = [pk_data[i] for i in test_indices]
        
        print(f"Created train split with {len(train_data)} environments")
        print(f"Created test split with {len(test_data)} environments")
        
        # Create Hugging Face dataset features
        features = Features({
            "pk": Value("string"),
            "codename": Value("string"),
            "source": Value("string"),
            "scenario": Value("string"),
            "relationship": Value("string"),
            "tag": Value("string")
        })
        
        # Create datasets for both splits
        train_dataset = Dataset.from_list(train_data, features=features)
        test_dataset = Dataset.from_list(test_data, features=features)
        
        # Create a dataset dict with both splits
        from datasets import DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        # Get Hugging Face username
        try:
            user_info = whoami()
            username = user_info["name"]
            print(f"Using Hugging Face username: {username}")
        except Exception as e:
            print(f"Error getting Hugging Face username: {e}")
            print("Using provided username: saintlyk1d")
            username = "saintlyk1d"
        
        # Create dataset name and repo ID
        dataset_name = "sotopia-environment-profile-pks"
        repo_id = f"{username}/{dataset_name}"
        
        # Create dataset card
        readme_content = f"""
# Sotopia Environment Profile PKs Dataset

This dataset contains the primary keys (pk) and metadata for all environment profiles in the Sotopia database.

## Dataset Description

- **Total environment profiles:** {len(pk_data)}
- **Train split:** {len(train_data)} environments
- **Test split:** {len(test_data)} environments (randomly selected)
- **Features:**
  - `pk`: Primary key (unique identifier) for each environment profile
  - `codename`: Codename of the environment
  - `source`: Source of the environment
  - `scenario`: Description of the social interaction scenario
  - `relationship`: Type of relationship between agents
  - `tag`: Tag for categorizing environments

## Usage

This dataset is intended for researchers and developers working with the Sotopia social simulation platform. The primary keys can be used to retrieve full environment profiles from the Sotopia database.

## Dataset Statistics

- **Total profiles:** {len(pk_data)}
- **Train profiles:** {len(train_data)}
- **Test profiles:** {len(test_data)}
- **Unique sources:** {len(set(item['source'] for item in pk_data))}
- **Unique relationship types:** {len(set(item['relationship'] for item in pk_data))}

## Example Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("saintlyk1d/sotopia-environment-profile-pks")

# Access train split
train_data = dataset['train']
print(f"Train split has {len(train_data)} examples")

# Access test split
test_data = dataset['test']
print(f"Test split has {len(test_data)} examples")

# Access the first few examples from train
print(train_data[:3])
```
"""
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Push to Hugging Face Hub
        print(f"Pushing dataset to Hugging Face Hub as '{repo_id}'...")
        dataset.push_to_hub(repo_id)
        
        # Also push the README
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id
        )
        
        print(f"Successfully pushed dataset to Hugging Face Hub")
        print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
        
        # Print sample data
        print("\nSample data from the train split:")
        for i, example in enumerate(dataset["train"][:3]):
            print(f"\nExample {i+1}:")
            print(f"PK: {example['pk']}")
            print(f"Codename: {example['codename']}")
            print(f"Source: {example['source']}")
            print(f"Relationship: {example['relationship']}")
            print(f"Tag: {example['tag']}")
            print(f"Scenario: {example['scenario'][:100]}...")
        
        print("\nSample data from the test split:")
        for i, example in enumerate(dataset["test"][:3]):
            print(f"\nExample {i+1}:")
            print(f"PK: {example['pk']}")
            print(f"Codename: {example['codename']}")
            print(f"Source: {example['source']}")
            print(f"Relationship: {example['relationship']}")
            print(f"Tag: {example['tag']}")
            print(f"Scenario: {example['scenario'][:100]}...")
        
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the function to create the dataset
    sotopia_dataset = create_sotopia_environment_pk_dataset() 