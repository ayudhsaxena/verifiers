#!/usr/bin/env python3
"""
Script to process EpisodeLog messages and save them in a readable format.

This script takes a list of environment IDs, fetches episodes for each environment,
creates ParallelSotopiaEnv instances to extract scenario information using 
env.background.to_natural_language(), and saves the formatted messages to text 
files in a structured directory layout.
"""

import os
import re
import json
import argparse
from datetime import datetime
from typing import List, Optional
from sotopia.database.logs import EpisodeLog
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.agents.llm_agent import LLMAgent, Agents
from sotopia.database.persistent_profile import AgentProfile
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator


def extract_scenario_from_episode_log(episode_log: EpisodeLog) -> str:
    """Extract scenario information using ParallelSotopiaEnv and env.background.to_natural_language()."""
    try:
        # Create ParallelSotopiaEnv with the environment ID from the episode log
        env = ParallelSotopiaEnv(
            uuid_str=episode_log.environment,
            action_order="round-robin",
            evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=20)],
        )
        
        # Create dummy agents for the environment setup
        agent_list = []
        for agent_id in episode_log.agents:
            try:
                agent_profile = AgentProfile.get(pk=agent_id)
                agent = LLMAgent(
                    agent_profile=agent_profile,
                    model_name="dummy",  # We don't need a real model for scenario extraction
                )
                agent_list.append(agent)
            except Exception:
                # If agent profile not found, create a dummy one
                agent = LLMAgent(
                    agent_name=f"Agent_{agent_id}",
                    model_name="dummy",
                )
                agent_list.append(agent)
        
        # Create Agents object
        agents = Agents({agent.agent_name: agent for agent in agent_list})
        
        # Reset the environment to initialize the background
        env.reset(agents=agents, omniscient=False)
        
        # Get the scenario using background.to_natural_language()
        scenario = env.background.to_natural_language()
        return scenario
        
    except Exception as e:
        # Fallback to original method if ParallelSotopiaEnv approach fails
        print(f"Warning: Could not create ParallelSotopiaEnv for episode {episode_log.environment}: {e}")
        return f"Environment ID: {episode_log.environment}\nAgents: {', '.join(episode_log.agents)}"


def extract_turn_messages(messages: List[List[tuple]]) -> List[str]:
    """Extract formatted turn messages from the episode messages."""
    turn_messages_map = {}
    turn_order = []
    
    # Skip the first turn (contains scenario setup)
    for turn_idx, turn_messages in enumerate(messages[1:], 1):
        for sender, receiver, message in turn_messages:
            # Look for environment messages that contain "Turn #X:" format
            if sender == "Environment" and receiver != "Environment":
                if "Turn #" in message:
                    # Extract just the turn message part
                    lines = message.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Turn #"):
                            turn_match = re.match(r"(Turn #\d+):", line)
                            if not turn_match:
                                continue

                            turn_label = turn_match.group(1)
                            existing_line = turn_messages_map.get(turn_label)

                            if existing_line is None:
                                turn_messages_map[turn_label] = line
                                turn_order.append(turn_label)
                            else:
                                existing_is_placeholder = "You said" in existing_line
                                candidate_is_placeholder = "You said" in line

                                if existing_is_placeholder and not candidate_is_placeholder:
                                    turn_messages_map[turn_label] = line
                                elif candidate_is_placeholder and not existing_is_placeholder:
                                    continue
                                elif line == existing_line:
                                    continue
                                else:
                                    continue

                            break  # Only take the first Turn # line from each message
    
    return [turn_messages_map[label] for label in turn_order]


def extract_raw_messages_interleaved(episode_log: EpisodeLog) -> str:
    """Extract and interleave raw messages with regular messages."""
    if not episode_log.raw_messages:
        return "No raw messages available"
    
    raw_messages = episode_log.raw_messages
    return "\n".join(raw_messages)
    
    


# def extract_raw_messages_interleaved(episode_log: EpisodeLog) -> str:
#     """Extract and interleave raw messages with regular messages."""
#     if not episode_log.raw_messages:
#         return "No raw messages available"
    
#     raw_messages = episode_log.raw_messages
#     messages = episode_log.messages
    
#     # Parse raw messages to extract agent name and content
#     raw_agent_name = None
#     parsed_raw_messages = []
    
#     for turn_idx, raw_msg in enumerate(raw_messages):
#         # Extract agent name from the beginning of the message
#         if ":" in raw_msg:
#             agent_name = raw_msg.split(":", 1)[0].strip()
#             if raw_agent_name is None:
#                 raw_agent_name = agent_name
#             content = raw_msg.split(":", 1)[1].strip()
#             parsed_raw_messages.append((turn_idx, content))
    
#     if not raw_agent_name:
#         return "Could not identify agent from raw messages"
    
#     # Extract other agent's name
#     other_agent_name = None
#     for agent in episode_log.agents:
#         # Get agent profile to find the actual name
#         try:
#             agent_profile = AgentProfile.get(pk=agent)
#             agent_full_name = f"{agent_profile.first_name} {agent_profile.last_name}"
#             if agent_full_name != raw_agent_name:
#                 other_agent_name = agent_full_name
#                 break
#         except:
#             # If can't get profile, use agent ID
#             if agent != raw_agent_name:
#                 other_agent_name = agent
#                 break
    
#     if not other_agent_name:
#         return "Could not identify the other agent"
    
#     # Extract other agent's messages from the messages field with turn numbers
#     other_agent_messages = []
    
#     # Get all turn messages first
#     all_turn_messages = extract_turn_messages(messages)
    
#     # Filter for the other agent's messages and extract turn numbers
#     for turn_message in all_turn_messages:
#         if other_agent_name in turn_message and raw_agent_name not in turn_message:
#             # Extract turn number from the message
#             import re
#             turn_match = re.match(r'Turn #(\d+):', turn_message)
#             if turn_match:
#                 turn_num = int(turn_match.group(1))
#                 other_agent_messages.append((turn_num, turn_message))
    
#     # Create chronological sequence by properly alternating messages
#     interleaved = []
#     current_turn = 0
    
#     # Determine the maximum number of turns
#     max_turns = max(len(parsed_raw_messages), len(other_agent_messages))
    
#     # Alternate between agents for each turn
#     for turn in range(max_turns):
#         # Add raw agent's message if available for this turn
#         if turn < len(parsed_raw_messages):
#             turn_idx, content = parsed_raw_messages[turn]
#             interleaved.append(f"Turn #{current_turn}: {raw_agent_name} (Raw):")
#             interleaved.append(content)
#             interleaved.append("")  # Empty line for readability
#             current_turn += 1
        
#         # Add other agent's message if available for this turn
#         if turn < len(other_agent_messages):
#             turn_num, original_message = other_agent_messages[turn]
#             # Replace the original turn number with the current sequential turn number
#             import re
#             updated_message = re.sub(r'Turn #\d+:', f'Turn #{current_turn}:', original_message)
#             interleaved.append(updated_message)
#             interleaved.append("")  # Empty line for readability
#             current_turn += 1
    
#     return "\n".join(interleaved)


def resolve_agent_name(agent_id: str) -> str:
    """Resolve a human-friendly agent display name from AgentProfile; fallback to the ID."""
    try:
        agent_profile = AgentProfile.get(pk=agent_id)
        first_name = getattr(agent_profile, "first_name", "") or ""
        last_name = getattr(agent_profile, "last_name", "") or ""
        full_name = f"{first_name} {last_name}".strip()
        return full_name if full_name else agent_id
    except Exception:
        return agent_id


def process_agent_observations(episode_log: EpisodeLog) -> str:
    """Format the agent observation history with agent display names."""
    observations = episode_log.agent_observation_history or []
    agent_ids = episode_log.agents or []
    lines = []
    for idx, observation in enumerate(observations):
        agent_id = agent_ids[idx] if idx < len(agent_ids) else f"Agent_{idx + 1}"
        agent_name = resolve_agent_name(agent_id)
        lines.append(f"{agent_name}:")
        lines.append(observation if isinstance(observation, str) else str(observation))
        if idx < len(observations) - 1:
            lines.append("")
    return "\n".join(lines) if lines else "No agent observations available"


def process_episode_log(episode_log: EpisodeLog) -> str:
    """Process an EpisodeLog and return formatted content."""
    messages = episode_log.messages
    
    if not messages:
        return "No messages found in episode"
    
    # Extract scenario using ParallelSotopiaEnv
    scenario = extract_scenario_from_episode_log(episode_log)
    
    # Extract turn messages
    turn_messages = extract_turn_messages(messages)
    
    # Format the content exactly as requested
    content_lines = []
    
    # Add scenario at the top
    content_lines.append(scenario)
    content_lines.append("")  # Empty line separator
    
    # Add turn messages - one per line as requested
    for turn_message in turn_messages:
        content_lines.append(turn_message)
    
    # Add rewards after the conversation
    content_lines.append("")  # Empty line separator
    content_lines.append("REWARDS:")
    content_lines.append(str(episode_log.rewards))
    
    # Add reasoning after the rewards
    content_lines.append("")  # Empty line separator
    content_lines.append("REASONING:")
    content_lines.append(episode_log.reasoning)
    
    return "\n".join(content_lines)


def process_raw_episode_log(episode_log: EpisodeLog) -> str:
    """Process an EpisodeLog and return formatted content with raw messages."""
    # Extract scenario using ParallelSotopiaEnv
    scenario = extract_scenario_from_episode_log(episode_log)
    
    # Extract interleaved raw messages
    raw_conversation = extract_raw_messages_interleaved(episode_log)
    
    # Format the content
    content_lines = []
    
    # Add scenario at the top
    content_lines.append(scenario)
    content_lines.append("")  # Empty line separator
    
    # Add raw conversation
    content_lines.append("RAW CONVERSATION (with internal reasoning):")
    content_lines.append("")
    content_lines.append(raw_conversation)
    
    # Add rewards after the conversation
    content_lines.append("")  # Empty line separator
    content_lines.append("REWARDS:")
    content_lines.append(str(episode_log.rewards))
    
    # Add reasoning after the rewards
    content_lines.append("")  # Empty line separator
    content_lines.append("REASONING:")
    content_lines.append(episode_log.reasoning)
    
    return "\n".join(content_lines)


def save_episodes_for_env_ids(
    env_ids: List[str], 
    tag: str, 
    models: List[str], 
    base_output_dir: str = "episode_outputs",
    output_dir_name: Optional[str] = None
):
    """
    Save formatted episode messages for given environment IDs, filtered by tag and models.
    
    Args:
        env_ids: List of environment IDs to process
        tag: Tag to filter episodes by
        models: List of models to filter episodes by
        base_output_dir: Base directory to save outputs
        output_dir_name: Optional custom name for the output subdirectory under base_output_dir.
            If not provided, the tag will be used as the subdirectory name.
    """
    
    # Create base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subdirectory for outputs: use provided name if present, else default to tag
    dir_name = output_dir_name if output_dir_name else tag
    tag_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(tag_dir, exist_ok=True)
    
    # Save evaluation configuration as JSON
    eval_config = {
        "tag": tag,
        "models": models,
        "env_ids": env_ids,
        "timestamp": datetime.now().isoformat(),
        "base_output_dir": base_output_dir,
        "output_dir_name": dir_name,
        "output_dir_path": tag_dir,
    }
    
    config_file = os.path.join(tag_dir, "eval_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved evaluation configuration to {config_file}")
    
    for env_id in env_ids:
        print(f"Processing environment ID: {env_id}")
        
        # Create directory for this environment inside the tag directory
        env_dir = os.path.join(tag_dir, env_id)
        os.makedirs(env_dir, exist_ok=True)
        
        try:
            # Query episodes for this environment and tag
            candidate_episodes = EpisodeLog.find(
                (EpisodeLog.environment == env_id) & (EpisodeLog.tag == tag)
            ).all()
            
            # Filter by models list
            episodes = []
            for episode in candidate_episodes:
                if episode.models == models:
                    episodes.append(episode)
            

            episodes.sort(key=lambda ep: tuple(ep.agents))
            
            print(f"Found {len(episodes)} episodes for environment {env_id} with tag '{tag}' and models {models}")
            
            if not episodes:
                print(f"No episodes found for environment {env_id} with tag '{tag}' and models {models}")
                continue
            
            # Process each episode
            for episode_idx, episode in enumerate(episodes):
                # Create subdirectory for this episode
                episode_dir = os.path.join(env_dir, f"episode_{episode_idx + 1}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # Process the episode and format content
                formatted_content = process_episode_log(episode)
                
                # Save regular messages to text file
                output_file = os.path.join(episode_dir, "messages.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                
                print(f"Saved episode {episode_idx + 1} to {output_file}")
                
                # Process and save raw messages if available
                if episode.raw_messages:
                    raw_formatted_content = process_raw_episode_log(episode)
                    raw_output_file = os.path.join(episode_dir, "raw_messages.txt")
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(raw_formatted_content)
                    
                    print(f"Saved raw messages for episode {episode_idx + 1} to {raw_output_file}")
                else:
                    print(f"No raw messages available for episode {episode_idx + 1}")
                
                # Process and save agent observation history if available
                if getattr(episode, "agent_observation_history", None):
                    observations_formatted = process_agent_observations(episode)
                    observations_output_file = os.path.join(episode_dir, "agent_observations.txt")
                    with open(observations_output_file, 'w', encoding='utf-8') as f:
                        f.write(observations_formatted)
                    print(
                        f"Saved agent observations for episode {episode_idx + 1} to {observations_output_file}"
                    )
                else:
                    print(
                        f"No agent observations available for episode {episode_idx + 1}"
                    )
                
        except Exception as e:
            print(f"Error processing environment {env_id}: {str(e)}")
            continue
    
    print("Processing complete!")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Save formatted episode messages and artifacts.")
    parser.add_argument("--output-dir-name", dest="output_dir_name", type=str, default=None,
                        help="Optional custom subdirectory name under the base output directory. Defaults to the tag.")
    args = parser.parse_args()

    # Example usage - replace with your actual values
    env_ids = [
        "01H7VFHNV13MHN97GAH73E3KM8",
        "01H7VFHN5WVC5HKKVBHZBA553R",
        "01H7VFHN9W0WAFZCBT09PKJJNK",
        "01H7VFHPDZVVCDZR3AARA547CY",
        "01H7VFHPQQQY6H4DNC6NBQ8XTG",
        "01H7VFHN7WJK7VWVRZZTQ6DX9T",
        "01H7VFHPS5WJW2694R1MNC8JFY",
        "01H7VFHNN7XTR99319DS8KZCQM",
        "01H7VFHQ11NAMZS4A2RDGDB01V",
        "01H7VFHPSWGDGEYRP63H2DJKV0",
        "01H7VFHNF4G18PC9JHGRC8A1R6",
        "01H7VFHNNYH3W0VRWVY178K2TK",
        "01H7VFHP8AN5643B0NR0NP00VE",
        "01H7VFHN7A1ZX5KSMT2YN9RXC4",
    ]
    # Example tag and models - replace with your actual values
    tag = "qwen_base_model_vs_qwen_2.5_7b_instruct_sotopia_hard_20_turns_mental_window_2_turns"
    models = [
        "custom/env_model@http://localhost:8020/v1",
        "custom/qwen_base_model@http://localhost:8000/v1",
        "custom/opp_model@http://localhost:8010/v1",
    ]
    
    if not env_ids:
        print("Please provide environment IDs in the env_ids list")
        print("Example:")
        print('env_ids = ["01HAK34YPB1H1RWXQDASDKHSNS", "another_env_id"]')
        return
    
    if not tag:
        print("Please provide a tag")
        return
        
    if not models:
        print("Please provide a models list")
        return
    
    # Run the processing
    save_episodes_for_env_ids(env_ids, tag, models, output_dir_name=tag + "")


if __name__ == "__main__":
    main()
