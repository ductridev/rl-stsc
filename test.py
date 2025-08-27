import os
import yaml
def save_config_snapshot(config, save_path):
    """
    Save config snapshot to the results folder as config_snapshot.json
    Args:
        config: Configuration dictionary
        save_path: Path to save the config snapshot
    """
    config_snapshot_file = os.path.join(save_path, "config_snapshot.yaml")
    try:
        with open(config_snapshot_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config snapshot saved to: {config_snapshot_file}")
    except Exception as e:
        print(f"Failed to save config snapshot: {e}")
"""
Multi-simulation testing script for traffic signal control.
Supports testing with Base (SUMO default), Actuated (queue-based), and DQN simulations.
"""
# import libsumo as traci
import traci
import argparse
import time
import os
from src.utils import import_test_configuration, set_sumo, set_test_path
from src.intersection import Intersection
from src.visualization import Visualization
from src.accident_manager import AccidentManager
from src.base_simulation import SimulationBase
from src.actuated_simulation import ActuatedSimulation
from src.simulation_test import Simulation
import glob
import re


def find_best_model_file(model_folder_path, traffic_light_id=None, specific_filename=None):
    """
    Automatically find the best SKRL model file in the specified folder.
    Looks for SKRL model files with patterns like:
    - skrl_model_{tl_id}_GLOBAL_BEST.pt (highest priority - global best model)
    - skrl_model_{tl_id}_CURRENT_BEST.pt (current best model)
    - skrl_model_{tl_id}_episode_*_BEST.pt (episode-specific best models)
    - skrl_model_{tl_id}_episode_*.pt (episode-specific models)
    
    Args:
        model_folder_path (str): Path to the model folder
        traffic_light_id (str): Traffic light ID to match in model filenames
        specific_filename (str): Specific filename to look for (overrides automatic search)
        
    Returns:
        tuple: (best_model_path, episode_number) or (None, None) if no model found
    """
    if not os.path.exists(model_folder_path):
        print(f"Model folder does not exist: {model_folder_path}")
        return None, None
    
    # If a specific filename is provided, look for it directly
    if specific_filename:
        specific_path = os.path.join(model_folder_path, specific_filename)
        if os.path.exists(specific_path):
            print(f"Found specific model file: {specific_filename}")
            # Try to extract episode info from filename
            if "_episode_" in specific_filename:
                match = re.search(r'episode_(\d+)', specific_filename)
                if match:
                    episode_info = int(match.group(1))
                else:
                    episode_info = "specific"
            elif "GLOBAL_BEST" in specific_filename:
                episode_info = "global_best"
            elif "CURRENT_BEST" in specific_filename:
                episode_info = "current_best"
            elif "_BEST" in specific_filename:
                episode_info = "best"
            else:
                episode_info = "specific"
            return specific_path, episode_info
        else:
            print(f"Specific model file not found: {specific_filename}")
            print(f"Available files in {model_folder_path}:")
            for file in os.listdir(model_folder_path):
                if file.endswith('.pt'):
                    print(f"  - {file}")
            return None, None
    
    # If traffic light ID is provided, use it to filter models
    if traffic_light_id:
        pattern_prefix = f"skrl_model_{traffic_light_id}_"
    else:
        pattern_prefix = "skrl_model_*_"
    
    print(f"Searching for models in: {model_folder_path}")
    print(f"Using pattern prefix: {pattern_prefix}")
    
    # Priority 1: Look for GLOBAL_BEST model
    global_best_files = glob.glob(os.path.join(model_folder_path, f"{pattern_prefix}GLOBAL_BEST.pt"))
    if global_best_files:
        best_file = max(global_best_files, key=os.path.getmtime)
        print(f"Found global best model: {os.path.basename(best_file)}")
        return best_file, "global_best"
    
    # Priority 2: Look for CURRENT_BEST model
    current_best_files = glob.glob(os.path.join(model_folder_path, f"{pattern_prefix}CURRENT_BEST.pt"))
    if current_best_files:
        best_file = max(current_best_files, key=os.path.getmtime)
        print(f"Found current best model: {os.path.basename(best_file)}")
        return best_file, "current_best"
    
    # Priority 3: Look for episode-specific BEST models
    best_episode_files = glob.glob(os.path.join(model_folder_path, f"{pattern_prefix}episode_*_BEST.pt"))
    if best_episode_files:
        episode_numbers = []
        for file in best_episode_files:
            match = re.search(r'episode_(\d+)_BEST', os.path.basename(file))
            if match:
                episode_numbers.append((int(match.group(1)), file))
        
        if episode_numbers:
            # Sort by episode number and get the highest
            episode_numbers.sort(key=lambda x: x[0], reverse=True)
            highest_episode, best_file = episode_numbers[0]
            print(f"Found best episode model: {os.path.basename(best_file)} (episode {highest_episode})")
            return best_file, highest_episode
    
    # Priority 4: Look for regular episode models and find the highest episode
    episode_files = glob.glob(os.path.join(model_folder_path, f"{pattern_prefix}episode_*.pt"))
    # Filter out BEST files that were already checked
    episode_files = [f for f in episode_files if "_BEST.pt" not in f]
    
    if episode_files:
        episode_numbers = []
        for file in episode_files:
            match = re.search(r'episode_(\d+)\.pt$', os.path.basename(file))
            if match:
                episode_numbers.append((int(match.group(1)), file))
        
        if episode_numbers:
            # Sort by episode number and get the highest
            episode_numbers.sort(key=lambda x: x[0], reverse=True)
            highest_episode, best_file = episode_numbers[0]
            print(f"Found highest episode model: {os.path.basename(best_file)} (episode {highest_episode})")
            return best_file, highest_episode
    
    # Priority 5: Look for any SKRL model files
    all_skrl_files = glob.glob(os.path.join(model_folder_path, "skrl_model_*.pt"))
    if all_skrl_files:
        latest_file = max(all_skrl_files, key=os.path.getmtime)
        print(f"Found latest SKRL model: {os.path.basename(latest_file)}")
        return latest_file, "latest"
    
    print(f"No SKRL model files found in {model_folder_path}")
    return None, None

def list_available_models(model_folder_path, traffic_light_id=None):
    """
    List all available SKRL model files in the specified folder.
    
    Args:
        model_folder_path (str): Path to the model folder
        traffic_light_id (str): Traffic light ID to filter models (optional)
    """
    if not os.path.exists(model_folder_path):
        print(f"Model folder does not exist: {model_folder_path}")
        return
    
    # Get all .pt files
    all_pt_files = glob.glob(os.path.join(model_folder_path, "*.pt"))
    
    # Filter by traffic light ID if provided
    if traffic_light_id:
        filtered_files = [f for f in all_pt_files if f"skrl_model_{traffic_light_id}_" in os.path.basename(f)]
    else:
        filtered_files = [f for f in all_pt_files if "skrl_model_" in os.path.basename(f)]
    
    if not filtered_files:
        print(f"No SKRL model files found in {model_folder_path}")
        if traffic_light_id:
            print(f"(searched for models with traffic light ID: {traffic_light_id})")
        return
    
    print(f"Available models in {model_folder_path}:")
    if traffic_light_id:
        print(f"(filtered for traffic light ID: {traffic_light_id})")
    
    # Categorize files
    global_best = []
    current_best = []
    episode_best = []
    regular_episode = []
    others = []
    
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        if "GLOBAL_BEST" in filename:
            global_best.append(filename)
        elif "CURRENT_BEST" in filename:
            current_best.append(filename)
        elif "_BEST.pt" in filename:
            episode_best.append(filename)
        elif "episode_" in filename:
            regular_episode.append(filename)
        else:
            others.append(filename)
    
    # Print categorized results
    if global_best:
        print("  Global Best Models:")
        for model in global_best:
            print(f"    - {model}")
    
    if current_best:
        print("  Current Best Models:")
        for model in current_best:
            print(f"    - {model}")
    
    if episode_best:
        print("  Episode Best Models (latest 5):")
        # Sort by episode number
        episode_best_sorted = []
        for model in episode_best:
            match = re.search(r'episode_(\d+)_BEST', model)
            if match:
                episode_num = int(match.group(1))
                episode_best_sorted.append((episode_num, model))
        episode_best_sorted.sort(reverse=True)
        for _, model in episode_best_sorted[:5]:
            print(f"    - {model}")
        if len(episode_best_sorted) > 5:
            print(f"    ... and {len(episode_best_sorted) - 5} more episode best models")
    
    if regular_episode:
        print("  Regular Episode Models (latest 5):")
        # Sort by episode number
        regular_episode_sorted = []
        for model in regular_episode:
            match = re.search(r'episode_(\d+)\.pt$', model)
            if match:
                episode_num = int(match.group(1))
                regular_episode_sorted.append((episode_num, model))
        regular_episode_sorted.sort(reverse=True)
        for _, model in regular_episode_sorted[:5]:
            print(f"    - {model}")
        if len(regular_episode_sorted) > 5:
            print(f"    ... and {len(regular_episode_sorted) - 5} more episode models")
    
    if others:
        print("  Other Models:")
        for model in others:
            print(f"    - {model}")


def test_base_simulation(config, path):
    """Test with base SUMO traffic light control"""
    print("\n" + "="*50)
    print("TESTING BASE SIMULATION")
    print("="*50)
    
    visualization = Visualization(path=path, dpi=100)
    
    # Set up SUMO
    port = set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])

    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            port=port,
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            # junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
    
    # Initialize base simulation
    base_simulation = SimulationBase(
        agent_cfg=config["agent"],
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        accident_manager=accident_manager,
        visualization=visualization,
        path=path,
        save_interval=1,
        port=port
    )

    base_simulation.testing_mode = True
    
    # Run simulation
    start_time = time.time()
    simulation_time = base_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"Base simulation completed in {simulation_time:.2f}s (total: {total_time:.2f}s)")
    
    # Collect completion tracker data before reset
    completion_data = None
    if hasattr(base_simulation, 'completion_tracker'):
        completion_data = {
            'completed_count': base_simulation.completion_tracker.get_completed_count(),
            'avg_travel_time': base_simulation.completion_tracker.get_average_total_travel_time()
        }
    
    # Collect detailed metrics data
    metrics_data = {}
    if hasattr(base_simulation, 'history'):
        history = base_simulation.history
        if history:
            # Handle nested structure where metrics are organized by traffic light ID
            # Calculate average metrics over the simulation with safe division
            
            # For reward - extract values from nested dict structure
            if 'reward' in history:
                reward_values = []
                for tl_id, values in history['reward'].items():
                    # Convert numpy types to regular Python floats/ints
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    reward_values.extend(converted_values)
                metrics_data['reward'] = sum(reward_values) / max(len(reward_values), 1) if reward_values else 0
            else:
                metrics_data['reward'] = 0
            
            # For waiting time - use 'waiting_time' key
            if 'waiting_time' in history:
                waiting_values = []
                for tl_id, values in history['waiting_time'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    waiting_values.extend(converted_values)
                metrics_data['avg_waiting_time'] = sum(waiting_values) / max(len(waiting_values), 1) if waiting_values else 0
            else:
                metrics_data['avg_waiting_time'] = 0
            
            # For travel delay - use 'travel_delay' key  
            if 'travel_delay' in history:
                delay_values = []
                for tl_id, values in history['travel_delay'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    delay_values.extend(converted_values)
                metrics_data['avg_travel_delay'] = sum(delay_values) / max(len(delay_values), 1) if delay_values else 0
            else:
                metrics_data['avg_travel_delay'] = 0
            
            # For throughput - use 'outflow' key
            if 'outflow' in history:
                outflow_values = []
                for tl_id, values in history['outflow'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    outflow_values.extend(converted_values)
                metrics_data['throughput'] = sum(outflow_values) / max(len(outflow_values), 1) if outflow_values else 0
            else:
                metrics_data['throughput'] = 0
            
            # For queue length - use 'queue_length' key
            if 'queue_length' in history:
                queue_values = []
                for tl_id, values in history['queue_length'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    queue_values.extend(converted_values)
                metrics_data['avg_queue_length'] = sum(queue_values) / max(len(queue_values), 1) if queue_values else 0
            else:
                metrics_data['avg_queue_length'] = 0
                
        print(f"Base simulation metrics collected: {metrics_data}")
    else:
        print("Warning: Base simulation has no history attribute for metrics collection")
    
    # Save metrics and plots
    print("Saving base simulation metrics...")
    base_simulation.save_metrics(episode=1)
    base_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    base_simulation.reset_history()
    
    # Reset completion tracker after analysis and plotting
    if hasattr(base_simulation, 'completion_tracker'):
        base_simulation.completion_tracker.reset()
        
    return completion_data, metrics_data


def test_actuated_simulation(config, path):
    """Test with traffic-actuated control"""
    print("\n" + "="*50)
    print("TESTING ACTUATED SIMULATION")
    print("="*50)
    
    visualization = Visualization(path=path, dpi=100)
    
    # Set up SUMO
    port = set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])

    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            port=port,
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            # junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
    
    # Initialize actuated simulation
    actuated_simulation = ActuatedSimulation(
        agent_cfg=config["agent"],
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        accident_manager=accident_manager,
        visualization=visualization,
        path=path,
        save_interval=1,
        min_green_time=config["agent"].get("min_green_time", 20),
        max_green_time=config["agent"].get("max_green_time", 60),
        detection_threshold=config["agent"].get("detection_threshold", 2),
        port=port
    )

    actuated_simulation.testing_mode = True
    
    # Run simulation
    start_time = time.time()
    simulation_time = actuated_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"Actuated simulation completed in {simulation_time:.2f}s (total: {total_time:.2f}s)")
    
    # Collect completion tracker data before reset
    completion_data = None
    if hasattr(actuated_simulation, 'completion_tracker'):
        completion_data = {
            'completed_count': actuated_simulation.completion_tracker.get_completed_count(),
            'avg_travel_time': actuated_simulation.completion_tracker.get_average_total_travel_time()
        }
    
    # Collect detailed metrics data
    metrics_data = {}
    if hasattr(actuated_simulation, 'history'):
        history = actuated_simulation.history
        if history:
            # Handle nested structure where metrics are organized by traffic light ID
            # Calculate average metrics over the simulation with safe division
            
            # For reward - extract values from nested dict structure
            if 'reward' in history:
                reward_values = []
                for tl_id, values in history['reward'].items():
                    # Convert numpy types to regular Python floats/ints
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    reward_values.extend(converted_values)
                metrics_data['reward'] = sum(reward_values) / max(len(reward_values), 1) if reward_values else 0
            else:
                metrics_data['reward'] = 0
            
            # For waiting time - use 'waiting_time' key
            if 'waiting_time' in history:
                waiting_values = []
                for tl_id, values in history['waiting_time'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    waiting_values.extend(converted_values)
                metrics_data['avg_waiting_time'] = sum(waiting_values) / max(len(waiting_values), 1) if waiting_values else 0
            else:
                metrics_data['avg_waiting_time'] = 0
            
            # For travel delay - use 'travel_delay' key  
            if 'travel_delay' in history:
                delay_values = []
                for tl_id, values in history['travel_delay'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    delay_values.extend(converted_values)
                metrics_data['avg_travel_delay'] = sum(delay_values) / max(len(delay_values), 1) if delay_values else 0
            else:
                metrics_data['avg_travel_delay'] = 0
            
            # For throughput - use 'outflow' key
            if 'outflow' in history:
                outflow_values = []
                for tl_id, values in history['outflow'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    outflow_values.extend(converted_values)
                metrics_data['throughput'] = sum(outflow_values) / max(len(outflow_values), 1) if outflow_values else 0
            else:
                metrics_data['throughput'] = 0
            
            # For queue length - use 'queue_length' key
            if 'queue_length' in history:
                queue_values = []
                for tl_id, values in history['queue_length'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    queue_values.extend(converted_values)
                metrics_data['avg_queue_length'] = sum(queue_values) / max(len(queue_values), 1) if queue_values else 0
            else:
                metrics_data['avg_queue_length'] = 0
                
        print(f"Actuated simulation metrics collected: {metrics_data}")
    else:
        print("Warning: Actuated simulation has no history attribute for metrics collection")
    
    # Save metrics and plots
    print("Saving actuated simulation metrics...")
    actuated_simulation.save_metrics(episode=1)
    actuated_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    actuated_simulation.reset_history()
    
    # Reset completion tracker after analysis and plotting
    if hasattr(actuated_simulation, 'completion_tracker'):
        actuated_simulation.completion_tracker.reset()
        
    return completion_data, metrics_data


def test_dqn_simulation(config, path, specific_model_file=None):
    """Test with DQN agent (requires pre-trained model)"""
    print("\n" + "="*50)
    print("TESTING DQN SIMULATION")
    print("="*50)
    
    # Get model folder from config
    model_folder = config['model_folder']  # Default to model_13
    
    # Check if model_folder is pointing directly to a .pt file
    if model_folder.endswith('.pt'):
        # Direct .pt file path specified
        model_file_path = model_folder
        episode_info = "direct"
        print(f"Using direct .pt model file: {model_file_path}")
        
        # Check if the file exists
        if not os.path.exists(model_file_path):
            # If not found, try prepending "models/" if not already there
            if not model_file_path.startswith('models/'):
                model_file_path_with_models = os.path.join('models', model_file_path)
                if os.path.exists(model_file_path_with_models):
                    model_file_path = model_file_path_with_models
                    print(f"Found model file at: {model_file_path}")
                else:
                    print(f"Error: Direct model file not found: {model_folder}")
                    print(f"Also checked: {model_file_path_with_models}")
                    print("Please check the file path in your configuration.")
                    return
            else:
                print(f"Error: Direct model file not found: {model_file_path}")
                print("Please check the file path in your configuration.")
                return
    else:
        # Traditional folder-based model loading
        model_folder_path = os.path.join("models", model_folder)
        print(f"Looking for DQN models in: {model_folder_path}")
        
        # Extract traffic light ID for model file matching
        traffic_light_id = None
        if config["traffic_lights"] and len(config["traffic_lights"]) > 0:
            # Get the first traffic light ID
            first_tl = config["traffic_lights"][0]
            if isinstance(first_tl, dict):
                # Find the key that contains 'id'
                for key, value in first_tl.items():
                    if isinstance(value, dict) and 'id' in value:
                        traffic_light_id = value['id']
                        break
            
        print(f"Traffic light ID for model search: {traffic_light_id}")
        
        # Auto-find the best model file (or use specific filename if provided)
        model_file_path, episode_info = find_best_model_file(model_folder_path, traffic_light_id, specific_model_file)
        
        if model_file_path is None:
            print(f"Warning: No DQN model files found in {model_folder_path}")
            if specific_model_file:
                print(f"Specific file '{specific_model_file}' not found.")
            print(f"\nTo see available models, run:")
            print(f"python test.py --list-models --model-folder {model_folder}")
            print("\nAvailable model folders:")
            models_dir = "models"
            if os.path.exists(models_dir):
                for item in os.listdir(models_dir):
                    item_path = os.path.join(models_dir, item)
                    if os.path.isdir(item_path):
                        pth_files = glob.glob(os.path.join(item_path, "*.pth"))
                        pt_files = glob.glob(os.path.join(item_path, "*.pt"))
                        total_files = len(pth_files) + len(pt_files)
                        print(f"  - {item}: {total_files} model files (.pth + .pt)")
            print("Skipping DQN simulation. Train a model first or check the model_folder in config.")
            return
    
    # Common path for both direct .pt files and found model files
    print(f"Using model: {model_file_path}")
    
    visualization = Visualization(path=path, dpi=100)
    
    # Set up SUMO
    # set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
    port = set_sumo(True, config["sumo_cfg_file"], config["max_steps"])

    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            port=port,
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            # junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
    
    # Initialize DQN simulation in testing mode
    dqn_simulation = Simulation(
        visualization=visualization,
        agent_cfg=config["agent"],
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        accident_manager=accident_manager,
        interphase_duration=config["interphase_duration"],
        path=path,
        model_file_path=model_file_path,
        use_skrl=True,
        port=port
    )
    
    # Run simulation with episode=1 (pure exploitation mode since model is loaded)
    start_time = time.time()
    dqn_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"DQN simulation completed in {total_time:.2f}s")
    
    # Debug: Check completion tracker before data collection
    # print(f"DEBUG: DQN simulation has completion_tracker: {hasattr(dqn_simulation, 'completion_tracker')}")
    # if hasattr(dqn_simulation, 'completion_tracker'):
    #     print(f"DEBUG: DQN completion tracker completed count: {dqn_simulation.completion_tracker.get_completed_count()}")
    #     print(f"DEBUG: DQN completion tracker avg travel time: {dqn_simulation.completion_tracker.get_average_total_travel_time()}")
    
    # Collect completion tracker data before reset
    completion_data = None
    if hasattr(dqn_simulation, 'completion_tracker'):
        completion_data = {
            'completed_count': dqn_simulation.completion_tracker.get_completed_count(),
            'avg_travel_time': dqn_simulation.completion_tracker.get_average_total_travel_time()
        }
        print(f"DEBUG: DQN completion_data collected: {completion_data}")
    else:
        print("DEBUG: DQN simulation does not have completion_tracker attribute")
    
    # Collect detailed metrics data
    metrics_data = {}
    if hasattr(dqn_simulation, 'history'):
        history = dqn_simulation.history
        if history:
            # Handle nested structure where metrics are organized by traffic light ID
            # Calculate average metrics over the simulation with safe division
            
            # For reward - extract values from nested dict structure
            if 'reward' in history:
                reward_values = []
                for tl_id, values in history['reward'].items():
                    # Convert numpy types to regular Python floats/ints
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    reward_values.extend(converted_values)
                metrics_data['reward'] = sum(reward_values) / max(len(reward_values), 1) if reward_values else 0
            else:
                metrics_data['reward'] = 0
            
            # For waiting time - use 'waiting_time' key
            if 'waiting_time' in history:
                waiting_values = []
                for tl_id, values in history['waiting_time'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    waiting_values.extend(converted_values)
                metrics_data['avg_waiting_time'] = sum(waiting_values) / max(len(waiting_values), 1) if waiting_values else 0
            else:
                metrics_data['avg_waiting_time'] = 0
            
            # For travel delay - use 'travel_delay' key  
            if 'travel_delay' in history:
                delay_values = []
                for tl_id, values in history['travel_delay'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    delay_values.extend(converted_values)
                metrics_data['avg_travel_delay'] = sum(delay_values) / max(len(delay_values), 1) if delay_values else 0
            else:
                metrics_data['avg_travel_delay'] = 0
            
            # For throughput - use 'outflow' key
            if 'outflow' in history:
                outflow_values = []
                for tl_id, values in history['outflow'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    outflow_values.extend(converted_values)
                metrics_data['throughput'] = sum(outflow_values) / max(len(outflow_values), 1) if outflow_values else 0
            else:
                metrics_data['throughput'] = 0
            
            # For queue length - use 'queue_length' key
            if 'queue_length' in history:
                queue_values = []
                for tl_id, values in history['queue_length'].items():
                    converted_values = [float(v) if hasattr(v, 'item') else v for v in values]
                    queue_values.extend(converted_values)
                metrics_data['avg_queue_length'] = sum(queue_values) / max(len(queue_values), 1) if queue_values else 0
            else:
                metrics_data['avg_queue_length'] = 0
                
        print(f"DQN simulation metrics collected: {metrics_data}")
    else:
        print("Warning: DQN simulation has no history attribute for metrics collection")
    
    # Save metrics and plots
    print("Saving DQN simulation metrics and plots...")
    dqn_simulation.save_plot(episode=1)
    dqn_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    dqn_simulation.reset_history()
    
    # Reset completion tracker after analysis and plotting
    if hasattr(dqn_simulation, 'completion_tracker'):
        dqn_simulation.completion_tracker.reset()
        
    return completion_data, metrics_data


def display_metrics_comparison(all_metrics_results, simulations_to_run):
    """
    Display a detailed comparison of metrics across all simulation types.
    
    Args:
        all_metrics_results (dict): Dictionary containing metrics data for each simulation type
        simulations_to_run (list): List of simulation types that were run
    """
    print("\n" + "="*70)
    print("DETAILED METRICS COMPARISON")
    print("="*70)
    
    # Define metric names and units for better display
    metric_info = {
        'reward': ('Total Reward', ''),
        'avg_waiting_time': ('Avg Waiting Time', 's'),
        'avg_travel_delay': ('Avg Travel Delay', 's'),
        'throughput': ('Throughput (Outflow)', 'vehicles/300 steps'),
        'avg_queue_length': ('Avg Queue Length', 'meters')
    }
    
    # Method mapping for display names
    method_mapping = {
        'base': 'Baseline (Fixed)',
        'actuated': 'Research Actuated',
        'dqn': 'SKRL DQN'
    }
    
    # Check if we have metrics data
    available_methods = []
    for method in simulations_to_run:
        if method in all_metrics_results and 'metrics' in all_metrics_results[method]:
            available_methods.append(method)
    
    if not available_methods:
        print("No metrics data available for comparison.")
        return
    
    print("Methods compared:", ", ".join([method_mapping.get(method, method.title()) for method in available_methods]))
    print()
    
    # Display each metric comparison
    for metric_key, (metric_name, unit) in metric_info.items():
        print(f"\n{metric_name}:")
        print("-" * (len(metric_name) + 1))
        
        values = {}
        for method in available_methods:
            if metric_key in all_metrics_results[method]['metrics']:
                value = all_metrics_results[method]['metrics'][metric_key]
                values[method] = value
                method_display = method_mapping.get(method, method.title())
                print(f"  {method_display:20}: {value:8.2f} {unit}")
        
        # Find best and worst values
        if values:
            if metric_key == 'reward' or metric_key == 'throughput':
                # Higher is better for reward and throughput
                best_method = max(values, key=values.get)
                worst_method = min(values, key=values.get)
            else:
                # Lower is better for waiting time, travel delay, queue length
                best_method = min(values, key=values.get)
                worst_method = max(values, key=values.get)
            
            best_display = method_mapping.get(best_method, best_method.title())
            print(f"  → Best: {best_display} ({values[best_method]:.2f} {unit})")
            
            # Calculate improvements vs baseline if available
            if 'base' in values and best_method != 'base':
                baseline_value = values['base']
                best_value = values[best_method]
                
                if baseline_value != 0:
                    if metric_key == 'reward' or metric_key == 'throughput':
                        improvement = ((best_value - baseline_value) / abs(baseline_value)) * 100
                    else:
                        improvement = ((baseline_value - best_value) / baseline_value) * 100
                    
                    print(f"  → Improvement vs Baseline: {improvement:+.1f}%")
    
    # Overall performance summary
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*50)
    
    # Calculate performance scores for each method
    performance_scores = {}
    
    for method in available_methods:
        score = 0
        metric_count = 0
        
        if 'metrics' in all_metrics_results[method]:
            metrics = all_metrics_results[method]['metrics']
            
            # Normalize and score each metric (0-100 scale)
            all_values = {metric: [] for metric in metric_info.keys()}
            
            # Collect all values for normalization
            for m in available_methods:
                if 'metrics' in all_metrics_results[m]:
                    for metric_key in metric_info.keys():
                        if metric_key in all_metrics_results[m]['metrics']:
                            all_values[metric_key].append(all_metrics_results[m]['metrics'][metric_key])
            
            # Calculate normalized scores
            for metric_key in metric_info.keys():
                if metric_key in metrics and all_values[metric_key]:
                    value = metrics[metric_key]
                    min_val = min(all_values[metric_key])
                    max_val = max(all_values[metric_key])
                    
                    if max_val != min_val:
                        if metric_key == 'reward' or metric_key == 'throughput':
                            # Higher is better
                            normalized_score = ((value - min_val) / (max_val - min_val)) * 100
                        else:
                            # Lower is better
                            normalized_score = ((max_val - value) / (max_val - min_val)) * 100
                        
                        score += normalized_score
                        metric_count += 1
            
            if metric_count > 0:
                performance_scores[method] = score / metric_count
    
    # Display performance ranking
    if performance_scores:
        sorted_methods = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Performance Ranking (0-100 scale, higher is better):")
        for i, (method, score) in enumerate(sorted_methods, 1):
            method_display = method_mapping.get(method, method.title())
            print(f"  {i}. {method_display:20}: {score:5.1f}/100")
        
        print(f"\n🏆 Best Overall Performance: {method_mapping.get(sorted_methods[0][0], sorted_methods[0][0].title())}")


def test_green_time_comparison(config, path, green_times=[5, 10, 15, 20, 25]):
    """Test performance across different green time durations"""
    print("\n" + "="*60)
    print("TESTING GREEN TIME COMPARISON")
    print("="*60)
    
    results = {}
    
    for green_time in green_times:
        print(f"\nTesting with {green_time}s green time...")
        
        # Modify agent config for this green time
        modified_config = config.copy()
        modified_config["agent"] = config["agent"].copy()
        modified_config["agent"]["min_green_time"] = green_time
        modified_config["agent"]["max_green_time"] = green_time  # Fixed green time
        
        # Test actuated simulation with fixed green time
        completion_data, metrics_data = test_actuated_simulation(modified_config, path)
        
        results[green_time] = {
            'completion_tracker': completion_data,
            'metrics': metrics_data
        }
    
    # Generate comparison plots
    create_green_time_comparison_plots(results, path)
    
    return results


def create_green_time_comparison_plots(results, save_path):
    """Create plots comparing performance across green times"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print("\nGenerating green time comparison plots...")
    
    # Extract data for plotting
    green_times = []
    avg_waiting_times = []
    avg_queue_lengths = []
    avg_travel_delays = []
    throughputs = []
    avg_travel_times = []
    completed_counts = []
    
    for green_time in sorted(results.keys()):
        data = results[green_time]
        if 'metrics' in data and 'completion_tracker' in data:
            metrics = data['metrics']
            completion = data['completion_tracker']
            
            green_times.append(green_time)
            avg_waiting_times.append(metrics.get('avg_waiting_time', 0))
            avg_queue_lengths.append(metrics.get('avg_queue_length', 0))
            avg_travel_delays.append(metrics.get('avg_travel_delay', 0))
            throughputs.append(metrics.get('throughput', 0))
            
            if completion:
                avg_travel_times.append(completion.get('avg_travel_time', 0))
                completed_counts.append(completion.get('completed_count', 0))
            else:
                avg_travel_times.append(0)
                completed_counts.append(0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Traffic Signal Performance vs Green Time Duration', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Waiting Time
    axes[0, 0].plot(green_times, avg_waiting_times, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 0].set_title('Average Waiting Time vs Green Time')
    axes[0, 0].set_xlabel('Green Time Duration (seconds)')
    axes[0, 0].set_ylabel('Average Waiting Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Queue Length
    axes[0, 1].plot(green_times, avg_queue_lengths, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_title('Average Queue Length vs Green Time')
    axes[0, 1].set_xlabel('Green Time Duration (seconds)')
    axes[0, 1].set_ylabel('Average Queue Length (vehicles)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Throughput
    axes[0, 2].plot(green_times, throughputs, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 2].set_title('Throughput vs Green Time')
    axes[0, 2].set_xlabel('Green Time Duration (seconds)')
    axes[0, 2].set_ylabel('Throughput (vehicles/step)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Average Travel Delay
    axes[1, 0].plot(green_times, avg_travel_delays, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_title('Average Travel Delay vs Green Time')
    axes[1, 0].set_xlabel('Green Time Duration (seconds)')
    axes[1, 0].set_ylabel('Average Travel Delay (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Average Travel Time
    axes[1, 1].plot(green_times, avg_travel_times, 'o-', linewidth=2, markersize=8, color='blue')
    axes[1, 1].set_title('Average Travel Time vs Green Time')
    axes[1, 1].set_xlabel('Green Time Duration (seconds)')
    axes[1, 1].set_ylabel('Average Travel Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Completed Vehicle Count
    axes[1, 2].plot(green_times, completed_counts, 'o-', linewidth=2, markersize=8, color='brown')
    axes[1, 2].set_title('Completed Vehicles vs Green Time')
    axes[1, 2].set_xlabel('Green Time Duration (seconds)')
    axes[1, 2].set_ylabel('Number of Completed Vehicles')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'green_time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Green time comparison plot saved to: {plot_path}")
    
    # Create summary table
    table_data = []
    for green_time in sorted(results.keys()):
        data = results[green_time]
        if 'metrics' in data and 'completion_tracker' in data:
            metrics = data['metrics']
            completion = data['completion_tracker']
            
            row = {
                'Green Time (s)': green_time,
                'Avg Waiting Time (s)': round(metrics.get('avg_waiting_time', 0), 2),
                'Avg Queue Length': round(metrics.get('avg_queue_length', 0), 2),
                'Avg Travel Delay (s)': round(metrics.get('avg_travel_delay', 0), 2),
                'Throughput (veh/step)': round(metrics.get('throughput', 0), 4),
                'Completed Vehicles': completion.get('completed_count', 0) if completion else 0,
                'Avg Travel Time (s)': round(completion.get('avg_travel_time', 0), 2) if completion else 0
            }
            table_data.append(row)
    
    # Save summary table
    df = pd.DataFrame(table_data)
    csv_path = os.path.join(save_path, 'green_time_performance_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Performance summary saved to: {csv_path}")
    
    # Print summary
    print("\nGREEN TIME PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test traffic signal control with different simulation types')
    parser.add_argument('--config', '-c', 
                       default='config/testing_testngatu6x1-4.yaml',
                       help='Path to testing configuration file')
    parser.add_argument('--simulations', '-s', 
                       nargs='+', 
                       choices=['base', 'actuated', 'dqn', 'all', 'green-time'],
                       default=['all'],
                       help='Simulation types to run')
    parser.add_argument('--gui', '-g', 
                       action='store_true',
                       help='Enable SUMO GUI')
    parser.add_argument('--model-folder', '-m',
                       help='Model folder or direct .pt file path to use for DQN simulation (e.g., model_11, model_12, or models/model_14/skrl_model_NgatuNorthEast_GLOBAL_BEST.pt). Overrides config setting.')
    parser.add_argument('--model-file', '-f',
                       help='Specific model filename to use within the model folder (e.g., skrl_model_NgatuNorthEast_episode_999_BEST.pt). Works with --model-folder.')
    parser.add_argument('--list-models', '-l',
                       action='store_true',
                       help='List available model files in the specified model folder and exit')
    parser.add_argument('--green-times', 
                       nargs='+', 
                       type=int,
                       default=[5, 10, 15, 20, 25],
                       help='Green time durations to test for green-time simulation (seconds)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = import_test_configuration(args.config)
    
    # Handle list-models option
    if args.list_models:
        model_folder = args.model_folder if args.model_folder else config.get("model_folder", "model_41")
        if model_folder.endswith('.pt'):
            print("Cannot list models for a direct .pt file path. Please specify a model folder.")
            return
        
        model_folder_path = os.path.join("models", model_folder)
        
        # Extract traffic light ID for filtering
        traffic_light_id = None
        if config["traffic_lights"] and len(config["traffic_lights"]) > 0:
            first_tl = config["traffic_lights"][0]
            if isinstance(first_tl, dict):
                for key, value in first_tl.items():
                    if isinstance(value, dict) and 'id' in value:
                        traffic_light_id = value['id']
                        break
        
        print(f"Listing models in folder: {model_folder_path}")
        list_available_models(model_folder_path, traffic_light_id)
        return
    
    # Override GUI setting if specified
    if args.gui:
        config["gui"] = True
        
    # Override model folder if specified via command line
    if args.model_folder:
        config["model_folder"] = args.model_folder
        if args.model_folder.endswith('.pt'):
            print(f"Using direct .pt model file from command line: {args.model_folder}")
        else:
            print(f"Using model folder from command line: {args.model_folder}")
    elif "model_folder" not in config:
        # Set default if not in config
        config["model_folder"] = "model_13"
        print(f"Using default model folder: model_13")
    else:
        if config["model_folder"].endswith('.pt'):
            print(f"Using direct .pt model file from config: {config['model_folder']}")
        else:
            print(f"Using model folder from config: {config['model_folder']}")
    
    # Display specific model file if provided
    if args.model_file:
        print(f"Using specific model file: {args.model_file}")
    
    # Set up testing path
    path = set_test_path(config["models_path_name"])
    print(f"Test results will be saved to: {path}")
    
    # Determine which simulations to run
    simulations_to_run = args.simulations
    if 'all' in simulations_to_run:
        simulations_to_run = ['base', 'actuated', 'dqn']
    
    # Handle green-time testing
    # if 'green-time' in simulations_to_run:
    #     print("\nStarting green time comparison testing...")
    #     print(f"Testing green times: {args.green_times} seconds")
        
    #     # Generate routes for green time testing
    #     print("\n" + "="*50)
    #     print("GENERATING ROUTES FOR GREEN TIME TESTING")
    #     print("="*50)
        
    #     simulation_path = config["sumo_cfg_file"].split("/")[1]
    #     demand_level = 'medium'  # Use medium demand for consistent testing
        
    #     print(f"Generating routes for {demand_level} demand level...")
    #     Intersection.generate_residential_demand_routes(
    #         config,
    #         simulation_path,
    #         demand_level=demand_level,
    #         enable_bicycle=True,
    #         enable_pedestrian=True,
    #         enable_motorcycle=True,
    #         enable_passenger=True,
    #         enable_truck=True,
    #     )
        
    #     # Run green time comparison
    #     try:
    #         green_time_results = test_green_time_comparison(config, path, args.green_times)
    #         print(f"\nGreen time comparison completed successfully!")
    #     except Exception as e:
    #         print(f"Error during green time testing: {e}")
    #         import traceback
    #         traceback.print_exc()
        
    #     return  # Exit early for green-time testing
    
    print("\nStarting multi-simulation testing...")
    print(f"Running simulations: {', '.join(simulations_to_run)}")
    print(f"Configuration: {args.config}")
    print(f"Max steps: {config['max_steps']}")
    print(f"Traffic lights: {len(config['traffic_lights'])}")
    
    # Generate routes once for all simulations (low, medium, high demand levels)
    print("\n" + "="*50)
    print("GENERATING ROUTES FOR ALL DEMAND LEVELS")
    print("="*50)

    # Run selected simulations
    overall_start = time.time()
    
    # Collect completion tracker results and metrics
    completion_results = {}
    all_metrics_results = {}
    
    simulation_path = config["sumo_cfg_file"].split("/")[1]
    demand_levels = ['medium']
    
    for demand_level in demand_levels:
        if args.config != "config/testing_testngatu6x1EastWestOverflow.yaml":
            print(f"\nGenerating routes for {demand_level} demand level...")
            Intersection.generate_residential_demand_routes(
                config,
                simulation_path,
                demand_level=demand_level,
                enable_bicycle=True,
                enable_pedestrian=True,
                enable_motorcycle=True,
                enable_passenger=True,
                enable_truck=True,
            )
    
        try:
            if 'base' in simulations_to_run:
                result = test_base_simulation(config, path)
                if result:
                    if isinstance(result, tuple) and len(result) == 2:
                        base_completion, base_metrics = result
                        completion_results['baseline'] = {'completion_tracker': base_completion}
                        all_metrics_results['base'] = {'completion_tracker': base_completion, 'metrics': base_metrics}
                    else:
                        # Backward compatibility - only completion data returned
                        base_completion = result
                        completion_results['baseline'] = {'completion_tracker': base_completion}
                        all_metrics_results['base'] = {'completion_tracker': base_completion, 'metrics': {}}
            
            if 'actuated' in simulations_to_run:
                result = test_actuated_simulation(config, path)
                if result:
                    if isinstance(result, tuple) and len(result) == 2:
                        actuated_completion, actuated_metrics = result
                        completion_results['actuated'] = {'completion_tracker': actuated_completion}
                        all_metrics_results['actuated'] = {'completion_tracker': actuated_completion, 'metrics': actuated_metrics}
                    else:
                        # Backward compatibility - only completion data returned
                        actuated_completion = result
                        completion_results['actuated'] = {'completion_tracker': actuated_completion}
                        all_metrics_results['actuated'] = {'completion_tracker': actuated_completion, 'metrics': {}}
            
            if 'dqn' in simulations_to_run:
                result = test_dqn_simulation(config, path, args.model_file)
                if result:
                    if isinstance(result, tuple) and len(result) == 2:
                        dqn_completion, dqn_metrics = result
                        completion_results['dqn'] = {'completion_tracker': dqn_completion}
                        all_metrics_results['dqn'] = {'completion_tracker': dqn_completion, 'metrics': dqn_metrics}
                    else:
                        # Backward compatibility - only completion data returned
                        dqn_completion = result
                        completion_results['dqn'] = {'completion_tracker': dqn_completion}
                        all_metrics_results['dqn'] = {'completion_tracker': dqn_completion, 'metrics': {}}

        except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
    
    overall_time = time.time() - overall_start
    
    # Generate comparison plots across all simulations
    try:
        print("\nGenerating comparison plots...")
        visualization = Visualization(path=path, dpi=100)
        
        # Generate plots comparing all simulation types
        metrics = ["density_avg", "travel_time_avg", "outflow_avg", "queue_length_avg", "waiting_time_avg", "junction_arrival_avg", "stopped_vehicles_avg", "travel_delay_avg"]
        names = []
        
        # Add simulation names based on what was run
        if 'base' in simulations_to_run:
            names.append("base")
        if 'actuated' in simulations_to_run:
            names.append("actuated") 
        if 'dqn' in simulations_to_run:
            names.extend(["skrl_dqn", "dqn"])  # DQN might save under different names
            
        if names:
            visualization.save_plot(episode=1, metrics=metrics, names=names)
            print("Comparison plots generated successfully!")
        
    except Exception as e:
        print(f"Warning: Could not generate comparison plots: {e}")
    
    # Generate vehicle comparison plots from logs
    try:
        print("\nGenerating vehicle comparison plots from logs...")
        visualization = Visualization(path=path, dpi=100)
        
        # Prepare simulation types for vehicle comparison
        vehicle_comparison_types = []
        if 'base' in simulations_to_run:
            vehicle_comparison_types.append("base")
        if 'actuated' in simulations_to_run:
            vehicle_comparison_types.append("actuated")
        if 'dqn' in simulations_to_run:
            vehicle_comparison_types.append("skrl_dqn")  # DQN logs are typically saved as skrl_dqn
            
        if vehicle_comparison_types:
            visualization.create_vehicle_comparison_from_logs(episode=1, simulation_types=vehicle_comparison_types)
            print("Vehicle comparison plots generated successfully!")
        else:
            print("No simulation types available for vehicle comparison.")
        
    except Exception as e:
        print(f"Warning: Could not generate vehicle comparison plots: {e}")
    
    # Generate completion tracker comparison plots
    try:
        print("\nGenerating completion tracker comparison plots...")
        
        if completion_results:
            # Use the visualization instance to create completion tracker plots
            visualization = Visualization(path=path, dpi=100)
            
            # Generate completion tracker comparison plot
            visualization.plot_completed_travel_time_comparison(completion_results, episode=1)
            
            # Generate travel time improvements plot
            visualization.plot_travel_time_improvements(completion_results, episode=1)
            
            print("✓ Completion tracker comparison plots generated successfully!")
            
            # Print summary of completion tracker results
            print("\n" + "="*50)
            print("COMPLETION TRACKER SUMMARY")
            print("="*50)
            
            method_mapping = {
                'actuated': 'Research Actuated',
                'baseline': 'Baseline (Fixed)', 
                'dqn': 'SKRL DQN'
            }
            
            best_travel_time = float('inf')
            best_method = ""
            
            for method_key, method_data in completion_results.items():
                if 'completion_tracker' in method_data:
                    completion_data = method_data['completion_tracker']
                    method_name = method_mapping.get(method_key, method_key.title())
                    completed_count = completion_data['completed_count']
                    avg_travel_time = completion_data['avg_travel_time']
                    
                    print(f"{method_name}:")
                    print(f"  Completed Vehicles: {completed_count}")
                    print(f"  Average Travel Time: {avg_travel_time:.2f}s")
                    
                    if avg_travel_time < best_travel_time and avg_travel_time > 0:
                        best_travel_time = avg_travel_time
                        best_method = method_name
                    print()
            
            if best_method:
                print(f"Best Performance: {best_method} ({best_travel_time:.2f}s average travel time)")
                
                # Calculate improvements vs baseline if available
                if 'baseline' in completion_results:
                    baseline_time = completion_results['baseline']['completion_tracker']['avg_travel_time']
                    if baseline_time > 0:
                        print("\nImprovements vs Baseline:")
                        for method_key in ['actuated', 'dqn']:
                            if method_key in completion_results:
                                method_time = completion_results[method_key]['completion_tracker']['avg_travel_time']
                                if method_time > 0:
                                    improvement = ((baseline_time - method_time) / baseline_time) * 100
                                    method_name = method_mapping.get(method_key, method_key.title())
                                    print(f"  {method_name}: {improvement:+.1f}%")
        else:
            print("No completion tracker data available for plotting")
        
    except Exception as e:
        print(f"Warning: Could not generate completion tracker plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Display detailed metrics comparison
    try:
        print("\nGenerating detailed metrics comparison...")
        display_metrics_comparison(all_metrics_results, simulations_to_run)
        
    except Exception as e:
        print(f"Warning: Could not generate metrics comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("TESTING COMPLETED")
    print("="*50)
    print(f"Total testing time: {overall_time:.2f}s")
    print(f"Results saved to: {path}")

    # Save config snapshot after testing
    save_config_snapshot(config, path)


if __name__ == "__main__":
    main()