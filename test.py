"""
Multi-simulation testing script for traffic signal control.
Supports testing with Base (SUMO default), Actuated (queue-based), and DQN simulations.
"""
import libsumo as traci
import argparse
import time
import os
from src.utils import import_test_configuration, set_sumo, set_test_path
from src.intersection import Intersection
from src.visualization import Visualization
from src.accident_manager import AccidentManager
from src.base_simulation import SimulationBase
from src.actuated_simulation import ActuatedSimulation
from src.simulation import Simulation
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
    
    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']]
        )
    
    # Initialize base simulation
    base_simulation = SimulationBase(
        agent_cfg=config["agent"],
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        accident_manager=accident_manager,
        visualization=visualization,
        path=path,
        save_interval=1
    )
    
    # Set up SUMO
    set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
    
    # Run simulation
    start_time = time.time()
    simulation_time = base_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"Base simulation completed in {simulation_time:.2f}s (total: {total_time:.2f}s)")
    
    # Save metrics and plots
    print("Saving base simulation metrics...")
    base_simulation.save_metrics(episode=1)
    base_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    base_simulation.reset_history()


def test_actuated_simulation(config, path):
    """Test with traffic-actuated control"""
    print("\n" + "="*50)
    print("TESTING ACTUATED SIMULATION")
    print("="*50)
    
    visualization = Visualization(path=path, dpi=100)
    
    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']]
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
        detection_threshold=config["agent"].get("detection_threshold", 2)
    )
    
    # Set up SUMO
    set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
    
    # Run simulation
    start_time = time.time()
    simulation_time = actuated_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"Actuated simulation completed in {simulation_time:.2f}s (total: {total_time:.2f}s)")
    
    # Save metrics and plots
    print("Saving actuated simulation metrics...")
    actuated_simulation.save_metrics(episode=1)
    actuated_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    actuated_simulation.reset_history()


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
    
    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']]
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
        epoch=1,  # Testing mode
        training_steps=float('inf'),  # Disable training by setting to infinity
        updating_target_network_steps=float('inf'),  # Disable target network updates
        save_interval=1
    )
    
    # Set testing mode flag to disable all training operations
    dqn_simulation.testing_mode = True
    
    # Load the trained model
    try:
        # Load the model directly using the found file path
        print(f"Loading model from: {model_file_path}")
        
        # Load model state dict directly into the agent's neural networks
        import torch
        model_state = torch.load(model_file_path, map_location=dqn_simulation.device, weights_only=False)

        # Get the first (and likely only) traffic light agent
        agent_manager = dqn_simulation.agent_manager
        if agent_manager.agents:
            # Load the model into all traffic light agents
            for tl_id, agent in agent_manager.agents.items():
                try:
                    # Try to load the state dict directly
                    if isinstance(model_state, dict):
                        # If the model state is a dictionary, try to find the right keys
                        if "model_state_dict" in model_state:
                            state_dict = model_state["model_state_dict"]
                        else:
                            state_dict = model_state
                    else:
                        state_dict = model_state
                    
                    # Load into both q_network and target_q_network
                    agent.models["q_network"].load_state_dict(state_dict)
                    agent.models["target_q_network"].load_state_dict(state_dict)
                    print(f"Successfully loaded model for traffic light: {tl_id}")
                    
                except Exception as load_error:
                    print(f"Failed to load model for {tl_id}: {load_error}")
                    # Try the old method as fallback
                    try:
                        # Extract episode number for loading
                        if episode_info == "best":
                            load_episode = 0  # Best model is typically saved as episode 0
                        elif episode_info == "final":
                            load_episode = "final"
                        elif episode_info == "latest":
                            load_episode = 0
                        elif isinstance(episode_info, int):
                            load_episode = episode_info
                        else:
                            load_episode = 0
                            
                        print(f"Fallback: Loading model with episode parameter: {load_episode}")
                        dqn_simulation.load_model(episode=load_episode)
                        break  # Exit the loop if fallback succeeds
                    except Exception as fallback_error:
                        print(f"Fallback method also failed: {fallback_error}")
                        raise load_error
                        
        print(f"Successfully loaded DQN model: {os.path.basename(model_file_path)}")
        print("DQN simulation configured in TESTING MODE - no training will occur")
        
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        print(f"Tried to load: {model_file_path}")
        print("Skipping DQN simulation.")
        return
    
    # Set up SUMO
    set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
    
    # Run simulation with episode=1 (pure exploitation mode since model is loaded)
    start_time = time.time()
    dqn_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"DQN simulation completed in {total_time:.2f}s")
    
    # Save metrics and plots
    print("Saving DQN simulation metrics and plots...")
    dqn_simulation.save_plot(episode=1)
    dqn_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for next simulation
    dqn_simulation.reset_history()


def main():
    parser = argparse.ArgumentParser(description='Test traffic signal control with different simulation types')
    parser.add_argument('--config', '-c', 
                       default='config/testing_testngatu6x1.yaml',
                       help='Path to testing configuration file')
    parser.add_argument('--simulations', '-s', 
                       nargs='+', 
                       choices=['base', 'actuated', 'dqn', 'all'],
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
    
    print("\nStarting multi-simulation testing...")
    print(f"Running simulations: {', '.join(simulations_to_run)}")
    print(f"Configuration: {args.config}")
    print(f"Max steps: {config['max_steps']}")
    print(f"Traffic lights: {len(config['traffic_lights'])}")
    
    # Generate routes once for all simulations (low, medium, high demand levels)
    print("\n" + "="*50)
    print("GENERATING ROUTES FOR ALL DEMAND LEVELS")
    print("="*50)
    
    simulation_path = config["sumo_cfg_file"].split("/")[1]
    # demand_levels = ['low', 'medium', 'high']
    
    # for demand_level in demand_levels:
        # if args.config != "config/testing_testngatu6x1EastWestOverflow.yaml":
        #     print(f"\nGenerating routes for {demand_level} demand level...")
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
    
    # Run selected simulations
    overall_start = time.time()
    
    try:
        if 'base' in simulations_to_run:
            test_base_simulation(config, path)
            
        if 'actuated' in simulations_to_run:
            test_actuated_simulation(config, path)
            
        if 'dqn' in simulations_to_run:
            test_dqn_simulation(config, path, args.model_file)
            
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
        metrics = ["density_avg", "travel_time_avg", "outflow_avg", "queue_length_avg", "waiting_time_avg"]
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
    
    print("\n" + "="*50)
    print("TESTING COMPLETED")
    print("="*50)
    print(f"Total testing time: {overall_time:.2f}s")
    print(f"Results saved to: {path}")
    
    # Clean up
    try:
        traci.close()
    except:
        pass


if __name__ == "__main__":
    main()