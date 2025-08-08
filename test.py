"""
Multi-simulation testing script for traffic signal control.
Supports testing with Base (SUMO default), Actuated (queue-based), and DQN simulations.
"""
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
import libsumo as traci
import glob
import re


def find_best_model_file(model_folder_path):
    """
    Automatically find the best .pth model file in the specified folder.
    Looks for files with patterns like:
    - *_BEST_CURRENT.pth (highest priority - current best model)
    - *_BEST_ep*.pth (best model from specific episodes)
    - *_final.pth 
    - *_episode_*.pth (finds the highest episode number)
    
    Args:
        model_folder_path (str): Path to the model folder
        
    Returns:
        tuple: (best_model_path, episode_number) or (None, None) if no model found
    """
    if not os.path.exists(model_folder_path):
        return None, None
    
    # Priority 1: Look for *_BEST_CURRENT.pth files (current best model)
    current_best_files = glob.glob(os.path.join(model_folder_path, "*_BEST_CURRENT.pth"))
    if current_best_files:
        # If multiple current best files, take the most recent one
        best_file = max(current_best_files, key=os.path.getmtime)
        print(f"Found current best model: {os.path.basename(best_file)}")
        return best_file, "best"
    
    # Priority 2: Look for *_BEST_ep*.pth files (episode-specific best models)
    best_episode_files = glob.glob(os.path.join(model_folder_path, "*_BEST_ep*.pth"))
    if best_episode_files:
        # Extract episode numbers and find the highest one
        episode_numbers = []
        for file in best_episode_files:
            match = re.search(r'_BEST_ep(\d+)', os.path.basename(file))
            if match:
                episode_numbers.append((int(match.group(1)), file))
        
        if episode_numbers:
            # Sort by episode number and get the highest
            episode_numbers.sort(key=lambda x: x[0], reverse=True)
            highest_episode, best_file = episode_numbers[0]
            print(f"Found best episode model: {os.path.basename(best_file)} (episode {highest_episode})")
            return best_file, "best"
    
    # Priority 3: Look for *_final.pth files
    final_files = glob.glob(os.path.join(model_folder_path, "*_final.pth"))
    if final_files:
        final_file = max(final_files, key=os.path.getmtime)
        print(f"Found final model: {os.path.basename(final_file)}")
        return final_file, "final"
    
    # Priority 4: Look for episode-specific files and find the highest episode
    episode_files = glob.glob(os.path.join(model_folder_path, "*_episode_*.pth"))
    if episode_files:
        # Extract episode numbers and find the highest one
        episode_numbers = []
        for file in episode_files:
            match = re.search(r'episode_(\d+)', os.path.basename(file))
            if match:
                episode_numbers.append((int(match.group(1)), file))
        
        if episode_numbers:
            # Sort by episode number and get the highest
            episode_numbers.sort(key=lambda x: x[0], reverse=True)
            highest_episode, best_file = episode_numbers[0]
            print(f"Found highest episode model: {os.path.basename(best_file)} (episode {highest_episode})")
            return best_file, highest_episode
    
    # Priority 5: Look for any .pth files
    pth_files = glob.glob(os.path.join(model_folder_path, "*.pth"))
    if pth_files:
        # Take the most recent .pth file
        latest_file = max(pth_files, key=os.path.getmtime)
        print(f"Found latest .pth model: {os.path.basename(latest_file)}")
        return latest_file, "latest"
    
    return None, None


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


def test_dqn_simulation(config, path):
    """Test with DQN agent (requires pre-trained model)"""
    print("\n" + "="*50)
    print("TESTING DQN SIMULATION")
    print("="*50)
    
    # Get model folder from config
    model_folder = config['model_folder']  # Default to model_13
    model_folder_path = os.path.join("models", model_folder)
    
    print(f"Looking for DQN models in: {model_folder_path}")
    
    # Auto-find the best model file
    model_file_path, episode_info = find_best_model_file(model_folder_path)
    
    if model_file_path is None:
        print(f"Warning: No DQN model files found in {model_folder_path}")
        print("Available model folders:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    pth_files = glob.glob(os.path.join(item_path, "*.pth"))
                    print(f"  - {item}: {len(pth_files)} .pth files")
        print("Skipping DQN simulation. Train a model first or check the model_folder in config.")
        return
    
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
        model_state = torch.load(model_file_path, map_location=dqn_simulation.device)
        
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
    
    # Run simulation with epsilon=0 (no exploration, pure exploitation)
    start_time = time.time()
    dqn_simulation.run(epsilon=0.0, episode=1)
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
                       help='Model folder to use for DQN simulation (e.g., model_11, model_12). Overrides config setting.')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = import_test_configuration(args.config)
    
    # Override GUI setting if specified
    if args.gui:
        config["gui"] = True
        
    # Override model folder if specified via command line
    if args.model_folder:
        config["model_folder"] = args.model_folder
        print(f"Using model folder from command line: {args.model_folder}")
    elif "model_folder" not in config:
        # Set default if not in config
        config["model_folder"] = "model_13"
        print(f"Using default model folder: model_13")
    else:
        print(f"Using model folder from config: {config['model_folder']}")
    
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
    demand_levels = ['low', 'medium', 'high']
    
    for demand_level in demand_levels:
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
    
        # Run selected simulations
        overall_start = time.time()
        
        try:
            if 'base' in simulations_to_run:
                test_base_simulation(config, path)
                
            if 'actuated' in simulations_to_run:
                test_actuated_simulation(config, path)
                
            if 'dqn' in simulations_to_run:
                test_dqn_simulation(config, path)
                
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