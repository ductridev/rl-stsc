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
import traci
import torch
import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from src.memory import ReplayMemory
from src.memory_palace import MemoryPalace
from src.utils import (
    set_train_path,
    set_sumo,
    import_train_configuration,
    set_load_model_path,
)
from src.model import DQN
from src.simulation import Simulation
from src.Qlearning import QSimulation
from src.actuated_simulation import ActuatedSimulation
from src.base_simulation import SimulationBase
from src.visualization import Visualization
from src.intersection import Intersection
from src.accident_manager import AccidentManager
from src.comparison_utils import SimulationComparison
from src.scripts.random_demand_sides import (
    generate_and_save_random_intervals
)
import datetime
import shutil
import os
import glob


def find_latest_trained_model(shared_path):
    """
    Find the latest trained model from the training session.
    
    Args:
        shared_path: Path where models are saved
        
    Returns:
        str: Path to latest model file (.pt) or None if not found
    """
    if not shared_path or not os.path.exists(shared_path):
        print(f"Shared path does not exist: {shared_path}")
        return None
    
    # Look for model files in the shared path
    model_patterns = [
        "skrl_model_*_GLOBAL_BEST.pt",
        "skrl_model_*_CURRENT_BEST.pt", 
        "skrl_model_*_episode_*_BEST.pt",
        "skrl_model_*.pt"
    ]
    
    latest_model = None
    latest_time = 0
    
    for pattern in model_patterns:
        model_files = glob.glob(os.path.join(shared_path, pattern))
        for model_file in model_files:
            file_time = os.path.getmtime(model_file)
            if file_time > latest_time:
                latest_time = file_time
                latest_model = model_file
    
    if latest_model:
        print(f"Found latest model: {os.path.basename(latest_model)}")
        # Return the full path to the .pt file for direct loading
        return latest_model
    else:
        print("No model files found in shared path")
        return None


def test_dqn_simulation_standalone(config, path, model_file_path):
    """
    Standalone DQN simulation test function.
    
    Args:
        config: Test configuration dictionary
        path: Path for saving test results
        model_file_path: Path to the model file to test
        
    Returns:
        dict: Test results and metrics
    """
    print(f"Testing DQN simulation with model: {os.path.basename(model_file_path)}")
    
    # Initialize visualization
    visualization = Visualization(path=path, dpi=100)
    
    # Initialize accident manager if accident configuration exists
    accident_manager = None
    if 'accident' in config:
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']],
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
        model_file_path=model_file_path
    )
    
    # Set up SUMO (disable GUI for automated testing)
    set_sumo(False, config["sumo_cfg_file"], config["max_steps"])
    
    # Run simulation
    start_time = time.time()
    dqn_simulation.run(episode=1)
    total_time = time.time() - start_time
    
    print(f"DQN simulation completed in {total_time:.2f}s")
    
    # Collect results
    completion_data = None
    if hasattr(dqn_simulation, 'completion_tracker'):
        completion_data = {
            'completed_count': dqn_simulation.completion_tracker.get_completed_count(),
            'avg_travel_time': dqn_simulation.completion_tracker.get_average_total_travel_time(),
            'total_travel_time': dqn_simulation.completion_tracker.get_total_travel_time()
        }
    
    # Collect metrics data
    metrics_data = {}
    if hasattr(dqn_simulation, 'history'):
        for metric_name, metric_data in dqn_simulation.history.items():
            if metric_data:
                # Calculate average across all traffic lights
                all_values = []
                for tl_values in metric_data.values():
                    all_values.extend(tl_values)
                if all_values:
                    metrics_data[f'avg_{metric_name}'] = sum(all_values) / len(all_values)
                    metrics_data[f'total_{metric_name}'] = sum(all_values)
    
    # Save metrics and plots
    print("Saving test simulation metrics and plots...")
    dqn_simulation.save_plot(episode=1)
    dqn_simulation.save_metrics_to_dataframe(episode=1)
    
    # Reset for cleanup
    dqn_simulation.reset_history()
    if hasattr(dqn_simulation, 'completion_tracker'):
        dqn_simulation.completion_tracker.reset()
    
    # Combine results
    result = {
        'simulation_time': total_time,
        'completion_data': completion_data,
        'metrics_data': metrics_data,
        'model_file': os.path.basename(model_file_path)
    }
    
    return result


def save_test_results_summary(test_results, test_results_path):
    """
    Save test results summary to files.
    
    Args:
        test_results: Dictionary of test results by config file
        test_results_path: Path to save results
    """
    import json
    import pandas as pd
    
    # Save JSON summary
    summary_file = os.path.join(test_results_path, "test_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"Test results summary saved: {summary_file}")
    
    # Create CSV summary for easier analysis
    csv_data = []
    for config_file, result in test_results.items():
        if 'error' in result:
            csv_data.append({
                'config_file': config_file,
                'status': 'ERROR',
                'error': result['error']
            })
        else:
            row = {
                'config_file': config_file,
                'status': 'SUCCESS',
                'simulation_time': result.get('simulation_time', 0),
                'model_file': result.get('model_file', ''),
            }
            
            # Add completion data
            if result.get('completion_data'):
                row.update({
                    'completed_count': result['completion_data'].get('completed_count', 0),
                    'avg_travel_time': result['completion_data'].get('avg_travel_time', 0),
                    'total_travel_time': result['completion_data'].get('total_travel_time', 0),
                })
            
            # Add metrics data
            if result.get('metrics_data'):
                for metric, value in result['metrics_data'].items():
                    row[metric] = value
            
            csv_data.append(row)
    
    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(test_results_path, "test_results_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"Test results CSV saved: {csv_file}")
        
        # Print summary
        print(f"\nTEST RESULTS SUMMARY:")
        print("="*60)
        for _, row in df.iterrows():
            config_name = os.path.basename(row['config_file']).replace('.yaml', '')
            if row['status'] == 'SUCCESS':
                print(f"{config_name}: SUCCESS")
                if 'completed_count' in row:
                    print(f"  Completed vehicles: {row['completed_count']}")
                if 'avg_travel_time' in row:
                    print(f"  Avg travel time: {row['avg_travel_time']:.2f}s")
                if 'simulation_time' in row:
                    print(f"  Simulation time: {row['simulation_time']:.2f}s")
            else:
                print(f"{config_name}: ERROR - {row.get('error', 'Unknown error')}")
        print("="*60)


def parse_arguments():
    """
    Parse command line arguments for checkpoint resuming.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='DQN Training with Checkpoint Resume Support')
    
    # Checkpoint/Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file or directory to resume training from')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pretrained model file (.pt) to load')
    parser.add_argument('--load-best', action='store_true', default=False,
                       help='Load the best available model from models directory')
    parser.add_argument('--start-episode', type=int, default=1,
                       help='Episode number to start from (default: 1)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Directory containing models to search for checkpoints')
    
    # Training configuration
    parser.add_argument('--config', type=str, nargs='+', 
                       default=["config/training_testngatu6x1.yaml"],
                       help='Configuration file(s) to use for training')
    
    return parser.parse_args()


def find_latest_checkpoint(model_dir):
    """
    Find the latest checkpoint in a model directory.
    
    Args:
        model_dir: Directory to search for checkpoints
        
    Returns:
        tuple: (checkpoint_path, episode_number) or (None, None) if not found
    """
    if not os.path.exists(model_dir):
        return None, None
    
    # Look for checkpoint files
    checkpoint_patterns = [
        "**/*_BEST_checkpoint.pt",
        "**/*_checkpoint.pt", 
        "**/*_CURRENT_BEST_checkpoint.pt",
        "**/*_GLOBAL_BEST_checkpoint.pt",
    ]
    
    latest_checkpoint = None
    latest_episode = -1
    
    for pattern in checkpoint_patterns:
        checkpoint_files = glob.glob(os.path.join(model_dir, pattern), recursive=True)
        
        for checkpoint_file in checkpoint_files:
            try:
                # Try to extract episode number from filename
                basename = os.path.basename(checkpoint_file)
                if "_episode_" in basename:
                    episode_str = basename.split("_episode_")[1].split("_")[0]
                    episode_num = int(episode_str)
                    
                    if episode_num > latest_episode:
                        latest_episode = episode_num
                        latest_checkpoint = checkpoint_file
                elif "BEST" in basename or "CURRENT" in basename:
                    # For BEST checkpoints, load the checkpoint to get episode info
                    try:
                        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                        episode_num = checkpoint_data.get('global_episode', checkpoint_data.get('episode', 0))
                        
                        if episode_num > latest_episode:
                            latest_episode = episode_num
                            latest_checkpoint = checkpoint_file
                    except:
                        continue
                        
            except (ValueError, IndexError):
                continue
    
    return latest_checkpoint, latest_episode if latest_episode > -1 else None


def find_best_model(model_dir):
    """
    Find the best available model in a directory.
    
    Args:
        model_dir: Directory to search for models
        
    Returns:
        dict: Dictionary with model paths for each traffic light or empty dict
    """
    if not os.path.exists(model_dir):
        return {}
    
    best_models = {}
    
    # Priority order for model types
    model_priorities = [
        "*_GLOBAL_BEST.pt",
        "*_CURRENT_BEST.pt", 
        "*_episode_*_BEST.pt",
        "*_BEST.pt",
    ]
    
    # Find all traffic light IDs
    all_model_files = glob.glob(os.path.join(model_dir, "**/*.pt"), recursive=True)
    tl_ids = set()
    
    for model_file in all_model_files:
        basename = os.path.basename(model_file)
        # Extract traffic light ID from filename
        if "skrl_model_" in basename:
            parts = basename.split("skrl_model_")[1].split("_")
            if parts:
                tl_ids.add(parts[0])
    
    # Find best model for each traffic light
    for tl_id in tl_ids:
        for priority_pattern in model_priorities:
            pattern = f"skrl_model_{tl_id}_{priority_pattern}"
            matches = glob.glob(os.path.join(model_dir, "**", pattern), recursive=True)
            
            if matches:
                # If multiple matches, pick the one with highest episode number
                if "_episode_" in priority_pattern:
                    matches.sort(key=lambda x: int(x.split("_episode_")[1].split("_")[0]), reverse=True)
                
                best_models[tl_id] = matches[0]
                break
    
    return best_models


def load_checkpoint_data(checkpoint_path):
    """
    Load checkpoint data and extract training metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Checkpoint data with metadata
    """
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract metadata
        metadata = {
            'checkpoint_path': checkpoint_path,
            'global_episode': checkpoint_data.get('global_episode', checkpoint_data.get('episode', 1)),
            'config_episode': checkpoint_data.get('config_episode', 1),
            'reward': checkpoint_data.get('reward', 0),
            'completion_rate': checkpoint_data.get('completion_rate', 0),
            'avg_outflow': checkpoint_data.get('avg_outflow', 0),
            'config_file': checkpoint_data.get('config_file', None),
            'metrics_won': checkpoint_data.get('metrics_won', 0),
            'has_model_state': 'model_state_dict' in checkpoint_data,
            'has_target_state': 'target_model_state_dict' in checkpoint_data,
            'has_optimizer_state': 'optimizer_state_dict' in checkpoint_data,
        }
        
        print(f"\nCheckpoint Metadata:")
        print(f"  Path: {os.path.basename(checkpoint_path)}")
        print(f"  Global Episode: {metadata['global_episode']}")
        print(f"  Config Episode: {metadata['config_episode']}")
        print(f"  Reward: {metadata['reward']:.2f}")
        print(f"  Completion Rate: {metadata['completion_rate']:.1f}%")
        print(f"  Avg Outflow: {metadata['avg_outflow']:.2f}")
        print(f"  Config File: {metadata['config_file'] or 'Unknown'}")
        print(f"  Model State: {'✓' if metadata['has_model_state'] else '✗'}")
        print(f"  Target State: {'✓' if metadata['has_target_state'] else '✗'}")
        print(f"  Optimizer State: {'✓' if metadata['has_optimizer_state'] else '✗'}")
        
        return checkpoint_data, metadata
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None


def resume_from_checkpoint(simulation_skrl, checkpoint_path_or_dir, args):
    """
    Resume training from a checkpoint file or directory.
    
    Args:
        simulation_skrl: DQN simulation object
        checkpoint_path_or_dir: Path to checkpoint file or directory
        args: Command line arguments
        
    Returns:
        tuple: (start_episode, metadata) or (None, None) if failed
    """
    print(f"\nATTEMPTING TO RESUME FROM: {checkpoint_path_or_dir}")
    print("=" * 60)
    
    checkpoint_path = None
    metadata = None
    
    # Determine if it's a file or directory
    if os.path.isfile(checkpoint_path_or_dir):
        if checkpoint_path_or_dir.endswith('.pt'):
            checkpoint_path = checkpoint_path_or_dir
        else:
            print(f"Invalid checkpoint file format: {checkpoint_path_or_dir}")
            return None, None
            
    elif os.path.isdir(checkpoint_path_or_dir):
        # Find latest checkpoint in directory
        checkpoint_path, latest_episode = find_latest_checkpoint(checkpoint_path_or_dir)
        
        if checkpoint_path is None:
            print(f"No checkpoint files found in directory: {checkpoint_path_or_dir}")
            return None, None
            
        print(f"Found latest checkpoint: {os.path.basename(checkpoint_path)} (Episode {latest_episode})")
    else:
        print(f"Checkpoint path does not exist: {checkpoint_path_or_dir}")
        return None, None
    
    # Load checkpoint data
    checkpoint_data, metadata = load_checkpoint_data(checkpoint_path)
    if checkpoint_data is None:
        return None, None
    
    # Load models from checkpoint
    loaded_agents = 0
    for tl_id, agent in simulation_skrl.agent_manager.agents.items():
        try:
            # Load main model
            if metadata['has_model_state']:
                agent.models["q_network"].load_state_dict(checkpoint_data['model_state_dict'])
                print(f"✓ Loaded main model for {tl_id}")
            
            # Load target model
            if metadata['has_target_state']:
                agent.models["target_q_network"].load_state_dict(checkpoint_data['target_model_state_dict'])
                print(f"✓ Loaded target model for {tl_id}")
            elif metadata['has_model_state']:
                # Use main model state for target if target not available
                agent.models["target_q_network"].load_state_dict(checkpoint_data['model_state_dict'])
                print(f"✓ Copied main model to target for {tl_id}")
            
            # Load optimizer state if available
            if metadata['has_optimizer_state'] and hasattr(agent.models["q_network"], 'optimizer'):
                try:
                    agent.models["q_network"].optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    print(f"✓ Loaded optimizer state for {tl_id}")
                except:
                    print(f"⚠ Could not load optimizer state for {tl_id} (continuing anyway)")
            
            loaded_agents += 1
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint for {tl_id}: {e}")
    
    if loaded_agents > 0:
        start_episode = metadata['global_episode'] + 1
        print(f"\n✓ CHECKPOINT LOADED SUCCESSFULLY")
        print(f"  Loaded models for {loaded_agents}/{len(simulation_skrl.agent_manager.agents)} agents")
        print(f"  Resuming from episode {start_episode}")
        print("=" * 60)
        
        return start_episode, metadata
    else:
        print(f"\n✗ FAILED TO LOAD CHECKPOINT")
        print(f"  Could not load models for any agents")
        print("=" * 60)
        return None, None


def load_pretrained_models(simulation_skrl, model_path_or_dir, args):
    """
    Load pretrained models without checkpoint metadata.
    
    Args:
        simulation_skrl: DQN simulation object  
        model_path_or_dir: Path to model file or directory
        args: Command line arguments
        
    Returns:
        bool: True if models loaded successfully
    """
    print(f"\nLOADING PRETRAINED MODELS FROM: {model_path_or_dir}")
    print("=" * 60)
    
    loaded_agents = 0
    
    if os.path.isfile(model_path_or_dir):
        # Single model file - load for all agents
        if not model_path_or_dir.endswith('.pt'):
            print(f"Invalid model file format: {model_path_or_dir}")
            return False
            
        try:
            state_dict = torch.load(model_path_or_dir, map_location='cpu')
            
            for tl_id, agent in simulation_skrl.agent_manager.agents.items():
                agent.models["q_network"].load_state_dict(state_dict)
                agent.models["target_q_network"].load_state_dict(state_dict)
                loaded_agents += 1
                print(f"✓ Loaded model for {tl_id}")
                
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
            
    elif os.path.isdir(model_path_or_dir):
        # Directory with multiple models
        if args.load_best:
            # Find best models for each traffic light
            best_models = find_best_model(model_path_or_dir)
            
            if not best_models:
                print(f"No suitable models found in directory: {model_path_or_dir}")
                return False
            
            for tl_id, agent in simulation_skrl.agent_manager.agents.items():
                if tl_id in best_models:
                    try:
                        state_dict = torch.load(best_models[tl_id], map_location='cpu')
                        agent.models["q_network"].load_state_dict(state_dict)
                        agent.models["target_q_network"].load_state_dict(state_dict)
                        loaded_agents += 1
                        print(f"✓ Loaded best model for {tl_id}: {os.path.basename(best_models[tl_id])}")
                    except Exception as e:
                        print(f"✗ Failed to load model for {tl_id}: {e}")
                else:
                    print(f"⚠ No model found for {tl_id}")
        else:
            # Use the agent manager's load function
            try:
                loaded_count = simulation_skrl.agent_manager.load_models(model_path_or_dir)
                loaded_agents = loaded_count
            except Exception as e:
                print(f"✗ Failed to load models: {e}")
                return False
                
    else:
        print(f"Model path does not exist: {model_path_or_dir}")
        return False
    
    if loaded_agents > 0:
        print(f"\n✓ PRETRAINED MODELS LOADED SUCCESSFULLY")
        print(f"  Loaded models for {loaded_agents}/{len(simulation_skrl.agent_manager.agents)} agents")
        print("=" * 60)
        return True
    else:
        print(f"\n✗ FAILED TO LOAD PRETRAINED MODELS")
        print("=" * 60)
        return False

def generate_routes_if_needed(config_file, config, demand = "high", global_episode=1):
    """
    Generate routes for simulation if needed based on configuration.
    
    Args:
        config_file: Path to configuration file
        config: Configuration dictionary
    """
    if config_file != "config/training_testngatu6x1EastWestOverflow.yaml":
        print("Generating routes...")
        if global_episode % 12 == 0:
            generate_and_save_random_intervals(
                sumo_cfg_file=config["sumo_cfg_file"],
                total_duration=7200,
                min_interval=3600,
                max_interval=3600,
                base_weight=0.0,
                high_min=100,
                high_max=400,
                min_active_sides=1,
                max_active_sides=1,
                edge_groups=config["edge_groups"],
            )
        else:
            generate_and_save_random_intervals(
                sumo_cfg_file=config["sumo_cfg_file"],
                total_duration=3600,
                min_interval=360,
                max_interval=360,
                base_weight=0.0,
                high_min=100,
                high_max=500,
                min_active_sides=1,
                max_active_sides=4,
                edge_groups=config["edge_groups"],
            )

        Intersection.generate_residential_demand_routes(
            config,
            config["sumo_cfg_file"].split("/")[1],
            demand_level=demand,
            enable_motorcycle=True,
            enable_passenger=True,
            enable_truck=True,
            enable_bicycle=True,
            enable_pedestrian=True,
        )
        print("Routes generated")


def extract_simulation_metrics(simulation, label):
    """
    Extract performance metrics from a simulation.
    
    Args:
        simulation: Simulation object
        label: Label for the simulation type
        
    Returns:
        dict: Dictionary containing extracted metrics
    """
    # Get vehicle statistics
    stats = simulation.vehicle_tracker.get_current_stats()
    completion_rate = (stats['total_arrived'] / max(stats['total_departed'], 1)) * 100
    
    # Extract metrics from simulation history
    avg_outflow, total_reward = 0, 0
    if hasattr(simulation, 'history'):
        if 'outflow' in simulation.history:
            outflow_data = []
            for tl_id, outflows in simulation.history['outflow'].items():
                if outflows:
                    outflow_data.append(sum(outflows) / len(outflows))
            avg_outflow = sum(outflow_data) / len(outflow_data) if outflow_data else 0
        
        if 'reward' in simulation.history:
            for tl_id, rewards in simulation.history['reward'].items():
                if rewards:
                    total_reward += sum(rewards)
    
    print(f"{label} Results: Completion={completion_rate:.1f}%, Reward={total_reward:.1f}, Outflow={avg_outflow:.2f}")
    
    return {
        'completion_rate': completion_rate,
        'total_reward': total_reward,
        'avg_outflow': avg_outflow,
        'total_arrived': stats['total_arrived'],
        'total_departed': stats['total_departed']
    }


def run_baseline_simulations(config_files, shared_path, shared_visualization, phase="before_training"):
    """
    Run baseline simulations (SimulationBase and ActuatedSimulation) for all configurations.
    
    Args:
        config_files: List of configuration file paths
        shared_path: Path for saving results
        shared_visualization: Visualization object
        phase: Phase of training ("before_training" or "after_training")
        
    Returns:
        dict: Dictionary containing baseline results for all configurations
    """
    print(f"\nRUNNING BASELINE SIMULATIONS {phase.upper().replace('_', ' ')}")
    print("=" * 80)
    
    baseline_results = {}
    
    for config_idx, config_file in enumerate(config_files):
        print(f"\n--- Baseline for Config {config_idx + 1}: {config_file} ---")
        config = import_train_configuration(config_file)
        
        # Initialize baseline components
        baseline_path = set_train_path(config["models_path_name"]) if config_idx == 0 and shared_path is None else shared_path
        baseline_visualization = shared_visualization if shared_visualization else Visualization(path=baseline_path, dpi=100)
        
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
        
        # Initialize SimulationBase
        simulation_base = SimulationBase(
            agent_cfg=config["agent"],
            max_steps=config["max_steps"],
            traffic_lights=config["traffic_lights"],
            accident_manager=accident_manager,
            visualization=baseline_visualization,
            epoch=config["training_epochs"],
            path=baseline_path,
            save_interval=10,
        )
        
        # Initialize ActuatedSimulation
        simulation_actuated = ActuatedSimulation(
            agent_cfg=config["agent"],
            max_steps=config["max_steps"],
            traffic_lights=config["traffic_lights"],
            accident_manager=accident_manager,
            visualization=baseline_visualization,
            epoch=config["training_epochs"],
            path=baseline_path,
            save_interval=10,
        )
        
        # Run SimulationBase
        print("Running SimulationBase (static baseline)...")
        set_sumo(False, config["sumo_cfg_file"], config["max_steps"])
        simulation_time_base = simulation_base.run(1)
        base_metrics = extract_simulation_metrics(simulation_base, "Base")
        base_metrics['simulation_time'] = simulation_time_base
        
        # Reset base simulation
        simulation_base.vehicle_tracker.reset()
        simulation_base.reset_history()
        
        # Run ActuatedSimulation
        print("Running ActuatedSimulation (queue-based baseline)...")
        set_sumo(False, config["sumo_cfg_file"], config["max_steps"])
        simulation_time_actuated = simulation_actuated.run(1)
        actuated_metrics = extract_simulation_metrics(simulation_actuated, "Actuated")
        actuated_metrics['simulation_time'] = simulation_time_actuated
        
        # Reset actuated simulation
        simulation_actuated.vehicle_tracker.reset()
        simulation_actuated.reset_history()
        
        # Store baseline results for this configuration
        config_name = config_file.split('/')[-1].replace('.yaml', '')
        baseline_results[config_name] = {
            'config_file': config_file,
            'base': base_metrics,
            'actuated': actuated_metrics
        }
        
        print(f"Baseline results stored for {config_name}")
    
    print(f"\nBASELINE SIMULATIONS COMPLETED - Results stored for comparison")
    print("=" * 80)
    
    return baseline_results


def load_checkpoint_from_config(simulation_skrl, config):
    """
    Load checkpoint or model specified in configuration file.
    
    Args:
        simulation_skrl: DQN simulation object
        config: Configuration dictionary
        
    Returns:
        tuple: (success, start_episode, metadata)
    """
    print(config.get("load_model_name"))
    load_model_name = config.get("load_model_name")
    
    if load_model_name is None or load_model_name == "null":
        return False, None, None
    
    print(f"\nLOADING MODEL FROM CONFIG: {load_model_name}")
    print("=" * 60)
    
    # Check if it's a checkpoint file (contains 'checkpoint' in name)
    if 'checkpoint' in load_model_name.lower():
        # Handle as checkpoint
        if os.path.isfile(load_model_name):
            checkpoint_data, metadata = load_checkpoint_data(load_model_name)
            if checkpoint_data is None:
                print(f"Failed to load checkpoint from config: {load_model_name}")
                return False, None, None
            
            # Load models from checkpoint
            loaded_agents = 0
            for tl_id, agent in simulation_skrl.agent_manager.agents.items():
                try:
                    # Load main model
                    if metadata['has_model_state']:
                        agent.models["q_network"].load_state_dict(checkpoint_data['model_state_dict'])
                        print(f"✓ Loaded main model for {tl_id}")
                    
                    # Load target model
                    if metadata['has_target_state']:
                        agent.models["target_q_network"].load_state_dict(checkpoint_data['target_model_state_dict'])
                        print(f"✓ Loaded target model for {tl_id}")
                    elif metadata['has_model_state']:
                        agent.models["target_q_network"].load_state_dict(checkpoint_data['model_state_dict'])
                        print(f"✓ Copied main model to target for {tl_id}")
                    
                    # Load optimizer state if available
                    if metadata['has_optimizer_state'] and hasattr(agent.models["q_network"], 'optimizer'):
                        try:
                            agent.models["q_network"].optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                            print(f"✓ Loaded optimizer state for {tl_id}")
                        except:
                            print(f"⚠ Could not load optimizer state for {tl_id}")
                    
                    loaded_agents += 1
                    
                except Exception as e:
                    print(f"✗ Failed to load checkpoint for {tl_id}: {e}")
            
            if loaded_agents > 0:
                start_episode = metadata['global_episode'] + 1
                print(f"\n✓ CONFIG CHECKPOINT LOADED SUCCESSFULLY")
                print(f"  Loaded models for {loaded_agents}/{len(simulation_skrl.agent_manager.agents)} agents")
                print(f"  Will resume from episode {start_episode}")
                print("=" * 60)
                return True, start_episode, metadata
            else:
                print(f"\n✗ FAILED TO LOAD CONFIG CHECKPOINT")
                print("=" * 60)
                return False, None, None
        else:
            print(f"Checkpoint file not found: {load_model_name}")
            return False, None, None
    else:
        # Handle as regular model file
        if os.path.isfile(load_model_name):
            try:
                state_dict = torch.load(load_model_name, map_location='cpu')
                
                loaded_agents = 0
                for tl_id, agent in simulation_skrl.agent_manager.agents.items():
                    agent.models["q_network"].load_state_dict(state_dict)
                    agent.models["target_q_network"].load_state_dict(state_dict)
                    loaded_agents += 1
                    print(f"✓ Loaded model for {tl_id}")
                
                print(f"\n✓ CONFIG MODEL LOADED SUCCESSFULLY")
                print(f"  Loaded models for {loaded_agents}/{len(simulation_skrl.agent_manager.agents)} agents")
                print("=" * 60)
                return True, None, None
                
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                print("=" * 60)
                return False, None, None
        else:
            print(f"Model file not found: {load_model_name}")
            return False, None, None


def initialize_dqn_simulation(config, shared_simulation_skrl, shared_path, shared_visualization, 
                             config_idx):
    """
    Initialize or update DQN simulation components.
    
    Args:
        config: Configuration dictionary
        shared_simulation_skrl: Existing shared simulation (None for first config)
        shared_path: Shared path for saving results
        shared_visualization: Shared visualization object
        config_idx: Configuration index
        
    Returns:
        tuple: (simulation_skrl, shared_path, shared_visualization)
    """
    accident_manager = None
    # Initialize shared components only once
    if shared_simulation_skrl is None:
        # Set model save path (use first config's model path)
        shared_path = set_train_path(config["models_path_name"])
        shared_visualization = Visualization(path=shared_path, dpi=100)
        
        print(f"Initializing shared DQN model - will be used across all configurations")
        
        # Initialize accident manager for this config
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
        
        save_interval = config.get("save_interval", 10)
        
        # First time initialization
        shared_simulation_skrl = Simulation(
            visualization=shared_visualization,
            agent_cfg=config["agent"],
            max_steps=config["max_steps"],
            traffic_lights=config["traffic_lights"],
            accident_manager=accident_manager,
            interphase_duration=config["interphase_duration"],
            epoch=config["training_epochs"],
            path=shared_path,
            training_steps=config["training_steps"],
            updating_target_network_steps=config["updating_target_network_steps"],
            save_interval=save_interval,
            memory_size=(config['memory_size_min'], config['memory_size_max'])
        )
        
        print(f"Shared DQN model initialized with first config")
    else:
        # Update existing simulation with new config parameters
        accident_manager = AccidentManager(
            start_step=config['accident']['start_step'],
            duration=config['accident']['duration'],
            junction_id_list=[junction["id"] for junction in config['accident']['junction']],
            detection_id_list= [detector["id"] for detector in config['accident']['detectors']]
        )
        
        shared_simulation_skrl.accident_manager = accident_manager
        shared_simulation_skrl.max_steps = config["max_steps"]
        shared_simulation_skrl.traffic_lights = config["traffic_lights"]
        shared_simulation_skrl.interphase_duration = config["interphase_duration"]
        shared_simulation_skrl.epoch = config["training_epochs"]
        shared_simulation_skrl.training_steps = config["training_steps"]
        shared_simulation_skrl.updating_target_network_steps = config["updating_target_network_steps"]
        shared_simulation_skrl.save_interval = config.get("save_interval", 10)
        print(f"Shared DQN model updated with new config parameters")
    
    return shared_simulation_skrl, shared_path, shared_visualization


def train_dqn_episode(simulation_skrl, global_episode, config_episode, config_file, config, 
                     global_episode_tracker):
    """
    Run a single DQN training episode.
    
    Args:
        simulation_skrl: DQN simulation object
        global_episode: Global episode counter
        config_episode: Config-specific episode counter
        config_file: Configuration file path
        config: Configuration dictionary
        global_episode_tracker: Dictionary to track global episode info
        
    Returns:
        tuple: (performance_metrics)
    """
    print(f"\n----- Config Episode {config_episode} of {config['total_episodes']} (Global: {global_episode}) -----")
    
    print("Running SKRL DQN Simulation...")
    # Enable GUI once every 100 episodes for visual monitoring
    gui_enabled = (global_episode % 100 == 0 and global_episode > 0)
    set_sumo(gui_enabled, config["sumo_cfg_file"], config["max_steps"])
    if gui_enabled:
        print(f"GUI enabled for config episode {config_episode} (global {global_episode}) - Visual monitoring")
    
    simulation_time_skrl, training_time_skrl = simulation_skrl.run(global_episode)
    print(f"Simulation (SKRL DQN) time: {simulation_time_skrl}, Training time: {training_time_skrl}")
    
    # Extract performance metrics
    try:
        skrl_stats = simulation_skrl.vehicle_tracker.get_current_stats()
        completion_rate = (skrl_stats['total_arrived'] / max(skrl_stats['total_departed'], 1)) * 100
        
        # Get reward metrics from simulation history
        dqn_avg_reward = 0
        dqn_total_reward = 0
        if hasattr(simulation_skrl, 'history') and 'reward' in simulation_skrl.history:
            reward_data = []
            for tl_id, rewards in simulation_skrl.history['reward'].items():
                if rewards:
                    total_reward_tl = sum(rewards)
                    dqn_total_reward += total_reward_tl
                    avg_reward_tl = dqn_total_reward / len(rewards)
                    reward_data.append(avg_reward_tl)
            if reward_data:
                dqn_avg_reward = sum(reward_data) / len(reward_data)
        
        # Extract avg_outflow from DQN simulation history
        dqn_avg_outflow = 0
        if hasattr(simulation_skrl, 'history') and 'outflow' in simulation_skrl.history:
            outflow_data = []
            for tl_id, outflows in simulation_skrl.history['outflow'].items():
                if outflows:
                    avg_outflow_tl = sum(outflows) / len(outflows)
                    outflow_data.append(avg_outflow_tl)
            if outflow_data:
                dqn_avg_outflow = sum(outflow_data) / len(outflow_data)
        
        print("DQN Results:")
        print(f"  Avg Outflow: {dqn_avg_outflow:.2f}")
        print(f"  Total Reward: {dqn_total_reward:.2f}")
        print(f"  Avg Reward: {dqn_avg_reward:.2f}")
        print("  Note: Baseline comparisons will be shown after all training completes")
        
        performance_metrics = {
            'episode': global_episode,
            'config_episode': config_episode,
            'completion_rate': completion_rate,
            'total_reward': dqn_total_reward,
            'combined_score': dqn_total_reward,
            'total_departed': skrl_stats['total_departed'],
            'total_arrived': skrl_stats['total_arrived'],
            'dqn_avg_outflow': dqn_avg_outflow,
        }
        
    except Exception as e:
        print(f"Error tracking performance: {e}")
        performance_metrics = {
            'episode': global_episode,
            'config_episode': config_episode,
        }
    
    # Reset vehicle trackers after performance tracking
    print("  Resetting vehicle tracker for DQN simulation")
    simulation_skrl.vehicle_tracker.reset()
    
    # Reset simulation history after extracting metrics to prevent memory accumulation
    simulation_skrl.reset_history()
    print("  Note: History and buffers are now automatically cleared in simulation._finalize_episode()")
    
    return performance_metrics


def evaluate_and_save_best_model(performance_metrics, config_best_performance, global_best_performance,
                                config_file, simulation_skrl, path, global_episode, config_episode):
    """
    Evaluate current performance and save model if it's the best so far.
    
    Args:
        performance_metrics: Current episode performance metrics
        config_best_performance: Best performance for current config
        global_best_performance: Best performance across all configs
        config_file: Configuration file path
        simulation_skrl: DQN simulation object
        path: Path for saving models
        global_episode: Global episode counter
        config_episode: Config episode counter
        
    Returns:
        tuple: (updated_config_best_performance, updated_global_best_performance)
    """
    # Check if this is the best performance using 3-metric voting system
    metrics_won = 0
    metric_details = []
    
    current_reward = performance_metrics.get('total_reward', 0)
    current_completion = performance_metrics.get('completion_rate', 0)
    current_outflow = performance_metrics.get('dqn_avg_outflow', 0)
    
    # Metric 1: Total Reward (higher is better)
    if current_reward > config_best_performance['total_reward']:
        metrics_won += 1
        metric_details.append(f"Reward: {current_reward:.2f} > {config_best_performance['total_reward']:.2f}")
    else:
        metric_details.append(f" Reward: {current_reward:.2f} <= {config_best_performance['total_reward']:.2f}")
    
    # Metric 2: Completion Rate (higher is better)
    if current_completion > config_best_performance['completion_rate']:
        metrics_won += 1
        metric_details.append(f"Completion: {current_completion:.1f}% > {config_best_performance['completion_rate']:.1f}%")
    else:
        metric_details.append(f"Completion: {current_completion:.1f}% <= {config_best_performance['completion_rate']:.1f}%")
    
    # Metric 3: Average Outflow (higher is better)
    if current_outflow > config_best_performance.get('dqn_avg_outflow', 0):
        metrics_won += 1
        metric_details.append(f"Outflow: {current_outflow:.2f} > {config_best_performance.get('dqn_avg_outflow', 0):.2f}")
    else:
        metric_details.append(f"Outflow: {current_outflow:.2f} <= {config_best_performance.get('dqn_avg_outflow', 0):.2f}")
    
    print(f"\n3-Metric Performance Comparison - Config Ep {config_episode} (Global {global_episode}):")
    for detail in metric_details:
        print(f"   {detail}")
    print(f"   Result: Won {metrics_won}/3 metrics")
    
    # Save model if it wins majority of metrics (2 or more out of 3)
    if metrics_won >= 2:
        print(f"\nNEW BEST PERFORMANCE for {config_file} at Config Ep {config_episode} (Global {global_episode})!")
        print(f"   Won {metrics_won}/3 metrics - Majority Victory!")
        
        # Update best performance records
        config_best_performance = performance_metrics.copy()
        config_best_performance['config_file'] = config_file
        config_best_performance['config_episode'] = config_episode
        
        # Check if this is also the global best
        global_metrics_won = 0
        if current_reward > global_best_performance['total_reward']:
            global_metrics_won += 1
        if current_completion > global_best_performance['completion_rate']:
            global_metrics_won += 1
        if current_outflow > global_best_performance.get('dqn_avg_outflow', 0):
            global_metrics_won += 1
        
        if global_metrics_won >= 2:
            global_best_performance = performance_metrics.copy()
            global_best_performance['config_file'] = config_file
            global_best_performance['dqn_avg_outflow'] = current_outflow
            print(f"   This is also the GLOBAL BEST across all configurations! ({global_metrics_won}/3 metrics)")
        
        # Save BEST model files
        save_best_model_files(simulation_skrl, path, global_episode, config_episode, performance_metrics, 
                            config_file, metrics_won, global_metrics_won)
    
    # Print performance summary
    print(f"\nPerformance Summary - Config Ep {config_episode} (Global {global_episode})")
    print(f"Current DQN: Reward={current_reward:.2f}, Outflow={current_outflow:.2f}, Completion={current_completion:.1f}%")
    print(f"Config Best: Reward={config_best_performance['total_reward']:.2f}, Global Ep={config_best_performance['episode']}, Completion={config_best_performance['completion_rate']:.1f}%")
    print(f"             Outflow={config_best_performance.get('dqn_avg_outflow', 0):.2f} (Won {metrics_won}/3 metrics this episode)")
    print(f"Global Best: Reward={global_best_performance['total_reward']:.2f}, Global Ep={global_best_performance['episode']}, Config={global_best_performance.get('config_file', 'N/A')}")
    print(f"             Outflow={global_best_performance.get('dqn_avg_outflow', 0):.2f}, Completion={global_best_performance['completion_rate']:.1f}%")
    print("Note: Final baseline comparisons will be available after all training completes")
    
    return config_best_performance, global_best_performance


def save_best_model_files(simulation_skrl, path, global_episode, config_episode, performance_metrics, 
                         config_file, metrics_won, global_metrics_won):
    """
    Save best model files in various formats.
    
    Args:
        simulation_skrl: DQN simulation object
        path: Path for saving models
        global_episode: Global episode counter
        config_episode: Config episode counter
        performance_metrics: Performance metrics dictionary
        config_file: Configuration file path
        metrics_won: Number of metrics won
        global_metrics_won: Number of global metrics won
    """
    print(f"   Saving BEST models in .pt format (global episode {global_episode})...")
    
    for tl_id in simulation_skrl.agent_manager.agents.keys():
        agent = simulation_skrl.agent_manager.agents[tl_id]
        
        # Best model with GLOBAL episode number
        best_model_path = path + f"skrl_model_{tl_id}_episode_{global_episode}_BEST.pt"
        torch.save(agent.models["q_network"].state_dict(), best_model_path)
        print(f"     {os.path.basename(best_model_path)}")
        
        # Best checkpoint with GLOBAL episode number and metadata
        best_checkpoint_path = path + f"skrl_model_{tl_id}_episode_{global_episode}_BEST_checkpoint.pt"
        checkpoint = {
            "model_state_dict": agent.models["q_network"].state_dict(),
            "target_model_state_dict": agent.models["target_q_network"].state_dict(),
            "global_episode": global_episode,
            "config_episode": config_episode,
            "reward": performance_metrics.get('total_reward', 0),
            "completion_rate": performance_metrics.get('completion_rate', 0),
            "avg_outflow": performance_metrics.get('dqn_avg_outflow', 0),
            "metrics_won": metrics_won,
            "config_file": config_file
        }
        torch.save(checkpoint, best_checkpoint_path)
        print(f"     {os.path.basename(best_checkpoint_path)}")
        
        # Current best (overwrite previous)
        current_best_model = path + f"skrl_model_{tl_id}_CURRENT_BEST.pt"
        torch.save(agent.models["q_network"].state_dict(), current_best_model)
        
        current_best_checkpoint = path + f"skrl_model_{tl_id}_CURRENT_BEST_checkpoint.pt"
        torch.save(checkpoint, current_best_checkpoint)
        
        # Global best (overwrite previous) 
        if global_metrics_won >= 2:
            global_best_model = path + f"skrl_model_{tl_id}_GLOBAL_BEST.pt"
            torch.save(agent.models["q_network"].state_dict(), global_best_model)
            
            global_best_checkpoint = path + f"skrl_model_{tl_id}_GLOBAL_BEST_checkpoint.pt"
            torch.save(checkpoint, global_best_checkpoint)
    
    # Also save using the standard method for compatibility
    simulation_skrl.save_model(global_episode)
    print(f"   Standard SKRL models also saved for global episode {global_episode}")


def compare_baselines_with_dqn(baseline_results, all_config_results):
    """
    Compare baseline results with DQN training results.
    
    Args:
        baseline_results: Dictionary containing baseline results
        all_config_results: List of configuration results from DQN training
    """
    print("\nCOMPREHENSIVE BASELINE vs DQN COMPARISON")
    print("=" * 90)
    
    # Determine which baseline to use for comparison
    print("BASELINE SELECTION:")
    print("   Before Training: Initial baseline without any DQN influence")  
    print("   After Training: Final baseline (same conditions as DQN's last run)")
    print("   Selected: AFTER TRAINING (most fair comparison)")
    
    selected_baseline = baseline_results['after_training']
    
    # Detailed comparison for each configuration
    for config_idx, config in enumerate(all_config_results):
        config_name = config['config_name']
        
        if config_name not in selected_baseline:
            print(f"WARNING: No baseline data for {config_name}, skipping...")
            continue
        
        baseline_data = selected_baseline[config_name]
        
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'='*60}")
        
        # DQN Best Performance (from training)
        dqn_completion = config['best_completion_rate']
        dqn_reward = config['best_total_reward'] 
        dqn_outflow = config['best_dqn_avg_outflow']
        dqn_arrived = config['best_total_arrived']
        
        # Base and Actuated Simulation Results
        base_completion = baseline_data['base']['completion_rate']
        base_reward = baseline_data['base']['total_reward']
        base_outflow = baseline_data['base']['avg_outflow']
        base_arrived = baseline_data['base']['total_arrived']
        
        actuated_completion = baseline_data['actuated']['completion_rate']
        actuated_reward = baseline_data['actuated']['total_reward']
        actuated_outflow = baseline_data['actuated']['avg_outflow']
        actuated_arrived = baseline_data['actuated']['total_arrived']
        
        # Performance Comparison Table
        print("PERFORMANCE COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Metric':<20} {'DQN (Best)':<15} {'Base':<15} {'Actuated':<15} {'DQN vs Base':<15}")
        print("-" * 80)
        
        # Calculate and display metrics
        metrics = [
            ('Completion Rate %', dqn_completion, base_completion, actuated_completion),
            ('Total Reward', dqn_reward, base_reward, actuated_reward),
            ('Avg Outflow', dqn_outflow, base_outflow, actuated_outflow),
            ('Vehicles Arrived', dqn_arrived, base_arrived, actuated_arrived)
        ]
        
        dqn_vs_base_wins = 0
        dqn_vs_actuated_wins = 0
        
        for metric_name, dqn_val, base_val, actuated_val in metrics:
            base_diff = ((dqn_val - base_val) / base_val * 100) if base_val != 0 else 0
            print(f"{metric_name:<20} {dqn_val:<14.1f} {base_val:<14.1f} {actuated_val:<14.1f} {base_diff:+.1f}%")
            
            if dqn_val > base_val:
                dqn_vs_base_wins += 1
            if dqn_val > actuated_val:
                dqn_vs_actuated_wins += 1
        
        print("-" * 80)
        
        # Performance summary
        print("PERFORMANCE SUMMARY:")
        print(f"   DQN vs Base: Won {dqn_vs_base_wins}/4 metrics")
        print(f"   DQN vs Actuated: Won {dqn_vs_actuated_wins}/4 metrics")
        
        # Determine performance level
        if dqn_vs_base_wins >= 3:
            print("   DQN SIGNIFICANTLY OUTPERFORMED Base Simulation")
        elif dqn_vs_base_wins >= 2:
            print("   DQN MODERATELY OUTPERFORMED Base Simulation")
        else:
            print("   DQN did not outperform Base Simulation")
        
        if dqn_vs_actuated_wins >= 3:
            print("   DQN SIGNIFICANTLY OUTPERFORMED Actuated Simulation")
        elif dqn_vs_actuated_wins >= 2:
            print("   DQN MODERATELY OUTPERFORMED Actuated Simulation") 
        else:
            print("   DQN did not outperform Actuated Simulation")
            
        # Recommend best baseline for comparison
        if actuated_completion > base_completion:
            print("   RECOMMENDATION: Use Actuated as primary baseline (better performance)")
        else:
            print("   RECOMMENDATION: Use Base as primary baseline (better performance)")


def save_comprehensive_results(all_config_results, global_best_performance, shared_path, 
                             baseline_results):
    """
    Save comprehensive training results to files.
    
    Args:
        all_config_results: List of configuration results
        global_best_performance: Global best performance metrics
        shared_path: Path for saving results
        baseline_results: Baseline comparison results
    """
    # Save results summary to file
    results_summary_file = shared_path + "comprehensive_results_summary.txt"
    try:
        with open(results_summary_file, 'w') as f:
            f.write("COMPREHENSIVE TRAINING RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed: {datetime.datetime.now()}\n")
            f.write(f"Total configurations: {len(all_config_results)}\n")
            f.write(f"Model Selection: 3-Metric Voting System (>=2/3 metrics must improve)\n\n")
            
            f.write("GLOBAL BEST PERFORMANCE:\n")
            f.write(f"Configuration: {global_best_performance['config_file']}\n")
            f.write(f"Episode: {global_best_performance['episode']}\n")
            f.write(f"Total Reward: {global_best_performance['total_reward']:.2f}\n")
            f.write(f"Completion Rate: {global_best_performance['completion_rate']:.2f}%\n")
            f.write(f"Avg Outflow: {global_best_performance.get('dqn_avg_outflow', 'N/A')}\n")
            
            f.write("\nCONFIGURATION DETAILS:\n")
            f.write("-" * 30 + "\n")
            for config in all_config_results:
                f.write(f"{config['config_name']}: Episodes {config['final_config_episode']}, ")
                f.write(f"Best Reward {config['best_total_reward']:.1f}, ")
                f.write(f"Best Completion {config['best_completion_rate']:.1f}%\n")
        
        print(f"Comprehensive results summary saved: {results_summary_file}")
    except Exception as e:
        print(f"Could not save results summary file: {e}")
    
    # Create comprehensive performance comparison CSV
    try:
        comparison_df = pd.DataFrame([
            {
                'Config_Name': config['config_name'],
                'Config_File': config['config_file'],
                'Best_Global_Episode': config['best_episode'],
                'Final_Config_Episode': config['final_config_episode'],
                'Final_Global_Episode': config['final_global_episode'],
                'Best_Total_Reward': config['best_total_reward'],
                'Best_Completion_Rate': config['best_completion_rate'],
                'Best_Avg_Outflow': config['best_dqn_avg_outflow'],
                'Best_Total_Arrived': config['best_total_arrived'],
                'Best_Total_departed': config['best_total_departed'],
                'Final_Epsilon': config['final_epsilon'],
                'Is_Global_Best': config['config_file'] == global_best_performance['config_file'],
            } for config in all_config_results
        ])
        
        comparison_csv = shared_path + "all_configs_performance_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Performance comparison CSV saved: {comparison_csv}")
    except Exception as e:
        print(f"Could not save performance comparison CSV: {e}")


def main():
    """
    Main training function - orchestrates the entire training process.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("Starting Multi-Configuration DQN Training with Checkpoint Resume Support")
    print("=" * 80)
    
    # Display resume/load information
    if args.resume:
        print(f"RESUME MODE: Will attempt to resume from {args.resume}")
    elif args.load_model:
        print(f"PRETRAINED MODE: Will load pretrained model from {args.load_model}")
    elif args.load_best:
        model_dir = args.model_dir or "models/"
        print(f"BEST MODEL MODE: Will load best available models from {model_dir}")
    
    if args.start_episode > 1:
        print(f"CUSTOM START EPISODE: Starting from episode {args.start_episode}")
    
    print("=" * 80)
    
    # Initialize shared variables
    shared_simulation_skrl = None
    shared_visualization = None
    shared_path = None
    global_episode = args.start_episode  # Use command line start episode
    
    # Track results across all configurations
    all_config_results = []
    global_best_performance = {
        'episode': -1,
        'completion_rate': -1,
        'total_reward': -float('inf'),
        'combined_score': -float('inf'),
        'config_file': None,
        'dqn_avg_outflow': 0,
        'total_arrived': 0,
        'total_departed': 0
    }
    
    config_files = args.config  # Use config files from command line
    
    # Load all configurations and determine total episodes
    configs = []
    max_episodes_per_config = 0
    for config_file in config_files:
        config = import_train_configuration(config_file)
        configs.append((config_file, config))
        max_episodes_per_config = max(max_episodes_per_config, config["total_episodes"])
    
    print(f"\nALTERNATING CONFIGURATION TRAINING:")
    print(f"   Configurations: {len(config_files)}")
    print(f"   Max episodes per config: {max_episodes_per_config}")
    print(f"   Pattern: Episode 1→Config 1, Episode 2→Config 2, Episode 3→Config 3, Episode 4→Config 1, etc.")
    print(f"   Each config gets equal training opportunities distributed across episodes")
    print("=" * 80)
    
    # Run initial baseline simulations
    # baseline_results_before = run_baseline_simulations(config_files, shared_path, shared_visualization, "before_training")
    # baseline_results = {'before_training': baseline_results_before}
    
    # Initialize tracking for each configuration
    config_trackers = {}
    for i, (config_file, config) in enumerate(configs):
        config_name = config_file.split('/')[-1].replace('.yaml', '')
        config_trackers[i] = {
            'config_file': config_file,
            'config_name': config_name,
            'config': config,
            'config_episode': 1,
            'performance_history': [],
            'best_performance': {
                'episode': -1,
                'config_episode': -1,
                'completion_rate': -1,
                'total_reward': -float('inf'),
                'combined_score': -float('inf'),
                'config_file': config_file,
                'total_arrived': 0,
                'total_departed': 0,
                'dqn_avg_outflow': 0
            },
            'training_start_time': datetime.datetime.now(),
            'completed': False
        }
    
    # Initialize DQN simulation with first config
    first_config_file, first_config = configs[0]
    shared_simulation_skrl, shared_path, shared_visualization = initialize_dqn_simulation(
        first_config, shared_simulation_skrl, shared_path, shared_visualization, 0
    )
    
    # Handle checkpoint/model loading - Priority order:
    # 1. Command line arguments (--resume, --load-model, --load-best)
    # 2. Configuration file settings (load_model_name)
    resume_metadata = None
    config_loaded = False
    
    if args.resume:
        # Priority 1: Command line resume
        start_episode, resume_metadata = resume_from_checkpoint(shared_simulation_skrl, args.resume, args)
        if start_episode is not None:
            global_episode = start_episode
            config_loaded = True
            
            # Update config trackers based on resume metadata if available
            if resume_metadata and resume_metadata.get('config_file'):
                resumed_config = resume_metadata['config_file']
                print(f"Checkpoint was from config: {resumed_config}")
                
                # Try to adjust config episode counters
                for i, tracker in config_trackers.items():
                    if tracker['config_file'] == resumed_config:
                        config_episode_from_checkpoint = resume_metadata.get('config_episode', 1)
                        tracker['config_episode'] = config_episode_from_checkpoint + 1
                        print(f"Adjusted {tracker['config_name']} to start from config episode {tracker['config_episode']}")
                        break
        else:
            print("Failed to resume from checkpoint")
            
    elif args.load_model:
        # Priority 2: Command line model loading
        success = load_pretrained_models(shared_simulation_skrl, args.load_model, args)
        if success:
            config_loaded = True
        else:
            print("Failed to load pretrained model")
            
    elif args.load_best:
        # Priority 3: Command line best model loading
        model_dir = args.model_dir or shared_path or "models/"
        success = load_pretrained_models(shared_simulation_skrl, model_dir, args)
        if success:
            config_loaded = True
        else:
            print("Failed to load best models")
    
    # Priority 4: Check configuration file for load_model_name
    if not config_loaded:
        config_success, config_start_episode, config_metadata = load_checkpoint_from_config(
            shared_simulation_skrl, first_config
        )
        
        if config_success:
            config_loaded = True
            if config_start_episode is not None:
                # It was a checkpoint, adjust episode counter
                global_episode = config_start_episode
                resume_metadata = config_metadata
                
                # Update config trackers based on config metadata
                if config_metadata and config_metadata.get('config_file'):
                    resumed_config = config_metadata['config_file']
                    print(f"Config checkpoint was from: {resumed_config}")
                    
                    for i, tracker in config_trackers.items():
                        if tracker['config_file'] == resumed_config:
                            config_episode_from_checkpoint = config_metadata.get('config_episode', 1)
                            tracker['config_episode'] = config_episode_from_checkpoint + 1
                            print(f"Adjusted {tracker['config_name']} to start from config episode {tracker['config_episode']}")
                            break
            else:
                # It was a regular model file
                print("Loaded pretrained model from configuration")
    
    # If nothing was loaded, show fresh start message
    if not config_loaded:
        print("\nNo checkpoint or model specified - starting fresh training")
    
    print("\nSTARTING ALTERNATING DQN TRAINING")
    print("=" * 80)
    
    # Main training loop - alternate between configurations
    while any(not tracker['completed'] for tracker in config_trackers.values()):
        # Determine which configuration to use for this episode
        config_idx = (global_episode - 1) % len(config_files)
        current_tracker = config_trackers[config_idx]

        # Skip if this configuration has completed all episodes
        if current_tracker['completed']:
            # Find next available configuration
            for i in range(len(config_files)):
                next_idx = (config_idx + i) % len(config_files)
                if not config_trackers[next_idx]['completed']:
                    config_idx = next_idx
                    current_tracker = config_trackers[config_idx]
                    break
            else:
                # All configurations completed
                break

        config_file = current_tracker['config_file']
        config = current_tracker['config']
        config_episode = current_tracker['config_episode']
        
        print(f"\n{'='*80}")
        print(f"GLOBAL EPISODE {global_episode}: Using Config {config_idx + 1}/{len(config_files)}")
        print(f"Config: {config_file}")
        print(f"Config Episode: {config_episode}/{config['total_episodes']}")
        
        # Show resume info for first episode
        if global_episode == args.start_episode and resume_metadata:
            print(f"RESUMED FROM CHECKPOINT: Episode {resume_metadata['global_episode']}")
            print(f"Previous Performance: Reward={resume_metadata['reward']:.2f}, "
                  f"Completion={resume_metadata['completion_rate']:.1f}%")
        
        print(f"{'='*80}")
        
        # Update simulation with current configuration parameters
        shared_simulation_skrl, shared_path, shared_visualization = initialize_dqn_simulation(
            config, shared_simulation_skrl, shared_path, shared_visualization, config_idx
        )

        for demand in ['medium']:
            # Generate routes if needed
            generate_routes_if_needed(current_tracker['config_file'], current_tracker['config'], demand, global_episode)

            # Train one episode with current configuration
            performance_metrics = train_dqn_episode(
                shared_simulation_skrl, global_episode, config_episode, config_file, 
                config, {}
            )

            # Store performance data for this configuration
            current_tracker['performance_history'].append(performance_metrics)

        # Evaluate and save best model if needed
        # current_tracker['best_performance'], global_best_performance = evaluate_and_save_best_model(
        #     performance_metrics, current_tracker['best_performance'], global_best_performance,
        #     config_file, shared_simulation_skrl, shared_path, global_episode, config_episode
        # )
        
        # Save comparison plots at intervals
        save_interval = config.get("save_interval", 10)
        if global_episode % save_interval == 0 and global_episode > 0:
            print(f"Generating plots at global episode {global_episode}...")
            shared_visualization.save_plot(
                episode=global_episode,
                metrics=["reward_avg", "queue_length_avg", "travel_delay_avg", "waiting_time_avg", "outflow_avg"],
                names=["skrl_dqn"],
            )
            print(f"Plots at global episode {global_episode} generated")
            shared_simulation_skrl.save_checkpoint(global_episode)
            print(f"SKRL DQN model saved at global episode {global_episode}")

        # Update episode counters for current configuration
        current_tracker['config_episode'] += 1
        
        # Check if this configuration has completed all episodes
        if current_tracker['config_episode'] > config['total_episodes']:
            current_tracker['completed'] = True
            current_tracker['training_end_time'] = datetime.datetime.now()
            print(f"\nConfiguration {config_file} COMPLETED after {config['total_episodes']} episodes")
        
        # Increment global episode counter
        global_episode += 1

    shared_simulation_skrl.save_model(1)
    
    # Create summary results for all configurations
    for tracker in config_trackers.values():
        config_summary = {
            'config_file': tracker['config_file'],
            'config_name': tracker['config_name'],
            'final_config_episode': tracker['config_episode'] - 1,
            'final_global_episode': global_episode - 1,
            'best_episode': tracker['best_performance']['episode'],
            'best_completion_rate': tracker['best_performance']['completion_rate'],
            'best_total_reward': tracker['best_performance']['total_reward'],
            'best_dqn_avg_outflow': tracker['best_performance'].get('dqn_avg_outflow', 0),
            'best_total_arrived': tracker['best_performance'].get('total_arrived', 0),
            'best_total_departed': tracker['best_performance'].get('total_departed', 0),
            'training_start_time': tracker['training_start_time'],
            'training_end_time': tracker.get('training_end_time', datetime.datetime.now())
        }
        all_config_results.append(config_summary)
        
        # Save performance history for this config
        if tracker['performance_history']:
            performance_df = pd.DataFrame(tracker['performance_history'])
            performance_csv = shared_path + f"performance_history_{tracker['config_name']}.csv"
            performance_df.to_csv(performance_csv, index=False)
            print(f"Performance history saved: {performance_csv}")
    
    print(f"\nALL CONFIGURATIONS COMPLETED - Total global episodes: {global_episode - 1}")
    
    # Final summary
    print("\nMULTI-CONFIGURATION ALTERNATING TRAINING COMPLETED!")
    print("=" * 85)
    if args.resume and resume_metadata:
        print("RESUME INFORMATION:")
        print(f"   Resumed from checkpoint: {os.path.basename(resume_metadata['checkpoint_path'])}")
        print(f"   Started from global episode: {args.start_episode}")
        print(f"   Checkpoint config: {resume_metadata.get('config_file', 'Unknown')}")
        print()
    elif args.load_model or args.load_best:
        print("PRETRAINED MODEL INFORMATION:")
        print(f"   Loaded from: {args.load_model or (args.model_dir or 'models/')}")
        print(f"   Load best models: {args.load_best}")
        print()
    
    print("TRAINING STRUCTURE:")
    print("   1. Support for checkpoint resuming and pretrained model loading")
    print(f"   2. DQN training with ALTERNATING configurations each episode")  
    print("   3. Automatic model saving at regular intervals")
    print()
    
    # Save final checkpoint with comprehensive metadata
    if shared_simulation_skrl and shared_path:
        final_checkpoint_data = {
            'training_completed': True,
            'final_global_episode': global_episode - 1,
            'total_configs': len(config_files),
            'config_files': config_files,
            'training_args': vars(args),
            'completion_timestamp': datetime.datetime.now().isoformat(),
        }
        
        final_checkpoint_path = shared_path + "final_training_checkpoint.pt"
        
        # Add model states from all agents
        for tl_id, agent in shared_simulation_skrl.agent_manager.agents.items():
            final_checkpoint_data[f'model_state_{tl_id}'] = agent.models["q_network"].state_dict()
            final_checkpoint_data[f'target_state_{tl_id}'] = agent.models["target_q_network"].state_dict()
        
        torch.save(final_checkpoint_data, final_checkpoint_path)
        print(f"Final training checkpoint saved: {os.path.basename(final_checkpoint_path)}")
    
    # Save config snapshot for the first config (can be extended for all configs)
    if shared_path and 'first_config' in locals() and first_config:
        save_config_snapshot(first_config, shared_path)
    print(f"\nAll results saved to: {shared_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
