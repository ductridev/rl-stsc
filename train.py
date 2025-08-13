import libsumo as traci
import torch
import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
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


def generate_routes_if_needed(config_file, config):
    """
    Generate routes for simulation if needed based on configuration.
    
    Args:
        config_file: Path to configuration file
        config: Configuration dictionary
    """
    if config_file != "config/training_testngatu6x1EastWestOverflow.yaml":
        print("Generating routes...")
        generate_and_save_random_intervals(
            sumo_cfg_file=config["sumo_cfg_file"],
            total_duration=3600,
            min_interval=360,
            max_interval=360,
            base_weight=100.0,
            high_min=200.0,
            high_max=400.0,
            min_active_sides=1,
            max_active_sides=2,
            edge_groups=config["edge_groups"],
        )

        Intersection.generate_residential_demand_routes(
            config,
            config["sumo_cfg_file"].split("/")[1],
            demand_level="low",
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
            start_step=config["start_step"],
            duration=config["duration"],
            junction_id_list=config["junction_id_list"],
        )
        
        # Generate routes for baseline
        generate_routes_if_needed(config_file, config)
        
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
    # Initialize shared components only once
    if shared_simulation_skrl is None:
        # Set model save path (use first config's model path)
        shared_path = set_train_path(config["models_path_name"])
        shared_visualization = Visualization(path=shared_path, dpi=100)
        
        print(f"Initializing shared DQN model - will be used across all configurations")
        
        # Initialize accident manager for this config
        accident_manager = AccidentManager(
            start_step=config["start_step"],
            duration=config["duration"],
            junction_id_list=config["junction_id_list"],
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
        )
        
        print(f"Shared DQN model initialized with first config")
    else:
        # Update existing simulation with new config parameters
        accident_manager = AccidentManager(
            start_step=config["start_step"],
            duration=config["duration"],
            junction_id_list=config["junction_id_list"],
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
    
    # Generate routes if needed
    generate_routes_if_needed(config_file, config)
    
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
    print("Starting Multi-Configuration DQN Training with Alternating Configurations")
    print("=" * 80)
    
    # Initialize shared variables
    shared_simulation_skrl = None
    shared_visualization = None
    shared_path = None
    global_episode = 1
    
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
    
    config_files = [
        "config/training_testngatu6x1.yaml",
    ]
    
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
    baseline_results_before = run_baseline_simulations(config_files, shared_path, shared_visualization, "before_training")
    baseline_results = {'before_training': baseline_results_before}
    
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
        print(f"{'='*80}")
        
        # Update simulation with current configuration parameters
        shared_simulation_skrl, shared_path, shared_visualization = initialize_dqn_simulation(
            config, shared_simulation_skrl, shared_path, shared_visualization, config_idx
        )
        
        # Train one episode with current configuration
        performance_metrics = train_dqn_episode(
            shared_simulation_skrl, global_episode, config_episode, config_file, 
            config, {}
        )
        
        # Evaluate and save best model if needed
        current_tracker['best_performance'], global_best_performance = evaluate_and_save_best_model(
            performance_metrics, current_tracker['best_performance'], global_best_performance,
            config_file, shared_simulation_skrl, shared_path, global_episode, config_episode
        )
        
        # Store performance data for this configuration
        current_tracker['performance_history'].append(performance_metrics)
        
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
        
        # Save model at intervals
        if global_episode % save_interval == 0 and global_episode > 0:
            shared_simulation_skrl.save_model(global_episode)
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
    
    # Run final baseline simulations for comparison
    baseline_results_after = run_baseline_simulations(config_files, shared_path, shared_visualization, "after_training")
    baseline_results['after_training'] = baseline_results_after
    
    # Compare baselines with DQN results
    compare_baselines_with_dqn(baseline_results, all_config_results)
    
    # Save comprehensive results
    save_comprehensive_results(all_config_results, global_best_performance, shared_path, 
                             baseline_results)
    
    # Final summary
    print("\nMULTI-CONFIGURATION ALTERNATING TRAINING COMPLETED!")
    print("=" * 85)
    print("TRAINING STRUCTURE:")
    print("   1. Baseline simulations run BEFORE DQN training (initial measurements)")
    print(f"   2. DQN training with ALTERNATING configurations each episode")  
    print("   3. Baseline simulations run AFTER DQN training (final comparison)")
    print("   4. Comprehensive performance analysis completed")
    print()
    print("ALTERNATING PATTERN USED:")
    for i, config_file in enumerate(config_files):
        config_name = config_file.split('/')[-1].replace('.yaml', '')
        print(f"   Config {i+1}: {config_name}")
    print(f"   Episodes alternated: 1→Config 1, 2→Config 2, 3→Config 3, 4→Config 1, 5→Config 2, etc.")
    print(f"   Total configurations: {len(config_files)}")
    print()
    print("BEST DQN PERFORMANCE:")
    print(f"   Configuration: {global_best_performance['config_file']}")
    print(f"   Episode: {global_best_performance['episode']}")
    print(f"   Completion Rate: {global_best_performance['completion_rate']:.1f}%")
    print(f"   Total Reward: {global_best_performance['total_reward']:.1f}")
    print(f"   Avg Outflow: {global_best_performance.get('dqn_avg_outflow', 'N/A')}")
    print()
    print(f"EXPLORATION: SKRL handles epsilon-greedy exploration automatically")
    print(f"All results saved to: {shared_path}")
    print("Baseline comparisons available in comprehensive analysis above")
    print("=" * 85)


if __name__ == "__main__":
    main()
