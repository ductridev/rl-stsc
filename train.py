import libsumo as traci
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
# from src.actuated_simulation import ActuatedSimulation  # Commented out for DQN-only training
# from src.base_simulation import SimulationBase  # Commented out for DQN-only training
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

if __name__ == "__main__":
    # Initialize shared model variables
    shared_simulation_skrl = None
    shared_visualization = None
    shared_path = None
    global_episode = 1
    global_epsilon = None
    
    # Track best results across all configurations
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
    
    # Configuration files to train with the same model
    config_files = ["config/training_testngatu6x1EastWestOverflow.yaml", "config/training_testngatu6x1.yaml"]
    
    for config_idx, config_file in enumerate(config_files):
        print(f"\n{'='*60}")
        print(f"Processing configuration {config_idx + 1}/{len(config_files)}: {config_file}")
        print(f"{'='*60}")
        
        # Load configuration
        config = import_train_configuration(config_file)
        min_epsilon = config["agent"]["min_epsilon"]
        decay_rate = config["agent"]["decay_rate"]
        start_epsilon = config["agent"]["epsilon"]
        simulation_path = config["sumocfg_path"]

        # Initialize shared components only once
        if shared_simulation_skrl is None:
            # Set model save path (use first config's model path)
            shared_path = set_train_path(config["models_path_name"])
            shared_visualization = Visualization(path=shared_path, dpi=100)
            
            print(f"Initializing shared DQN model - will be used across all {len(config_files)} configurations")
        else:
            print(f"Reusing shared DQN model for configuration: {config_file}")
        
        # Use shared path and visualization
        path = shared_path
        visualization = shared_visualization

        # Initialize accident manager for this config
        accident_manager = AccidentManager(
            start_step=config["start_step"],
            duration=config["duration"],
            junction_id_list=config["junction_id_list"],
        )

        save_interval = config.get("save_interval", 10)  # Default to every 10 episodes

        # Initialize or reuse shared SKRL-based DQN simulation
        if shared_simulation_skrl is None:
            # First time initialization
            shared_simulation_skrl = Simulation(
                visualization=visualization,
                agent_cfg=config["agent"],
                max_steps=config["max_steps"],
                traffic_lights=config["traffic_lights"],
                accident_manager=accident_manager,
                interphase_duration=config["interphase_duration"],
                epoch=config["training_epochs"],
                path=path,
                training_steps=config["training_steps"],
                updating_target_network_steps=config["updating_target_network_steps"],
                save_interval=save_interval,
            )
            global_epsilon = start_epsilon
            print(f"Shared DQN model initialized with config: {config_file}")
        else:
            # Update existing simulation with new config parameters
            shared_simulation_skrl.accident_manager = accident_manager
            shared_simulation_skrl.max_steps = config["max_steps"]
            shared_simulation_skrl.traffic_lights = config["traffic_lights"]
            shared_simulation_skrl.interphase_duration = config["interphase_duration"]
            shared_simulation_skrl.epoch = config["training_epochs"]
            shared_simulation_skrl.training_steps = config["training_steps"]
            shared_simulation_skrl.updating_target_network_steps = config["updating_target_network_steps"]
            shared_simulation_skrl.save_interval = save_interval
            print(f"Shared DQN model updated with config: {config_file}")
            
        # Use the shared simulation
        simulation_skrl = shared_simulation_skrl

        # Initialize Q-learning simulation for comparison
        agent_memory_q = MemoryPalace(
            max_size_per_palace=config["memory_size_max"], 
            min_size_to_sample=config["memory_size_min"]
        )

        # Load existing SKRL model if specified in config (only for first config)
        start_episode = 1  # Reset episode counter for each configuration
        if config_idx == 0 and config.get("load_model_name") and config["load_model_name"] is not None:
            model_load_path = set_load_model_path(config["models_path_name"]) + config["load_model_name"] + ".pth"
            try:
                simulation_skrl.load_model(start_episode)  # Will try to load from the specified episode
                print(f"Successfully loaded SKRL model from: {model_load_path}")
            except FileNotFoundError:
                print(f"SKRL model file not found: {model_load_path}. Starting with fresh model.")
            except Exception as e:
                print(f"Error loading SKRL model: {e}. Starting with fresh model.")

        # simulation = SimulationBase(
        #     agent_cfg=config["agent"],
        #     max_steps=config["max_steps"],
        #     traffic_lights=config["traffic_lights"],
        #     accident_manager=accident_manager,
        #     visualization=visualization,
        #     epoch=config["training_epochs"],
        #     path=path,
        #     save_interval=save_interval,
        # )

        simulation_q = QSimulation(
            memory=agent_memory_q,
            visualization=visualization,
            agent_cfg=config["agent"],
            max_steps=config["max_steps"],
            traffic_lights=config["traffic_lights"],
            interphase_duration=config["interphase_duration"],
            accident_manager=accident_manager,
            epoch=config["training_epochs"],
            path=path,
            training_steps=config["training_steps"],
            updating_target_network_steps=config["updating_target_network_steps"],
        )

        # Initialize actuated simulation for comparison (commented out for DQN-only training)
        # simulation_actuated = ActuatedSimulation(
        #     agent_cfg=config["agent"],
        #     max_steps=config["max_steps"],
        #     traffic_lights=config["traffic_lights"],
        #     accident_manager=accident_manager,
        #     visualization=visualization,
        #     epoch=config["training_epochs"],
        #     path=path,
        #     save_interval=save_interval,
        # )

        # Load existing Q-table if specified in config
        if config.get("load_q_table_name") and config["load_q_table_name"] is not None:
            q_table_load_path = (
                set_load_model_path(config["models_path_name"])
                + config["load_q_table_name"]
                + ".pkl"
            )
            simulation_q.load_q_table(q_table_load_path)

        # Initialize comparison utility
        comparison = SimulationComparison(path=path)

        episode = start_episode
        epsilon = global_epsilon if global_epsilon is not None else start_epsilon
        timestamp_start = datetime.datetime.now()
        
        print(f"Starting training for config {config_idx + 1}/{len(config_files)}")
        print(f"   Config file: {config_file}")
        print(f"   Episodes: {episode} to {config['total_episodes']}")
        print(f"   Initial epsilon: {epsilon:.4f}")
        print(f"   Shared model path: {path}")
        
        # Track best performance for this specific configuration
        config_best_performance = {
            'episode': -1,
            'completion_rate': -1,
            'total_reward': -float('inf'),
            'combined_score': -float('inf'),
            'config_file': config_file,
            'total_arrived': 0,
            'total_departed': 0,
            'dqn_avg_outflow': 0
        }
        config_performance_history = []

        while episode <= config["total_episodes"]:
            print("\n----- Episode", str(episode), "of", str(config["total_episodes"]))
            # Run the build routes file command
            # Turn off when don't need route generation
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

            # print("Running SimulationBase (static baseline)...")
            # set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
            # simulation_time_base = simulation.run(episode)
            # # After running the base simulation
            # base_stats = simulation.vehicle_tracker.get_current_stats()
            # base_completion_rate = (base_stats['total_arrived'] / max(base_stats['total_departed'], 1)) * 100
            # base_total_arrived = base_stats['total_arrived']
            # # Extract avg_outflow from baseline simulation history
            # base_avg_outflow = 0
            # if hasattr(simulation, 'history') and 'outflow' in simulation.history:
            #     # Get average outflow from all traffic lights
            #     outflow_data = []
            #     for tl_id, outflows in simulation.history['outflow'].items():
            #         if outflows:
            #             avg_outflow_tl = sum(outflows) / len(outflows)
            #             outflow_data.append(avg_outflow_tl)
            #     if outflow_data:
            #         base_avg_outflow = sum(outflow_data) / len(outflow_data)
            
            # # Extract reward from baseline simulation history
            # base_total_reward = 0
            # base_avg_reward = 0
            # if hasattr(simulation, 'history') and 'reward' in simulation.history:
            #     reward_data = []
            #     for tl_id, rewards in simulation.history['reward'].items():
            #         if rewards:
            #             total_reward_tl = sum(rewards)
            #             base_total_reward += total_reward_tl
            #             avg_reward_tl = total_reward_tl / len(rewards)
            #             reward_data.append(avg_reward_tl)
            #     if reward_data:
            #         base_avg_reward = sum(reward_data) / len(reward_data)
            
            # print(f"Base Simulation Results:")
            # print(f"  Total Departed: {base_stats['total_departed']}")
            # print(f"  Total Arrived: {base_total_arrived}")
            # print(f"  Completion Rate: {base_completion_rate:.2f}%")
            # print(f"  Avg Outflow (from history): {base_avg_outflow:.2f}")
            # print(f"  Total Reward: {base_total_reward:.2f}")
            # print(f"  Avg Reward: {base_avg_reward:.2f}")

            # print("SimulationBase time:", simulation_time_base)
            # # Reset base simulation vehicle tracker after its run
            # print("  Resetting vehicle tracker for base simulation")
            # simulation.vehicle_tracker.reset()
            # print("  Resetting history for base simulation")
            # simulation.reset_history()

            # print("Running QSimulation (Q-learning)...")
            # set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
            # simulation_time_q = simulation_q.run(epsilon, episode)
            # print("QSimulation time:", simulation_time_q)

            print("Running SKRL DQN Simulation...")
            # Enable GUI once every 100 episodes for visual monitoring
            gui_enabled = (episode % 100 == 0 and episode > 0)
            set_sumo(gui_enabled, config["sumo_cfg_file"], config["max_steps"])
            if gui_enabled:
                print(f"GUI enabled for episode {episode} - Visual monitoring")
            simulation_time_skrl, training_time_skrl = simulation_skrl.run(epsilon, episode)
            print(
                f"Simulation (SKRL DQN) time:",
                simulation_time_skrl,
                "Training time:",
                training_time_skrl,
            )

            # print("Running ActuatedSimulation (queue-based baseline)...")
            # set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
            # simulation_time_actuated = simulation_actuated.run(episode)
            # # After running the actuated simulation
            # actuated_stats = simulation_actuated.vehicle_tracker.get_current_stats()
            # actuated_completion_rate = (actuated_stats['total_arrived'] / max(actuated_stats['total_departed'], 1)) * 100
            # actuated_total_arrived = actuated_stats['total_arrived']
            # # Extract avg_outflow from actuated simulation history
            # actuated_avg_outflow = 0
            # if hasattr(simulation_actuated, 'history') and 'outflow' in simulation_actuated.history:
            #     # Get average outflow from all traffic lights
            #     outflow_data = []
            #     for tl_id, outflows in simulation_actuated.history['outflow'].items():
            #         if outflows:
            #             avg_outflow_tl = sum(outflows) / len(outflows)
            #             outflow_data.append(avg_outflow_tl)
            #     if outflow_data:
            #         actuated_avg_outflow = sum(outflow_data) / len(outflow_data)
            # 
            # # Extract reward from actuated simulation history
            # actuated_total_reward = 0
            # actuated_avg_reward = 0
            # if hasattr(simulation_actuated, 'history') and 'reward' in simulation_actuated.history:
            #     reward_data = []
            #     for tl_id, rewards in simulation_actuated.history['reward'].items():
            #         if rewards:
            #             total_reward_tl = sum(rewards)
            #             actuated_total_reward += total_reward_tl
            #             avg_reward_tl = total_reward_tl / len(rewards)
            #             reward_data.append(avg_reward_tl)
            # 
            # print(f"Actuated Simulation Results:")
            # print(f"  Total Departed: {actuated_stats['total_departed']}")
            # print(f"  Total Arrived: {actuated_total_arrived}")
            # print(f"  Completion Rate: {actuated_completion_rate:.2f}%")
            # print(f"  Avg Outflow (from history): {actuated_avg_outflow:.2f}")
            # print(f"  Total Reward: {actuated_total_reward:.2f}")

            # print("ActuatedSimulation time:", simulation_time_actuated)
            # # Reset actuated simulation vehicle tracker after its run
            # print("  Resetting vehicle tracker for actuated simulation")
            # simulation_actuated.vehicle_tracker.reset()
            # print("  Resetting history for actuated simulation")
            # simulation_actuated.reset_history()

            epsilon = max(min_epsilon, epsilon * decay_rate)

            # --- Track Performance ---
            # DUAL COMPARISON SYSTEM:
            # 1. DQN vs DQN (Best Model): Use total_reward - higher reward indicates better learning
            # 2. DQN vs Baseline: Use avg_outflow from simulation history - higher outflow is better traffic flow
            try:
                # Get SKRL DQN vehicle statistics
                skrl_stats = simulation_skrl.vehicle_tracker.get_current_stats()
                skrl_total_arrived = skrl_stats['total_arrived']
                # Debug: Print vehicle statistics
                print(f"Debug - Vehicle stats for episode {episode}:")
                print(f"  Total departed: {skrl_stats.get('total_departed', 'N/A')}")
                print(f"  Total arrived: {skrl_stats.get('total_arrived', 'N/A')}")
                print(f"  Current running: {skrl_stats.get('total_running', 'N/A')}")
                
                # Calculate performance metrics
                completion_rate = (skrl_stats['total_arrived'] / max(skrl_stats['total_departed'], 1)) * 100
                
                # Try to get reward metrics from simulation history
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
                    # Get average outflow from all traffic lights
                    outflow_data = []
                    for tl_id, outflows in simulation_skrl.history['outflow'].items():
                        if outflows:
                            avg_outflow_tl = sum(outflows) / len(outflows)
                            outflow_data.append(avg_outflow_tl)
                    if outflow_data:
                        dqn_avg_outflow = sum(outflow_data) / len(outflow_data)
                
                # DUAL COMPARISON SYSTEM:
                # 1. Use total_reward for DQN vs DQN (best model tracking)
                combined_score = dqn_total_reward
                
                # 2. Use avg_outflow for DQN tracking (no comparison needed)
                print(f"DQN Results:")
                print(f"  Avg Outflow: {dqn_avg_outflow:.2f}")  # Removed baseline reference
                print(f"  Total Reward: {dqn_total_reward:.2f}")
                print(f"  Avg Reward: {dqn_avg_reward:.2f}")
                
                # print(f"Actuated Results:")
                # print(f"  Avg Outflow: {actuated_avg_outflow:.2f}")
                # print(f"  Total Reward: {actuated_total_reward:.2f}")
                # print(f"  Avg Reward: {actuated_avg_reward:.2f}")
                # 
                # print(f"Comparison:")
                # print(f"  DQN vs Actuated - Reward: {dqn_total_reward:.2f} vs {actuated_total_reward:.2f}")
                # print(f"  DQN vs Actuated - Outflow: {dqn_avg_outflow:.2f} vs {actuated_avg_outflow:.2f}")
                # print(f"  DQN vs Actuated - Arrivals: {skrl_total_arrived} vs {actuated_total_arrived}")
                
                # Compare DQN vs actuated using multiple metrics (commented out for DQN-only training)
                # performance_improvements = 0
                # if dqn_total_reward > actuated_total_reward:
                #     print(f"DQN reward advantage: {dqn_total_reward:.2f} > {actuated_total_reward:.2f}")
                #     performance_improvements += 1
                # else:
                #     print(f"DQN reward disadvantage: {dqn_total_reward:.2f} <= {actuated_total_reward:.2f}")
                #     
                # if dqn_avg_outflow > actuated_avg_outflow:
                #     print(f"DQN outflow advantage: {dqn_avg_outflow:.2f} > {actuated_avg_outflow:.2f}")
                #     performance_improvements += 1
                # else:
                #     print(f"DQN outflow disadvantage: {dqn_avg_outflow:.2f} <= {actuated_avg_outflow:.2f}")
                #     
                # if skrl_total_arrived > actuated_total_arrived:
                #     print(f"DQN arrival advantage: {skrl_total_arrived} > {actuated_total_arrived}")
                #     performance_improvements += 1
                # else:
                #     print(f"DQN arrival disadvantage: {skrl_total_arrived} <= {actuated_total_arrived}")
                #     
                # # Overall performance summary
                # print(f"🏁 Overall Performance: DQN won {performance_improvements}/3 metrics vs Actuated")
                
                # Save simulation folder if DQN performs better in majority of metrics
                # if performance_improvements >= 2:
                #     dest_folder = f"{simulation_path}_better_SKRL_ep{episode}"
                #     # Remove the destination if it already exists to avoid errors
                #     if os.path.exists(dest_folder):
                #         shutil.rmtree(dest_folder)
                #     shutil.copytree(simulation_path, dest_folder)
                #     print(f"SKRL outperformed base! Higher total_arrived: {skrl_total_arrived:.2f} > {base_total_arrived:.2f}")
                #     print(f"   Simulation folder copied to: {dest_folder}")
                # else:
                #     print(f"SKRL did not outperform base: {skrl_total_arrived:.2f} <= {base_total_arrived:.2f}")
                # Store performance data (actuated variables commented out for DQN-only training)
                current_performance = {
                    'episode': episode,
                    'completion_rate': completion_rate,
                    'total_reward': dqn_total_reward,
                    'combined_score': combined_score,  # total_reward for best model tracking
                    'epsilon': epsilon,
                    'total_departed': skrl_stats['total_departed'],
                    'total_arrived': skrl_stats['total_arrived'],
                    'dqn_avg_outflow': dqn_avg_outflow,  # DQN avg outflow
                    # 'actuated_avg_outflow': actuated_avg_outflow,  # Actuated avg outflow for comparison
                    # 'actuated_total_reward': actuated_total_reward,  # Actuated total reward for comparison
                    # 'actuated_avg_reward': actuated_avg_reward,  # Actuated average reward for comparison
                    # 'actuated_total_arrived': actuated_total_arrived,  # Actuated total arrived for comparison
                    # 'performance_improvements': performance_improvements  # Number of metrics where DQN outperformed actuated
                    # 'base_avg_outflow': base_avg_outflow,  # Baseline avg outflow for comparison
                    # 'base_total_reward': base_total_reward,  # Baseline total reward for comparison
                    # 'base_avg_reward': base_avg_reward,  # Baseline average reward for comparison
                    # 'base_total_arrived': base_total_arrived,  # Baseline total arrived for comparison
                    # 'performance_improvements': performance_improvements  # Number of metrics where DQN outperformed base
                }
                config_performance_history.append(current_performance)
                
                # Check if this is the best performance using 3-metric voting system
                metrics_won = 0
                metric_details = []
                
                # Metric 1: Total Reward (higher is better)
                if dqn_total_reward > config_best_performance['total_reward']:
                    metrics_won += 1
                    metric_details.append(f"Reward: {dqn_total_reward:.2f} > {config_best_performance['total_reward']:.2f}")
                else:
                    metric_details.append(f" Reward: {dqn_total_reward:.2f} <= {config_best_performance['total_reward']:.2f}")
                
                # Metric 2: Completion Rate (higher is better)
                if completion_rate > config_best_performance['completion_rate']:
                    metrics_won += 1
                    metric_details.append(f"Completion: {completion_rate:.1f}% > {config_best_performance['completion_rate']:.1f}%")
                else:
                    metric_details.append(f"Completion: {completion_rate:.1f}% <= {config_best_performance['completion_rate']:.1f}%")
                
                # Metric 3: Average Outflow (higher is better)
                if dqn_avg_outflow > config_best_performance.get('dqn_avg_outflow', 0):
                    metrics_won += 1
                    metric_details.append(f"Outflow: {dqn_avg_outflow:.2f} > {config_best_performance.get('dqn_avg_outflow', 0):.2f}")
                else:
                    metric_details.append(f"Outflow: {dqn_avg_outflow:.2f} <= {config_best_performance.get('dqn_avg_outflow', 0):.2f}")
                
                print(f"\n3-Metric Performance Comparison - Episode {episode}:")
                for detail in metric_details:
                    print(f"   {detail}")
                print(f"   Result: Won {metrics_won}/3 metrics")
                
                # Save model if it wins majority of metrics (2 or more out of 3)
                if metrics_won >= 2:
                    # Clean up old best model files first
                    model_save_name = config.get("save_model_name", "dqn_model")
                    
                    # Remove old best files if they exist
                    if config_best_performance['episode'] != -1:  # If there was a previous best
                        # Remove old best model files for all traffic lights
                        import glob
                        old_best_pattern = path + f"skrl_model_*_episode_{config_best_performance['episode']}_BEST.pt"
                        old_checkpoint_pattern = path + f"skrl_model_*_episode_{config_best_performance['episode']}_BEST_checkpoint.pt"
                        
                        try:
                            # Remove old best model files
                            for old_file in glob.glob(old_best_pattern):
                                os.remove(old_file)
                                print(f"   Removed old best model: {os.path.basename(old_file)}")
                            
                            for old_file in glob.glob(old_checkpoint_pattern):
                                os.remove(old_file)
                                print(f"   Removed old best checkpoint: {os.path.basename(old_file)}")
                        except Exception as e:
                            print(f"   Could not remove old best files: {e}")
                    
                    # Update best performance record
                    config_best_performance = current_performance.copy()
                    config_best_performance['config_file'] = config_file
                    print(f"\nNEW BEST PERFORMANCE for {config_file} at Episode {episode}!")
                    print(f"   Won {metrics_won}/3 metrics - Majority Victory!")
                    print(f"   Reward: {dqn_total_reward:.2f}")
                    print(f"   Completion Rate: {completion_rate:.1f}%")
                    print(f"   Avg Outflow: {dqn_avg_outflow:.2f}")
                    
                    # Check if this is also the global best using same 3-metric system
                    global_metrics_won = 0
                    
                    # Compare against global best
                    if dqn_total_reward > global_best_performance['total_reward']:
                        global_metrics_won += 1
                    if completion_rate > global_best_performance['completion_rate']:
                        global_metrics_won += 1
                    if dqn_avg_outflow > global_best_performance.get('dqn_avg_outflow', 0):
                        global_metrics_won += 1
                    
                    if global_metrics_won >= 2:
                        global_best_performance = current_performance.copy()
                        global_best_performance['config_file'] = config_file
                        global_best_performance['dqn_avg_outflow'] = dqn_avg_outflow
                        print(f"   This is also the GLOBAL BEST across all configurations! ({global_metrics_won}/3 metrics)")
                    
                    # Save BEST model files in .pt format
                    print(f"   Saving BEST models in .pt format...")
                    
                    # Save current best models with BEST suffix
                    for tl_id in simulation_skrl.agent_manager.agents.keys():
                        agent = simulation_skrl.agent_manager.agents[tl_id]
                        
                        # Best model with episode number
                        best_model_path = path + f"skrl_model_{tl_id}_episode_{episode}_BEST.pt"
                        torch.save(agent.models["q_network"].state_dict(), best_model_path)
                        print(f"     {os.path.basename(best_model_path)}")
                        
                        # Best checkpoint with episode number
                        best_checkpoint_path = path + f"skrl_model_{tl_id}_episode_{episode}_BEST_checkpoint.pt"
                        checkpoint = {
                            "model_state_dict": agent.models["q_network"].state_dict(),
                            "target_model_state_dict": agent.models["target_q_network"].state_dict(),
                            "episode": episode,
                            "epsilon": epsilon,
                            "reward": dqn_total_reward,
                            "completion_rate": completion_rate,
                            "avg_outflow": dqn_avg_outflow,
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
                    simulation_skrl.save_model(episode)
                    print(f"   Standard SKRL models also saved for episode {episode}")
                else:
                    print(f"   Not saving: Only won {metrics_won}/3 metrics (need 2+ for majority)")
                
                # Print current vs best performance every episode
                print(f"\nPerformance Summary - Episode {episode}")
                print(f"Current DQN: Reward={dqn_total_reward:.2f}, Outflow={dqn_avg_outflow:.2f}, Completion={completion_rate:.1f}%")
                # print(f"Current Actuated: Reward={actuated_total_reward:.2f}, Outflow={actuated_avg_outflow:.2f}, Completion={actuated_completion_rate:.1f}%")
                print(f"Best DQN:    Reward={config_best_performance['total_reward']:.2f}, Episode={config_best_performance['episode']}, Completion={config_best_performance['completion_rate']:.1f}%")
                print(f"             Outflow={config_best_performance.get('dqn_avg_outflow', 0):.2f} (Won {metrics_won if 'metrics_won' in locals() else 'N/A'}/3 metrics this episode)")
                print(f"Global Best: Reward={global_best_performance['total_reward']:.2f}, Episode={global_best_performance['episode']}, Config={global_best_performance.get('config_file', 'N/A')}")
                print(f"             Outflow={global_best_performance.get('dqn_avg_outflow', 0):.2f}, Completion={global_best_performance['completion_rate']:.1f}%")
                # print(f"DQN vs Actuated: Reward={dqn_total_reward:.2f} vs {actuated_total_reward:.2f}, Outflow={dqn_avg_outflow:.2f} vs {actuated_avg_outflow:.2f}")
                # print(f"Current Base: Reward={base_total_reward:.2f}, Outflow={base_avg_outflow:.2f}, Completion={base_completion_rate:.1f}%")
                # print(f"DQN vs Base: Reward={dqn_total_reward:.2f} vs {base_total_reward:.2f}, Outflow={dqn_avg_outflow:.2f} vs {base_avg_outflow:.2f}")
                    
            except AttributeError as e:
                if 'num_atoms' in str(e):
                    print(f"Model compatibility error: {e}")
                    print("   This error indicates the DQN model is missing the 'num_atoms' attribute for C51 distributional DQN.")
                    print("   The model should be updated to include: self.num_atoms = 51")
                else:
                    print(f"Attribute error tracking performance: {e}")
                # Continue training even if performance tracking fails
            except Exception as e:
                print(f"Error tracking performance: {e}")
                # Continue training even if performance tracking fails
            
            # Reset vehicle trackers after performance tracking is complete
            print(f"  Resetting vehicle tracker for DQN simulation")
            simulation_skrl.vehicle_tracker.reset()
            print(f"  Resetting history for DQN simulation")
            simulation_skrl.reset_history()

            # --- Save comparison plots ---
            print("Saving comparison plots...")
            if episode % save_interval == 0 and episode > 0:
                print("Generating plots at episode", episode, "...")
                visualization.save_plot(
                    episode=episode,
                    metrics=[
                        "reward_avg",
                        "queue_length_avg",
                        "travel_delay_avg",
                        "waiting_time_avg",
                        "outflow_avg"
                    ],
                    names=["skrl_dqn"],  # DQN only (actuated commented out)
                    # names=["skrl_dqn", "actuated"],  # Include both DQN and actuated simulations
                    # names=["skrl_dqn", "base"],  # Only include actually running simulations
                )
                print("Plots at episode", episode, "generated")

                # --- Generate traffic light comparison tables ---
                print("Generating traffic light comparison tables...")
                try:
                    # Specify the actual simulation types that are running and saving data
                    available_sim_types = ["skrl_dqn"]  # DQN only (actuated commented out)
                    # available_sim_types = ["skrl_dqn", "actuated"]  # Only DQN (baseline commented out)
                    # available_sim_types = ["baseline", "skrl_dqn"]  # Add q_learning when it's enabled
                    metrics = ["reward", "queue_length", "travel_delay", "waiting_time", "outflow"]
                    comparison.save_comparison_tables(episode, metrics, simulation_types=available_sim_types)
                    comparison.print_comparison_summary(episode, metrics, simulation_types=available_sim_types)
                    print("Traffic light comparison tables generated successfully")
                except Exception as e:
                    print(f"Error generating comparison tables: {e}")
                    print(
                        "Comparison tables will be generated when CSV files are available"
                    )
                # --- Generate comparison results were generated ---
                try:
                    print("Traffic light comparison tables generated successfully")
                    visualization.save_comparison_plots(episode=episode)
                except Exception as e:
                    print(f"Error generating comparison plots: {e}")
                    print("No comparison results available for plotting.")

                # --- Generate vehicle comparison from logs ---
                print("Generating vehicle comparison from logs...")
                try:
                    visualization.create_vehicle_comparison_from_logs(episode, ["skrl_dqn"])  # DQN only (actuated commented out)
                    # visualization.create_vehicle_comparison_from_logs(episode, ["skrl_dqn", "actuated"])  # Include both DQN and actuated
                    # visualization.create_vehicle_comparison_from_logs(episode, ["skrl_dqn", "base"])  # Updated for SKRL
                    print("Vehicle comparison from logs generated successfully")
                except Exception as e:
                    print(f"Error generating vehicle comparison from logs: {e}")
                    print("Vehicle comparison will be generated when log files are available")

            # Save model at specified intervals
            if episode % save_interval == 0 and episode > 0:
                model_save_name = config.get("save_model_name", "skrl_dqn_model")

                # Save SKRL model
                simulation_skrl.save_model(episode)
                print(f"SKRL DQN model saved at episode {episode}")

                # Save Q-learning Q-table (only if Q-learning is running)
                # Note: Uncomment when Q-learning is enabled
                # q_table_save_path = path + f"q_table_{model_save_name}_episode_{episode}.pkl"
                # simulation_q.save_q_table(path, episode)

            episode += 1

        # Update global episode and epsilon for next configuration
        global_episode = episode
        global_epsilon = epsilon
        
        print(f"\nCompleted training for config: {config_file}")
        print(f"   Final episode: {episode-1}")
        print(f"   Final epsilon: {epsilon:.4f}")
        print(f"   Config best performance: Episode {config_best_performance['episode']}, Score {config_best_performance['total_reward']:.2f}")

        # Show saved model files summary
        print(f"\nSAVED MODEL FILES SUMMARY:")
        try:
            import glob
            model_patterns = [
                (f"*_BEST.pt", "Best Models"),
                (f"*_BEST_checkpoint.pt", "Best Checkpoints"), 
                (f"*_CURRENT_BEST.pt", "Current Best Models"),
                (f"*_GLOBAL_BEST.pt", "Global Best Models"),
                (f"skrl_model_*_episode_*.pt", "Standard Episode Models")
            ]
            
            for pattern, description in model_patterns:
                files = glob.glob(path + pattern)
                if files:
                    print(f"   {description}: {len(files)} files")
                    for file in files[-3:]:  # Show last 3 files
                        print(f"     - {os.path.basename(file)}")
                    if len(files) > 3:
                        print(f"     ... and {len(files) - 3} more")
        except Exception as e:
            print(f"   Could not list model files: {e}")

        # Store results for this configuration
        config_summary = {
            'config_file': config_file,
            'config_name': config_file.split('/')[-1].replace('.yaml', ''),
            'final_episode': episode - 1,
            'final_epsilon': epsilon,
            'best_episode': config_best_performance['episode'],
            'best_completion_rate': config_best_performance['completion_rate'],
            'best_total_reward': config_best_performance['combined_score'],
            'best_dqn_avg_outflow': config_best_performance.get('dqn_avg_outflow', 0),
            'best_total_arrived': config_best_performance.get('total_arrived', 0),
            'best_total_departed': config_best_performance.get('total_departed', 0),
            'training_start_time': timestamp_start,
            'training_end_time': datetime.datetime.now()
        }
        all_config_results.append(config_summary)

        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", path)
        
        # Print best performance summary for this config
        print(f"\nBEST PERFORMANCE FOR {config_file} (3-Metric Voting System)")
        print("=" * 60)
        print(f"Best Episode: {config_best_performance['episode']}")
        print(f"Metrics that Won:")
        print(f"   Total Reward: {config_best_performance['total_reward']:.2f}")
        print(f"   Completion Rate: {config_best_performance['completion_rate']:.2f}%")
        print(f"   Avg Outflow: {config_best_performance.get('dqn_avg_outflow', 'N/A')}")
        print(f"Vehicles: {config_best_performance['total_arrived']}/{config_best_performance['total_departed']}")
        print(f"Note: Model saved when winning ≥2 out of 3 metrics")
        
        # Save performance history to CSV for this config
        if config_performance_history:
            import pandas as pd
            performance_df = pd.DataFrame(config_performance_history)
            config_name = config_file.split('/')[-1].replace('.yaml', '')
            performance_csv = path + f"performance_history_{config_name}.csv"
            performance_df.to_csv(performance_csv, index=False)
            print(f"Performance history saved: {performance_csv}")
            
            # Create performance plot for this config
            try:
                import matplotlib.pyplot as plt
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Training Performance Over Time - {config_name}', fontsize=16)
                
                episodes = performance_df['episode']
                
                # Plot 1: Completion Rate
                ax1.plot(episodes, performance_df['completion_rate'], 'b-', linewidth=2)
                ax1.axhline(y=config_best_performance['completion_rate'], color='r', linestyle='--', alpha=0.7)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Completion Rate (%)')
                ax1.set_title('Vehicle Completion Rate')
                ax1.grid(True, alpha=0.3)
                ax1.text(0.02, 0.98, f"Best: {config_best_performance['completion_rate']:.1f}% (Ep {config_best_performance['episode']})", 
                        transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                # Plot 2: Reward Progress (DQN only)
                ax2.plot(episodes, performance_df['total_reward'], 'g-', linewidth=2, label='DQN Reward')
                # ax2.plot(episodes, performance_df['actuated_total_reward'], 'orange', linewidth=2, label='Actuated Reward')
                ax2.axhline(y=config_best_performance['combined_score'], color='r', linestyle='--', alpha=0.7, label='Best DQN')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Total Reward')
                ax2.set_title('DQN Reward Progress')  # Updated title for DQN only
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.text(0.02, 0.98, f"Best DQN: {config_best_performance['combined_score']:.1f} (Ep {config_best_performance['episode']})", 
                        transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                # Plot 3: Outflow Progress (DQN only)
                ax3.plot(episodes, performance_df['dqn_avg_outflow'], 'purple', linewidth=2, label='DQN Outflow')
                # ax3.plot(episodes, performance_df['actuated_avg_outflow'], 'brown', linewidth=2, label='Actuated Outflow')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Avg Outflow')
                ax3.set_title('DQN Outflow Progress')  # Updated title for DQN only
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                # Plot 4: Epsilon Decay
                ax4.plot(episodes, performance_df['epsilon'], 'purple', linewidth=2)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Epsilon')
                ax4.set_title('Exploration Rate (Epsilon)')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                performance_plot = path + f"performance_history_{config_name}.png"
                plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Performance plot saved: {performance_plot}")
                
            except Exception as e:
                print(f"Could not create performance plot: {e}")
        

        print(f"Saving final model at {path}...")
        # model_save_name = config.get("save_model_name", "dqn_model")

        # # Save final model weights
        # model_final_path = path + f"{model_save_name}_final.pth"
        # simulation_dqn.agent.save(model_final_path)

        # # Save final checkpoint
        # checkpoint_final_path = path + f"{model_save_name}_final_checkpoint.pth"
        # simulation_dqn.agent.save_checkpoint(checkpoint_final_path, episode=episode, epsilon=epsilon)

        # print(f"Final DQN model saved: {model_final_path}")
        # print(f"Final DQN checkpoint saved: {checkpoint_final_path}")

        # Save final Q-table
        simulation_q.save_q_table(path, episode="final")
        print(f"Final Q-table saved for config: {config_file}")
        print("---------------------------------------")

        print("Generating plots...")
        # We simplify by averaging the history over all traffic lights
        avg_history = {}

        for metric, data_per_tls in simulation_skrl.history.items():
            # Transpose the list of lists into per-timestep values
            # Filter out missing/empty lists first
            data_lists = [data for data in data_per_tls.values() if len(data) > 0]

            if not data_lists:
                continue  # Skip if no data

            # Truncate to minimum length to avoid index errors
            min_length = min(len(data) for data in data_lists)
            data_lists = [data[:min_length] for data in data_lists]

            # Average per timestep
            avg_data = [sum(step_vals) / len(step_vals) for step_vals in zip(*data_lists)]

            # Save to avg_history
            avg_history[metric] = avg_data

        config_name = config_file.split('/')[-1].replace('.yaml', '')
        for metric, data in avg_history.items():
            visualization.save_data_and_plot(
                data=data,
                filename=f"skrl_{metric}_avg_{config_name}",
                xlabel="Step",
                ylabel=metric.replace("_", " ").title(),
            )

        print(f"Plots generated for config: {config_file}")
        
    # Final summary across all configurations
    print(f"\nTRAINING COMPLETED FOR ALL {len(config_files)} CONFIGURATIONS")
    print("=" * 80)
    print(f"Shared DQN model trained across multiple traffic scenarios")
    print(f"All results saved to: {shared_path}")
    print(f"Final global episode: {global_episode - 1}")
    print(f"Final epsilon: {global_epsilon:.6f}")
    
    # Save final shared model
    if shared_simulation_skrl is not None:
        print(f"\nSaving final shared DQN model...")
        shared_simulation_skrl.save_model(global_episode - 1)
        
        # Also save final models in .pt format with descriptive names
        print(f"   Saving final models in .pt format...")
        final_episode = global_episode - 1
        
        for tl_id in shared_simulation_skrl.agent_manager.agents.keys():
            agent = shared_simulation_skrl.agent_manager.agents[tl_id]
            
            # Final model
            final_model_path = shared_path + f"skrl_model_{tl_id}_FINAL_episode_{final_episode}.pt"
            torch.save(agent.models["q_network"].state_dict(), final_model_path)
            
            # Final checkpoint with comprehensive metadata
            final_checkpoint_path = shared_path + f"skrl_model_{tl_id}_FINAL_episode_{final_episode}_checkpoint.pt"
            final_checkpoint = {
                "model_state_dict": agent.models["q_network"].state_dict(),
                "target_model_state_dict": agent.models["target_q_network"].state_dict(),
                "episode": final_episode,
                "epsilon": global_epsilon,
                "global_best_reward": global_best_performance['total_reward'],
                "global_best_completion_rate": global_best_performance['completion_rate'],
                "global_best_avg_outflow": global_best_performance.get('dqn_avg_outflow', 0),
                "global_best_config": global_best_performance['config_file'],
                "total_configs_trained": len(config_files),
                "training_completion_time": datetime.datetime.now().isoformat(),
                "model_selection_method": "3-Metric Voting System",
                "selection_threshold": ">=2 out of 3 metrics must improve"
            }
            torch.save(final_checkpoint, final_checkpoint_path)
            print(f"     {os.path.basename(final_model_path)}")
            print(f"     {os.path.basename(final_checkpoint_path)}")
        
        print(f"Final shared DQN model saved at episode {global_episode - 1}")
    
    # ==================== COMPREHENSIVE RESULTS SUMMARY ====================
    print(f"\n{'COMPREHENSIVE RESULTS SUMMARY':^80}")
    print("=" * 80)
    
    # Overall training summary
    total_training_time = datetime.timedelta(0)
    for config in all_config_results:
        training_duration = config['training_end_time'] - config['training_start_time']
        total_training_time += training_duration
    
    print(f"OVERALL TRAINING STATISTICS:")
    print(f"   Total configurations trained: {len(config_files)}")
    print(f"   Total episodes completed: {global_episode - 1}")
    print(f"   Total training time: {total_training_time}")
    print(f"   Average training time per config: {total_training_time / len(config_files)}")
    
    # Best performance across all configurations
    print(f"\nGLOBAL BEST PERFORMANCE (3-Metric Voting System):")
    print("-" * 70)
    print(f"Best Configuration: {global_best_performance['config_file']}")
    print(f"   Episode: {global_best_performance['episode']}")
    print(f"   Won majority (≥2/3) of these metrics:")
    print(f"      Total Reward: {global_best_performance['total_reward']:.2f}")
    print(f"      Completion Rate: {global_best_performance['completion_rate']:.2f}%")
    print(f"      Avg Outflow: {global_best_performance.get('dqn_avg_outflow', 'N/A')}")
    print(f"   Vehicles: {global_best_performance['total_arrived']}/{global_best_performance['total_departed']}")
    print(f"   Epsilon at best: {global_best_performance.get('epsilon', 'N/A'):.6f}")
    
    # Configuration-by-configuration breakdown
    print(f"\nPERFORMANCE BY CONFIGURATION:")
    print("-" * 80)
    
    for i, config in enumerate(all_config_results, 1):
        config_name = config['config_name']
        training_duration = config['training_end_time'] - config['training_start_time']
        
        # Mark if this is the global best
        global_best_marker = " GLOBAL BEST" if config['config_file'] == global_best_performance['config_file'] else ""
        
        print(f"{i}. {config_name}{global_best_marker}")
        print(f"   File: {config['config_file']}")
        print(f"   Training time: {training_duration}")
        print(f"   Episodes: {config['final_episode']} (best at ep {config['best_episode']})")
        print(f"   Best completion rate: {config['best_completion_rate']:.2f}%")
        print(f"   Best total reward: {config['best_total_reward']:.2f}")
        print(f"   Best avg outflow: {config['best_dqn_avg_outflow']:.2f}")
        print(f"   Best vehicles: {config['best_total_arrived']}/{config['best_total_departed']}")
        print(f"   Final epsilon: {config['final_epsilon']:.6f}")
        print()
    
    # Performance comparison table  
    print(f"PERFORMANCE COMPARISON TABLE (3-Metric Voting System):")
    print("-" * 85)
    print(f"{'Config':<25} {'Best Episode':<12} {'Completion%':<12} {'Reward':<12} {'Outflow':<10} {'Status':<8}")
    print("-" * 85)
    
    for config in all_config_results:
        config_name = config['config_name'][:22] + "..." if len(config['config_name']) > 25 else config['config_name']
        is_best = "[BEST]" if config['config_file'] == global_best_performance['config_file'] else "     "
        status = "BEST" if config['config_file'] == global_best_performance['config_file'] else "     "
        
        print(f"{is_best}{config_name:<20} {config['best_episode']:<12} {config['best_completion_rate']:<11.1f}% {config['best_total_reward']:<11.1f} {config['best_dqn_avg_outflow']:<9.1f} {status:<8}")
    
    print("-" * 85)
    print("Note: Models saved when winning ≥2 out of 3 metrics (Reward, Completion Rate, Outflow)")
    
    # Save comprehensive results to file
    results_summary_file = shared_path + "comprehensive_results_summary.txt"
    try:
        with open(results_summary_file, 'w') as f:
            f.write("COMPREHENSIVE TRAINING RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed: {datetime.datetime.now()}\n")
            f.write(f"Total configurations: {len(config_files)}\n")
            f.write(f"Total episodes: {global_episode - 1}\n")
            f.write(f"Total training time: {total_training_time}\n")
            f.write(f"Model Selection: 3-Metric Voting System (≥2/3 metrics must improve)\n")
            f.write(f"Metrics: Total Reward, Completion Rate, Average Outflow\n\n")
            
            f.write("GLOBAL BEST PERFORMANCE (3-Metric Voting System):\n")
            f.write(f"Configuration: {global_best_performance['config_file']}\n")
            f.write(f"Episode: {global_best_performance['episode']}\n")
            f.write(f"Total Reward: {global_best_performance['total_reward']:.2f}\n")
            f.write(f"Completion Rate: {global_best_performance['completion_rate']:.2f}%\n")
            f.write(f"Avg Outflow: {global_best_performance.get('dqn_avg_outflow', 'N/A')}\n")
            f.write(f"Vehicles: {global_best_performance['total_arrived']}/{global_best_performance['total_departed']}\n\n")
            
            f.write("CONFIGURATION DETAILS:\n")
            f.write("-" * 30 + "\n")
            for config in all_config_results:
                f.write(f"\nConfig: {config['config_name']}\n")
                f.write(f"  File: {config['config_file']}\n")
                f.write(f"  Best Episode: {config['best_episode']}\n")
                f.write(f"  Best Total Reward: {config['best_total_reward']:.2f}\n")
                f.write(f"  Best Completion Rate: {config['best_completion_rate']:.2f}%\n")
                f.write(f"  Best Avg Outflow: {config['best_dqn_avg_outflow']:.2f}\n")
                f.write(f"  Training Duration: {config['training_end_time'] - config['training_start_time']}\n")
            
            f.write(f"\nNOTE: Models were saved only when they achieved majority wins (≥2/3) across all three metrics.\n")
        
        print(f"Comprehensive results summary saved: {results_summary_file}")
    except Exception as e:
        print(f" Could not save results summary file: {e}")
    
    # Create comprehensive performance comparison CSV
    try:
        import pandas as pd
        comparison_df = pd.DataFrame([
            {
                'Config_Name': config['config_name'],
                'Config_File': config['config_file'],
                'Best_Episode': config['best_episode'],
                'Best_Total_Reward': config['best_total_reward'],
                'Best_Completion_Rate': config['best_completion_rate'],
                'Best_Avg_Outflow': config['best_dqn_avg_outflow'],
                'Best_Total_Arrived': config['best_total_arrived'],
                'Best_Total_departed': config['best_total_departed'],
                'Final_Episode': config['final_episode'],
                'Final_Epsilon': config['final_epsilon'],
                'Training_Duration_Seconds': (config['training_end_time'] - config['training_start_time']).total_seconds(),
                'Is_Global_Best': config['config_file'] == global_best_performance['config_file'],
                'Model_Selection_Method': '3-Metric_Voting_System',
                'Metrics_Used': 'Total_Reward,Completion_Rate,Avg_Outflow',
                'Selection_Threshold': 'Majority_Win_2_of_3_Metrics'
            } for config in all_config_results
        ])
        
        comparison_csv = shared_path + "all_configs_performance_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Performance comparison CSV saved: {comparison_csv}")
    except Exception as e:
        print(f" Could not save performance comparison CSV: {e}")
    
    # Create comprehensive model inventory
    print(f"\nCOMPREHENSIVE MODEL INVENTORY")
    print("=" * 70)
    try:
        import glob
        
        all_models = glob.glob(shared_path + "*.pt")
        if all_models:
            model_categories = {
                "Best Performance Models": [f for f in all_models if "_BEST" in f and "CURRENT" not in f and "GLOBAL" not in f],
                "Global Best Models": [f for f in all_models if "GLOBAL_BEST" in f],
                "Current Best Models": [f for f in all_models if "CURRENT_BEST" in f],
                "Standard Episode Models": [f for f in all_models if "episode_" in f and "BEST" not in f and "FINAL" not in f],
                "Final Models": [f for f in all_models if "FINAL" in f],
            }
            
            total_models = 0
            for category, files in model_categories.items():
                if files:
                    print(f"\n{category}: {len(files)} files")
                    total_models += len(files)
                    # Show a few examples
                    for file in files[:3]:
                        size_mb = os.path.getsize(file) / (1024 * 1024)
                        print(f"  {os.path.basename(file)} ({size_mb:.2f} MB)")
                    if len(files) > 3:
                        print(f"  ... and {len(files) - 3} more files")
            
            print(f"\nTOTAL MODELS SAVED: {total_models} .pt files")
            total_size_mb = sum(os.path.getsize(f) for f in all_models) / (1024 * 1024)
            print(f"TOTAL SIZE: {total_size_mb:.2f} MB")
            
            # Best model recommendations
            print(f"\nRECOMMENDED MODELS FOR INFERENCE:")
            global_best_files = [f for f in all_models if "GLOBAL_BEST" in f and "checkpoint" not in f]
            if global_best_files:
                print(f"   Primary: Global Best Models (won ≥2/3 metrics)")
                for file in global_best_files:
                    print(f"     {os.path.basename(file)}")
            
            current_best_files = [f for f in all_models if "CURRENT_BEST" in f and "checkpoint" not in f]
            if current_best_files:
                print(f"   Alternative: Current Best Models (latest best per config)")
                for file in current_best_files[:2]:  # Show first 2
                    print(f"    {os.path.basename(file)}")
        else:
            print("No .pt model files found!")
    
    except Exception as e:
        print(f"Could not create model inventory: {e}")
    
    print("=" * 70)

    # Final success message
    print(f"\nMULTI-CONFIGURATION TRAINING SESSION COMPLETED SUCCESSFULLY! 🎊")
    print(f"Best performance achieved with: {global_best_performance['config_file']}")
    print(f"Best reward: {global_best_performance['total_reward']:.2f}")
    print(f"Best completion rate: {global_best_performance['completion_rate']:.2f}%")
    print(f"Best outflow: {global_best_performance.get('dqn_avg_outflow', 'N/A')}")
    print(f"Model selected using 3-metric voting system (≥2/3 metrics must improve)")
    print("=" * 85)