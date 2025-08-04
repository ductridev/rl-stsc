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

if __name__ == "__main__":
    # Load configuration
    config = import_train_configuration("config/training_testngatu6x1.yaml")
    min_epsilon = config["agent"]["min_epsilon"]
    decay_rate = config["agent"]["decay_rate"]
    start_epsilon = config["agent"]["epsilon"]

    # Set model save path
    path = set_train_path(config["models_path_name"])

    visualization = Visualization(path=path, dpi=100)

    # Initialize accident manager
    accident_manager = AccidentManager(
        start_step=config["start_step"],
        duration=config["duration"],
        junction_id_list=config["junction_id_list"],
    )

    # Initialize SKRL-based DQN simulation (no memory needed - SKRL handles it internally)
    simulation_skrl = Simulation(
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
        save_interval=config["save_interval"],
    )

    # Initialize Q-learning simulation for comparison
    agent_memory_q = MemoryPalace(
        max_size_per_palace=config["memory_size_max"], 
        min_size_to_sample=config["memory_size_min"]
    )

    # Load existing SKRL model if specified in config
    start_episode = 0
    if config.get("load_model_name") and config["load_model_name"] is not None:
        model_load_path = set_load_model_path(config["models_path_name"]) + config["load_model_name"] + ".pth"
        try:
            simulation_skrl.load_model(start_episode)  # Will try to load from the specified episode
            print(f"Successfully loaded SKRL model from: {model_load_path}")
        except FileNotFoundError:
            print(f"SKRL model file not found: {model_load_path}. Starting with fresh model.")
        except Exception as e:
            print(f"Error loading SKRL model: {e}. Starting with fresh model.")

    simulation = SimulationBase(
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        accident_manager=accident_manager,
        visualization=visualization,
        epoch=config["training_epochs"],
        path=path,
    )

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
    epsilon = start_epsilon
    timestamp_start = datetime.datetime.now()
    
    # Track best performance
    best_performance = {
        'episode': -1,
        'completion_rate': -1,
        'avg_travel_time': float('inf'),
        'avg_waiting_time': float('inf'),
        'combined_score': -float('inf')
    }
    performance_history = []

    while episode < config["total_episodes"]:
        print("\n----- Episode", str(episode + 1), "of", str(config["total_episodes"]))
        print("Generating routes...")
        # Run the build routes file command

        generate_and_save_random_intervals(
            sumo_cfg_file=config["sumo_cfg_file"],
            total_duration=3600,
            min_interval=360,
            max_interval=360,
            base_weight=100.0,
            high_min=200.0,
            high_max=400.0,
            min_active_sides=1,
            max_active_sides=3,
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

        print("Running SimulationBase (static baseline)...")
        set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
        simulation_time_base = simulation.run(episode)
        print("SimulationBase time:", simulation_time_base)
        # Reset base simulation vehicle tracker after its run
        print("  Resetting vehicle tracker for base simulation")
        simulation.vehicle_tracker.reset()
        print("  Resetting history for base simulation")
        simulation.reset_history()

        # print("Running QSimulation (Q-learning)...")
        # set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
        # simulation_time_q = simulation_q.run(epsilon, episode)
        # print("QSimulation time:", simulation_time_q)

        print("Running SKRL DQN Simulation...")
        # Enable GUI once every 100 episodes for visual monitoring
        gui_enabled = (episode % 100 == 0 and episode > 0)
        set_sumo(gui_enabled, config["sumo_cfg_file"], config["max_steps"])
        if gui_enabled:
            print(f"🖥️  GUI enabled for episode {episode} - Visual monitoring")
        simulation_time_skrl, training_time_skrl = simulation_skrl.run(epsilon, episode)
        print(
            f"Simulation (SKRL DQN) time:",
            simulation_time_skrl,
            "Training time:",
            training_time_skrl,
        )

        epsilon = max(min_epsilon, epsilon * decay_rate)

        # --- Track Performance ---
        try:
            # Get SKRL DQN vehicle statistics
            skrl_stats = simulation_skrl.vehicle_tracker.get_current_stats()
            
            # Debug: Print vehicle statistics
            print(f"Debug - Vehicle stats for episode {episode}:")
            print(f"  Total departed: {dqn_stats.get('total_departed', 'N/A')}")
            print(f"  Total arrived: {dqn_stats.get('total_arrived', 'N/A')}")
            print(f"  Current running: {dqn_stats.get('total_running', 'N/A')}")
            
            # Calculate performance metrics
            completion_rate = (skrl_stats['total_arrived'] / max(skrl_stats['total_departed'], 1)) * 100
            
            # Get traffic metrics (if available)
            avg_travel_time = 0
            avg_waiting_time = 0
            
            # Debug: Check if history exists and has data
            if hasattr(dqn_sim, 'history'):
                print(f"  History keys: {list(dqn_sim.history.keys())}")
                if 'travel_time' in dqn_sim.history:
                    travel_time_data = dqn_sim.history['travel_time']
                    print(f"  Travel time data available for TLs: {list(travel_time_data.keys())}")
                    for tl_id, times in travel_time_data.items():
                        print(f"    {tl_id}: {len(times)} measurements")
            else:
                print("  No history attribute found")
            
            # Try to get traffic metrics from simulation history
            if hasattr(simulation_skrl, 'history') and 'travel_time' in simulation_skrl.history:
                travel_times = []
                waiting_times = []
                for tl_id, times in simulation_skrl.history['travel_time'].items():
                    if times:
                        travel_times.extend(times[-10:])  # Last 10 measurements
                for tl_id, times in simulation_skrl.history['waiting_time'].items():
                    if times:
                        waiting_times.extend(times[-10:])  # Last 10 measurements
                
                avg_travel_time = sum(travel_times) / len(travel_times) if travel_times else 0
                avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Combined score (higher is better)
            # Weight: completion_rate (60%) - travel_time penalty (20%) - waiting_time penalty (20%)
            combined_score = (completion_rate * 0.6) - (avg_travel_time * 0.2) - (avg_waiting_time * 0.2)
            
            # Store performance data
            current_performance = {
                'episode': episode,
                'completion_rate': completion_rate,
                'avg_travel_time': avg_travel_time,
                'avg_waiting_time': avg_waiting_time,
                'combined_score': combined_score,
                'epsilon': epsilon,
                'total_departed': skrl_stats['total_departed'],
                'total_arrived': skrl_stats['total_arrived']
            }
            performance_history.append(current_performance)
            
            # Check if this is the best performance so far
            if combined_score > best_performance['combined_score']:
                # Clean up old best model files first
                model_save_name = config.get("save_model_name", "dqn_model")
                
                # Remove old best files if they exist
                if best_performance['episode'] != -1:  # If there was a previous best
                    old_best_model = path + f"{model_save_name}_BEST_ep{best_performance['episode']}.pth"
                    old_best_checkpoint = path + f"{model_save_name}_BEST_ep{best_performance['episode']}_checkpoint.pth"
                    
                    try:
                        if os.path.exists(old_best_model):
                            os.remove(old_best_model)
                            print(f"   🗑️  Removed old best model: ep{best_performance['episode']}")
                        if os.path.exists(old_best_checkpoint):
                            os.remove(old_best_checkpoint)
                            print(f"   🗑️  Removed old best checkpoint: ep{best_performance['episode']}")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove old best files: {e}")
                
                # Update best performance record
                best_performance = current_performance.copy()
                print(f"\n🏆 NEW BEST PERFORMANCE at Episode {episode}!")
                print(f"   Completion Rate: {completion_rate:.1f}%")
                print(f"   Avg Travel Time: {avg_travel_time:.2f}")
                print(f"   Avg Waiting Time: {avg_waiting_time:.2f}")
                print(f"   Combined Score: {combined_score:.2f}")
                
                # Create new best model files
                best_model_path = path + f"{model_save_name}_BEST_ep{episode}.pt"
                best_checkpoint_path = path + f"{model_save_name}_BEST_ep{episode}_checkpoint.pt"
                
                # Save SKRL model as best
                simulation_skrl.save_model(episode)
                print(f"   ✅ Best SKRL model saved: ep{episode}")
                
                # Always update the "CURRENT" best files (overwrite)
                current_best_model = path + f"{model_save_name}_BEST_CURRENT.pt"
                current_best_checkpoint = path + f"{model_save_name}_BEST_CURRENT_checkpoint.pt"
                
                simulation_skrl.save_model(episode)  # This will save with the episode number
                print(f"   ✅ Current best updated: ep{episode}")
            
            # Print current vs best performance every 10 episodes
            if episode % 10 == 0:
                print(f"\n📊 Performance Summary - Episode {episode}")
                print(f"Current: Score={combined_score:.2f}, Completion={completion_rate:.1f}%, Travel={avg_travel_time:.2f}, Wait={avg_waiting_time:.2f}")
                print(f"Best:    Score={best_performance['combined_score']:.2f}, Episode={best_performance['episode']}, Completion={best_performance['completion_rate']:.1f}%")
                
        except AttributeError as e:
            if 'num_atoms' in str(e):
                print(f"⚠️  Model compatibility error: {e}")
                print("   This error indicates the DQN model is missing the 'num_atoms' attribute for C51 distributional DQN.")
                print("   The model should be updated to include: self.num_atoms = 51")
            else:
                print(f"⚠️  Attribute error tracking performance: {e}")
            # Continue training even if performance tracking fails
        except Exception as e:
            print(f"⚠️  Error tracking performance: {e}")
            # Continue training even if performance tracking fails
        
        # Reset vehicle trackers after performance tracking is complete
        for loss_type, sim_dqn in simulations_dqn.items():
            print(f"  Resetting vehicle tracker for DQN simulation")
            sim_dqn.vehicle_tracker.reset()
            print(f"  Resetting history for DQN simulation")
            sim_dqn.reset_history()
            break  # Only need to do this once since there's only one DQN simulation

        # --- Save comparison plots ---
        print("Saving comparison plots...")
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            visualization.save_plot(
                episode=episode,
                metrics=[
                    "density_avg",
                    "travel_time_avg",
                    "outflow_avg",
                    "travel_delay_avg",
                    "waiting_time_avg",
                    "queue_length_avg",
                ],
                names=["skrl_dqn", "base"],  # Only include actually running simulations
            )
            print("Plots at episode", episode, "generated")

            # --- Generate traffic light comparison tables ---
            print("Generating traffic light comparison tables...")
            try:
                # Specify the actual simulation types that are running and saving data
                available_sim_types = ["baseline", "skrl_dqn"]  # Add q_learning when it's enabled
                comparison_results = comparison.save_comparison_tables(episode, simulation_types=available_sim_types)
                comparison.print_comparison_summary(episode, simulation_types=available_sim_types)
                print("Traffic light comparison tables generated successfully")
            except Exception as e:
                print(f"Error generating comparison tables: {e}")
                print(
                    "Comparison tables will be generated when CSV files are available"
                )

            # --- Generate vehicle comparison from logs ---
            print("Generating vehicle comparison from logs...")
            try:
                visualization.create_vehicle_comparison_from_logs(episode, ["skrl_dqn", "base"])  # Updated for SKRL
                print("Vehicle comparison from logs generated successfully")
            except Exception as e:
                print(f"Error generating vehicle comparison from logs: {e}")
                print("Vehicle comparison will be generated when log files are available")

        # Save model at specified intervals
        save_interval = config.get("save_interval", 10)  # Default to every 10 episodes
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

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)
    
    # Print best performance summary
    print(f"\n🏆 BEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Best Episode: {best_performance['episode']}")
    print(f"Best Completion Rate: {best_performance['completion_rate']:.2f}%")
    print(f"Best Travel Time: {best_performance['avg_travel_time']:.2f}")
    print(f"Best Waiting Time: {best_performance['avg_waiting_time']:.2f}")
    print(f"Best Combined Score: {best_performance['combined_score']:.2f}")
    print(f"Vehicles: {best_performance['total_arrived']}/{best_performance['total_departed']}")
    
    # Save performance history to CSV
    if performance_history:
        import pandas as pd
        performance_df = pd.DataFrame(performance_history)
        performance_csv = path + "performance_history.csv"
        performance_df.to_csv(performance_csv, index=False)
        print(f"📈 Performance history saved: {performance_csv}")
        
        # Create performance plot
        try:
            import matplotlib.pyplot as plt
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Performance Over Time', fontsize=16)
            
            episodes = performance_df['episode']
            
            # Plot 1: Completion Rate
            ax1.plot(episodes, performance_df['completion_rate'], 'b-', linewidth=2)
            ax1.axhline(y=best_performance['completion_rate'], color='r', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Completion Rate (%)')
            ax1.set_title('Vehicle Completion Rate')
            ax1.grid(True, alpha=0.3)
            ax1.text(0.02, 0.98, f"Best: {best_performance['completion_rate']:.1f}% (Ep {best_performance['episode']})", 
                     transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Plot 2: Combined Score
            ax2.plot(episodes, performance_df['combined_score'], 'g-', linewidth=2)
            ax2.axhline(y=best_performance['combined_score'], color='r', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Combined Score')
            ax2.set_title('Overall Performance Score')
            ax2.grid(True, alpha=0.3)
            ax2.text(0.02, 0.98, f"Best: {best_performance['combined_score']:.1f} (Ep {best_performance['episode']})", 
                     transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Plot 3: Travel Time
            ax3.plot(episodes, performance_df['avg_travel_time'], 'orange', linewidth=2)
            if best_performance['avg_travel_time'] > 0:
                ax3.axhline(y=best_performance['avg_travel_time'], color='r', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Avg Travel Time')
            ax3.set_title('Average Travel Time')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Epsilon Decay
            ax4.plot(episodes, performance_df['epsilon'], 'purple', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.set_title('Exploration Rate (Epsilon)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            performance_plot = path + "performance_history.png"
            plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Performance plot saved: {performance_plot}")
            
        except Exception as e:
            print(f"⚠️  Could not create performance plot: {e}")

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
    print(f"Final Q-table saved")
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

    for metric, data in avg_history.items():
        visualization.save_data_and_plot(
            data=data,
            filename=f"skrl_{metric}_avg",
            xlabel="Step",
            ylabel=metric.replace("_", " ").title(),
        )

    print("Plots generated")