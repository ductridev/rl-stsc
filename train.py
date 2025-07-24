from src.memory import ReplayMemory
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
    generate_random_intervals,
    save_to_same_dir_as_cfg,
)
import datetime

if __name__ == "__main__":
    # Load configuration
    config = import_train_configuration("config/training_testngatu4.yaml")
    min_epsilon = config["agent"]["min_epsilon"]
    decay_rate = config["agent"]["decay_rate"]
    epsilon = 1

    # Create replay memory for the agent
    agent_memory_dqn = ReplayMemory(
        max_size=config["memory_size_max"], min_size=config["memory_size_min"]
    )

    agent_memory_q = ReplayMemory(
        max_size=config["memory_size_max"], min_size=config["memory_size_min"]
    )

    # Set model save path
    path = set_train_path(config["models_path_name"])

    visualization = Visualization(path=path, dpi=100)

    # Initialize accident manager
    accident_manager = AccidentManager(
        start_step=config["start_step"],
        duration=config["duration"],
        junction_id_list=config["junction_id_list"],
    )

    # Initialize simulation
    simulations_dqn = {
        loss: Simulation(
            memory=agent_memory_dqn,
            visualization=visualization,
            agent_cfg={**config["agent"], "loss_type": loss},
            max_steps=config["max_steps"],
            traffic_lights=config["traffic_lights"],
            interphase_duration=config["interphase_duration"],
            accident_manager=accident_manager,
            epoch=config["training_epochs"],
            path=path,
        )
        for loss in config["agent"]["model"]["loss_type"]
    }

    # Load existing model if specified in config
    start_episode = 0
    start_epsilon = epsilon

    # if config.get("load_model_name") and config["load_model_name"] is not None:
    #     model_load_path = set_load_model_path(config["models_path_name"]) + config["load_model_name"] + ".pth"
    #     checkpoint_load_path = set_load_model_path(config["models_path_name"]) + config["load_model_name"] + "_checkpoint.pth"

    #     # Try to load checkpoint first (for continuing training)
    #     try:
    #         training_state = simulation_dqn.agent.load_checkpoint(checkpoint_load_path)
    #         start_episode = training_state.get('episode', 0)
    #         start_epsilon = training_state.get('epsilon', epsilon)
    #         print(f"Successfully loaded checkpoint from: {checkpoint_load_path}")
    #         print(f"Resuming from episode {start_episode} with epsilon {start_epsilon}")
    #     except FileNotFoundError:
    #         # If checkpoint not found, try to load just the model weights
    #         try:
    #             simulation_dqn.agent.load(model_load_path, for_training=True)
    #             print(f"Successfully loaded model weights from: {model_load_path}")
    #             print("Starting training from episode 0 with fresh optimizer state")
    #         except FileNotFoundError:
    #             print(f"Model file not found: {model_load_path}. Starting with fresh model.")
    #         except Exception as e:
    #             print(f"Error loading model: {e}. Starting with fresh model.")
    #     except Exception as e:
    #         print(f"Error loading checkpoint: {e}. Starting with fresh model.")

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

    while episode < config["total_episodes"]:
        print("\n----- Episode", str(episode + 1), "of", str(config["total_episodes"]))
        print("Generating routes...")
        # Run the build routes file command

        edge_data = generate_random_intervals(
            total_duration=3600,
            min_interval=600,
            max_interval=1800,
            base_weight=100.0,
            high_min=200.0,
            high_max=500.0,
            min_active_sides=1,
            max_active_sides=3,
            edge_groups=config["edge_groups"],
        )

        random_demand_name = save_to_same_dir_as_cfg(edge_data, config["sumo_cfg_file"])

        Intersection.generate_residential_demand_routes(
            config,
            config["sumo_cfg_file"].split("/")[1],
            demand_level="low",
            enable_motorcycle=True,
            enable_passenger=True,
            enable_truck=True,
            enable_bicycle=True,
            enable_pedestrian=True,
            random_demand_name=random_demand_name,
        )
        print("Routes generated")

        # --- Run all three simulations ---
        print("Running SimulationBase (static baseline)...")
        set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
        simulation_time_base = simulation.run(episode)
        print("SimulationBase time:", simulation_time_base)

        print("Running QSimulation (Q-learning)...")
        set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
        simulation_time_q = simulation_q.run(epsilon, episode)
        print("QSimulation time:", simulation_time_q)

        for loss_type, sim_dqn in simulations_dqn.items():
            print(f"Running DQN Simulation (loss: {loss_type})...")
            set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
            simulation_time_dqn, training_time_dqn = sim_dqn.run(epsilon, episode)
            print(
                f"Simulation (DQN - {loss_type}) time:",
                simulation_time_dqn,
                "Training time:",
                training_time_dqn,
            )

        epsilon = max(min_epsilon, epsilon * decay_rate)

        # --- Save comparison plots ---
        print("Saving comparison plots...")
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            visualization.save_plot(
                episode=episode,
                metrics=[
                    "density_avg",
                    "green_time_avg",
                    "travel_time_avg",
                    "outflow_avg",
                    "travel_speed_avg",
                    "waiting_time_avg",
                    "queue_length_avg",
                ],
                names=["dqn_qr", "dqn_mse", "dqn_huber", "dqn_weighted", "q", "base"],
            )
            print("Plots at episode", episode, "generated")

            # --- Generate traffic light comparison tables ---
            print("Generating traffic light comparison tables...")
            try:
                comparison_results = comparison.save_comparison_tables(episode)
                comparison.print_comparison_summary(episode)
                print("Traffic light comparison tables generated successfully")
            except Exception as e:
                print(f"Error generating comparison tables: {e}")
                print(
                    "Comparison tables will be generated when CSV files are available"
                )

        # Save model at specified intervals
        save_interval = config.get("save_interval", 10)  # Default to every 10 episodes
        if episode % save_interval == 0 and episode > 0:
            model_save_name = config.get("save_model_name", "dqn_model")

            # # Save model weights only
            # model_save_path = path + f"{model_save_name}_episode_{episode}.pth"
            # simulation_dqn.agent.save(model_save_path)

            # # Save complete checkpoint for continuing training
            # checkpoint_save_path = path + f"{model_save_name}_episode_{episode}_checkpoint.pth"
            # simulation_dqn.agent.save_checkpoint(checkpoint_save_path, episode=episode, epsilon=epsilon)

            # print(f"DQN model saved at episode {episode}: {model_save_path}")
            # print(f"DQN checkpoint saved at episode {episode}: {checkpoint_save_path}")

            # Also save Q-learning Q-table
            q_table_save_path = (
                path + f"q_table_{model_save_name}_episode_{episode}.pkl"
            )
            simulation_q.save_q_table(path, episode)

        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

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
    # We simple by averaging the history over all traffic lights
    avg_history = {}

    for metric, data_per_tls in simulations_dqn.history.items():
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
            filename=f"{metric}_avg",
            xlabel="Step",
            ylabel=metric.replace("_", " ").title(),
        )

    print("Plots generated")
