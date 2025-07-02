from src.memory import ReplayMemory
from src.utils import set_train_path, set_sumo, import_train_configuration
from src.model import DQN
from src.simulation import Simulation
from src.Qlearning import QSimulation
from src.base_simulation import SimulationBase
from src.visualization import Visualization
from src.intersection import Intersection
from src.accident_manager import AccidentManager
import datetime

if __name__ == "__main__":
    # Load configuration
    config = import_train_configuration("config/training_testngatu.yaml")
    green_duration_deltas = config["agent"]["green_duration_deltas"]
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
    simulation_dqn = Simulation(
        memory=agent_memory_dqn,
        visualization=visualization,
        agent_cfg=config["agent"],
        max_steps=config["max_steps"],
        traffic_lights=config["traffic_lights"],
        interphase_duration=config["interphase_duration"],
        accident_manager= accident_manager,
        epoch=config["training_epochs"],
        path=path,
    )

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
    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config["total_episodes"]:
        print("\n----- Episode", str(episode + 1), "of", str(config["total_episodes"]))
        print("Generating routes...")
        # Run the build routes file command

        Intersection.generate_routes(
            config["sumo_cfg_file"].split("/")[1],
            enable_bicycle=True,
            enable_pedestrian=True,
            enable_motorcycle=True,
            enable_passenger=True,
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

        print("Running Simulation (DQN)...")
        set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])
        simulation_time_dqn, training_time_dqn = simulation_dqn.run(epsilon, episode)
        print("Simulation (DQN) time:", simulation_time_dqn, "Training time:", training_time_dqn)

        epsilon = max(min_epsilon, epsilon * decay_rate)

        # --- Save comparison plots ---
        print("Saving comparison plots...")
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            visualization.save_plot(
                episode=episode,
                metrics=["density_avg", "green_time_avg", "travel_time_avg", "outflow_rate_avg", "travel_speed_avg", "agent_reward_avg"],
                names=["dqn", "q", "base"],
            )
            print("Plots at episode", episode, "generated")
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    print(f"Saving models at {path}...")
    model_path = path + f"model.pth"
    simulation.agent.save(model_path)
    print("Models saved")
    print("---------------------------------------")

    print("Generating plots...")
    # We simple by averaging the history over all traffic lights
    avg_history = {}

    for metric, data_per_tls in simulation.history.items():
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
