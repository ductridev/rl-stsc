from src.memory import ReplayMemory
from src.utils import set_train_path, set_sumo, import_train_configuration
from src.model import DQN
from src.simulation import Simulation
from src.visualization import Visualization
from src.intersection import Intersection

import traci
import datetime

if __name__ == "__main__":
    # Load configuration
    config = import_train_configuration('config/training_westDragonBridge_cfg.yaml')
    green_duration_deltas = config['agent']['green_duration_deltas']

    # Create replay memory for the agent
    agent_memory = ReplayMemory(
        max_size=config['memory_size_max'],
        min_size=config['memory_size_min']
    )

    # Set model save path
    path = set_train_path(config['models_path_name'])

    visualization = Visualization(path=path, dpi=100)

    # Initialize simulation
    simulation = Simulation(
        memory=agent_memory,
        agent_cfg=config['agent'],
        max_steps=config['max_steps'],
        traffic_lights=config['traffic_lights'],
        interphase_duration=config['interphase_duration'],
        epoch=config['training_epochs'],
        path=path
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        print("Generating routes...")
        # Run the build routes file command

        # Intersection.generate_routes(config['sumo_cfg_file'].split("/")[1], enable_bicycle=True, enable_pedestrian=True, enable_motorcycle=True, enable_passenger=True)
        print("Routes generated")

        set_sumo(config['gui'], config['sumo_cfg_file'], config['max_steps'])

        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(epsilon, episode)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')

        traci.close(False)
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    visualization.save_data_and_plot(simulation.agent_reward, 'reward', 'Episode', 'Reward')
    visualization.save_data_and_plot(simulation.travel_speed, 'travel_speed', 'Episode', 'Travel speed')
    visualization.save_data_and_plot(simulation.travel_time, 'travel_time', 'Episode', 'Travel time')
    visualization.save_data_and_plot(simulation.density, 'density', 'Episode', 'Density')
    visualization.save_data_and_plot(simulation.outflow_rate, 'outflow_rate', 'Episode', 'Outflow rate')
    visualization.save_data_and_plot(simulation.green_time, 'green_time', 'Episode', 'Green time')