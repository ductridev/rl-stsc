from src.memory import ReplayMemory
from src.utils import set_train_path, set_sumo
from src.model import DQN
from src.utils import import_train_configuration
from src.simulation import Simulation
from src.visualization import Visualization
from src.intersection import Intersection

import traci

import datetime

if __name__ == "__main__":
    config = import_train_configuration('config/training_westDragonBridge_cfg.yaml')

    green_duration_agent_memory = ReplayMemory(max_size=config['memory_size_max'], min_size=config['memory_size_min'])
    selector_phase_agent_memory = ReplayMemory(max_size=config['memory_size_max'], min_size=config['memory_size_min'])

    path = set_train_path(config['models_path_name'])

    green_duration_agent = DQN(
        num_layers=config['green_duration_agent']['num_layers'],
        batch_size=config['green_duration_agent']['batch_size'],
        learning_rate=config['green_duration_agent']['learning_rate'],
        input_dim=config['green_duration_agent']['num_states'],
        output_dim=config['green_duration_agent']['num_actions'],
        gamma=config['green_duration_agent']['gamma'],
    )

    selector_phase_agent = DQN(
        num_layers=config['selector_phase_agent']['num_layers'],
        batch_size=config['selector_phase_agent']['batch_size'],
        learning_rate=config['selector_phase_agent']['learning_rate'],
        input_dim=config['selector_phase_agent']['num_states'],
        output_dim=config['selector_phase_agent']['num_actions'],
        gamma=config['selector_phase_agent']['gamma'],
    )

    # Initialize the simulation
    simulation = Simulation(
        green_duration_agent=green_duration_agent,
        selector_phase_agent=selector_phase_agent,
        green_duration_agent_memory=green_duration_agent_memory,
        selector_phase_agent_memory=selector_phase_agent_memory,
        green_duration_agent_cfg=config['green_duration_agent'],
        selector_phase_agent_cfg=config['selector_phase_agent'],
        max_steps=config['max_steps'],
        traffic_lights=config['traffic_lights'],
        interphase_duration=config['interphase_duration'],
        epoch=config['training_epochs'],
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        print("Generating routes...")
        # Run the build routes file command

        # Intersection.generate_routes(config['sumo_cfg_file'].split("/")[1], enable_bicycle=True, enable_pedestrian=True, enable_motorcycle=True, enable_passenger=True)
        print("Routes generated")

        sumo_cmd = set_sumo(config['gui'], config['sumo_cfg_file'], config['max_steps'])

        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')

        traci.close()
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    green_duration_agent.save(path + '/green_duration_agent.pth')
    selector_phase_agent.save(path + '/selector_phase_agent.pth')
