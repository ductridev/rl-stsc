from src.utils import set_train_path, set_sumo, import_train_configuration
import traci
import numpy as np
import random
import time
from src.intersection import Intersection
from src.simulation import Simulation
from src.visualization import Visualization
from src.memory import ReplayMemory
from src.accident_manager import AccidentManager

config = import_train_configuration("config/training_westDragonBridge_cfg.yaml")

set_sumo(True, config["sumo_cfg_file"], config["max_steps"])
config = import_train_configuration("config/training_westDragonBridge_cfg.yaml")
green_duration_deltas = config["agent"]["green_duration_deltas"]
min_epsilon = config["agent"]["min_epsilon"]
decay_rate = config["agent"]["decay_rate"]
start_step = config["start_step"]
duration = config["duration"]
junction_id_list = config["junction_id_list"]
epsilon = 1

agent_memory = ReplayMemory(
    max_size=config["memory_size_max"], min_size=config["memory_size_min"]
)

path = set_train_path(config["models_path_name"])

visualization = Visualization(path=path, dpi=100)

accident_manager= AccidentManager(
    start_step= start_step, duration= duration, junction_id_list=junction_id_list
)

# Initialize simulation
simulation = Simulation(
    memory=agent_memory,
    visualization=visualization,
    agent_cfg=config["agent"],
    max_steps=config["max_steps"],
    traffic_lights=config["traffic_lights"],
    interphase_duration=config["interphase_duration"],
    accident_manager= accident_manager,
    epoch=config["training_epochs"],
    path=path,
)




Intersection.generate_routes(
    config["sumo_cfg_file"].split("/")[1],
    enable_bicycle=True,
    enable_pedestrian=True,
    enable_motorcycle=True,
    enable_passenger=True,
)

episode = 0
simulation_time, training_time = simulation.run(epsilon, episode)
epsilon = max(min_epsilon, epsilon * decay_rate)
print(
    "Simulation time:",
    simulation_time,
    "s - Training time:",
    training_time,
    "s - Total:",
    round(simulation_time + training_time, 1),
    "s",
)