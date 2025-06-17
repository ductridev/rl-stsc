from src.desra import DESRA
from src.utils import import_train_configuration, set_sumo
from src.intersection import Intersection

import traci

config = import_train_configuration("config/training_westDragonBridge_cfg.yaml")

desra = DESRA(interphase_duration=5)

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

set_sumo(config["gui"], config["sumo_cfg_file"], config["max_steps"])

step = 0

while step < 3600:
    if step % 50 == 0:
        for traffic_light in config["traffic_lights"]:
            best_phase, best_green_time = desra.select_phase(traffic_light)
            print("Best phase:", best_phase, "Best green time:", best_green_time)
    step += 1
    traci.simulationStep()