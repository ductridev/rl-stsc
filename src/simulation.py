from src.model import DQN
from src.memory import ReplayMemory
from src.visualization import Visualization
from src.normalizer import Normalizer
from src.desra import DESRA
from src.sumo import SUMO
from src.accident_manager import AccidentManager
import torch.nn.functional as F
import traci
import numpy as np
import random
import torch
import time
import torch.nn as nn
import copy

GREEN_ACTION = 0
RED_ACTION = 1


def index_to_action(index, actions_map):
    return actions_map[index]["phase"]


def phase_to_index(phase, actions_map, duration):
    for i, action in actions_map.items():
        if action["phase"] == phase:
            return i


class Simulation(SUMO):
    def __init__(
        self,
        memory: ReplayMemory,
        visualization: Visualization,
        agent_cfg,
        max_steps,
        traffic_lights,
        accident_manager: AccidentManager,
        interphase_duration=3,
        epoch=1000,
        path=None,
    ):
        self.memory = memory
        self.visualization = visualization
        self.agent_cfg = agent_cfg
        self.loss_type = agent_cfg["loss_type"]
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.interphase_duration = interphase_duration
        self.epoch = epoch
        self.path = path
        self.weight = agent_cfg["weight"]
        self.outflow_rate_normalizer = Normalizer()
        self.density_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_speed_normalizer = Normalizer()

        self.step = 0
        self.num_actions = {}
        self.actions_map = {}

        self.agent_reward = {}
        self.agent_state = {}
        self.agent_old_state = {}
        self.agent_memory = {}

        self.outflow_rate = {}
        self.old_outflow_rate = {}
        self.travel_speed = {}
        self.travel_time = {}
        self.density = {}
        self.old_travel_speed = {}
        self.old_travel_time = {}
        self.old_density = {}
        self.green_time = {}

        self.history = {
            "agent_reward": {},
            "travel_speed": {},
            "travel_time": {},
            "density": {},
            "outflow": {},
            "q_value": {},
            "max_next_q_value": {},
            "target": {},
            "loss": {},
            "queue_length": {},
            "waiting_time": {},
        }

        self.initState()

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent: DQN = DQN(
            num_layers=self.agent_cfg["num_layers"],
            batch_size=self.agent_cfg["batch_size"],
            learning_rate=self.agent_cfg["learning_rate"],
            input_dim=self.agent_cfg["num_states"],
            output_dims=self.get_output_dims(),
            gamma=self.agent_cfg["gamma"],
            device=self.device,
            loss_type=self.loss_type,
        )

        self.desra = DESRA(interphase_duration=self.interphase_duration)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.agent = nn.DataParallel(self.agent)

        self.agent = self.agent.to(self.device)
        self.target_net = copy.deepcopy(self.agent)
        self.target_net.eval()

        self.longest_phase, self.longest_phase_len, self.longest_phase_id = (
            self.get_longest_phase()
        )

    def get_longest_phase(self):
        max_len = -1
        longest_phase = None
        longest_id = None

        for tl_id, actions in self.actions_map.items():
            for action in actions.values():
                phase = action["phase"]
                if len(phase) > max_len:
                    max_len = len(phase)
                    longest_phase = phase
                    longest_id = tl_id

        return longest_phase, max_len, longest_id

    def initState(self):
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            self.agent_memory[traffic_light_id] = self.memory

            # Initialize the green time
            self.green_time[traffic_light_id] = 20

            # Initialize the number of actions
            self.num_actions[traffic_light_id] = len(traffic_light["phase"])

            # Initialize the action map
            self.actions_map[traffic_light_id] = {}

            # We will convert from number of phases and green delta to phase index and green delta
            i = 0

            for phase in traffic_light["phase"]:
                self.actions_map[traffic_light_id][i] = {
                    "phase": phase,
                }

                i += 1

            # Initialize the agent reward
            self.agent_reward[traffic_light_id] = 0

            # Initialize the agent state
            self.agent_state[traffic_light_id] = 0

            # Initialize the agent old state
            self.agent_old_state[traffic_light_id] = 0

            # Initialize the old outflow rate
            self.old_outflow_rate[traffic_light_id] = 0

            # Initialize the outflow rate
            self.outflow_rate[traffic_light_id] = 0

            # Initialize the old travel speed
            self.old_travel_speed[traffic_light_id] = 0

            # Initialize the travel speed
            self.travel_speed[traffic_light_id] = 0

            # Initialize the old travel time
            self.old_travel_time[traffic_light_id] = 0

            # Initialize the travel time
            self.travel_time[traffic_light_id] = 0

            # Initialize the old density
            self.old_density[traffic_light_id] = 0

            # Initialize the density
            self.density[traffic_light_id] = 0

            # Initialize the history
            for key in self.history:
                self.history[key][traffic_light_id] = []

    def get_output_dims(self):
        """
        Get multiple outputs of the agent for each traffic light

        Returns:
            output_dims (list[int]): The output dimension of the agent
        """

        output_dims = []
        for traffic_light in self.traffic_lights:
            output_dims.append(self.num_actions[traffic_light["id"]])

        return output_dims

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start = time.time()

        # Initialize per-light state tracking
        tl_states = {}
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            tl_states[tl_id] = {
                "green_time_remaining": 0,
                "travel_speed_sum": 0,
                "travel_time_sum": 0,
                "density_sum": 0,
                "outflow": 0,
                "old_vehicle_ids": [],
                "state": None,
                "action_idx": None,
                "phase": None,
                "green_time": 0,
                "step_travel_speed_sum": 0,
                "step_travel_time_sum": 0,
                "step_density_sum": 0,
                "step_outflow": 0,
                "step_queue_length": 0,
                "step_waiting_time": 0,
                "step_old_vehicle_ids": [],
            }

        num_vehicles = 0

        while self.step < self.max_steps:
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                # If time to choose a new action
                if tl_state["green_time_remaining"] <= 0:
                    state, desra_greens = self.get_state(tl)
                    action_idx, predicted_green = self.select_action(
                        tl_id,
                        self.agent,
                        state,
                        desra_greens,
                        epsilon,
                    )

                    phase = index_to_action(
                        action_idx,
                        self.actions_map[tl_id],
                    )

                    green_time = max(
                        1, min(predicted_green, self.max_steps - self.step)
                    )

                    if tl_state["phase"] is not None:
                        self.set_yellow_phase(tl_id, tl_state["phase"])
                        for _ in range(self.interphase_duration):
                            self.accident_manager.create_accident(
                                current_step=self.step
                            )
                            traci.simulationStep()
                            self.step += 1

                            # Calculate step metrics during interphase for all traffic lights
                            for tl_inner in self.traffic_lights:
                                tl_inner_id = tl_inner["id"]
                                tl_inner_state = tl_states[tl_inner_id]

                                step_new_vehicle_ids = self.get_vehicles_in_phase(
                                    tl_inner,
                                    (
                                        tl_inner_state["phase"]
                                        if tl_inner_state["phase"] is not None
                                        else 0
                                    ),
                                )
                                step_outflow = sum(
                                    1
                                    for vid in tl_inner_state["step_old_vehicle_ids"]
                                    if vid not in step_new_vehicle_ids
                                )
                                tl_inner_state[
                                    "step_travel_speed_sum"
                                ] += self.get_avg_speed(tl_inner)
                                tl_inner_state[
                                    "step_travel_time_sum"
                                ] += self.get_avg_travel_time(tl_inner)
                                tl_inner_state[
                                    "step_density_sum"
                                ] += self.get_avg_density(tl_inner)
                                tl_inner_state["step_outflow"] += step_outflow
                                tl_inner_state[
                                    "step_queue_length"
                                ] += self.get_avg_queue_length(tl_inner)
                                tl_inner_state[
                                    "step_waiting_time"
                                ] += self.get_avg_waiting_time(tl_inner)
                                tl_inner_state["step_old_vehicle_ids"] = (
                                    step_new_vehicle_ids
                                )

                                # Check if it's time to save data (dynamic interval for exactly 60 data points)
                                if self.step % 60 == 0:
                                    travel_speed_avg = (
                                        tl_inner_state["step_travel_speed_sum"] / 60
                                    )
                                    travel_time_avg = (
                                        tl_inner_state["step_travel_time_sum"] / 60
                                    )
                                    density_avg = (
                                        tl_inner_state["step_density_sum"] / 60
                                    )
                                    outflow_avg = tl_inner_state["step_outflow"]
                                    queue_length_avg = (
                                        tl_inner_state["step_queue_length"] / 60
                                    )
                                    waiting_time_avg = (
                                        tl_inner_state["step_waiting_time"] / 60
                                    )

                                    self.history["travel_speed"][tl_inner_id].append(
                                        travel_speed_avg
                                    )
                                    self.history["travel_time"][tl_inner_id].append(
                                        travel_time_avg
                                    )
                                    self.history["density"][tl_inner_id].append(
                                        density_avg
                                    )
                                    self.history["outflow"][tl_inner_id].append(
                                        outflow_avg
                                    )
                                    self.history["queue_length"][tl_inner_id].append(
                                        queue_length_avg
                                    )
                                    self.history["waiting_time"][tl_inner_id].append(
                                        waiting_time_avg
                                    )

                                    # Reset step metrics
                                    tl_inner_state["step_travel_speed_sum"] = 0
                                    tl_inner_state["step_travel_time_sum"] = 0
                                    tl_inner_state["step_density_sum"] = 0
                                    tl_inner_state["step_outflow"] = 0
                                    tl_inner_state["step_queue_length"] = 0
                                    tl_inner_state["step_waiting_time"] = 0

                    self.set_green_phase(tl_id, green_time, phase)

                    tl_state.update(
                        {
                            "green_time": green_time,
                            "green_time_remaining": green_time,
                            "travel_speed_sum": 0,
                            "density_sum": 0,
                            "outflow": 0,
                            "old_vehicle_ids": self.get_vehicles_in_phase(tl, phase),
                            "state": state,
                            "action_idx": action_idx,
                            "phase": phase,
                        }
                    )

            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            self.step += 1
            self.desra.update_traffic_parameters()

            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                # save plot every dynamic interval
                if self.step % 60 == 0:
                    travel_speed_avg = tl_state["step_travel_speed_sum"] / 60
                    travel_time_avg = tl_state["step_travel_time_sum"] / 60
                    density_avg = tl_state["step_density_sum"] / 60
                    outflow_avg = tl_state["step_outflow"]
                    queue_length_avg = tl_state["step_queue_length"] / 60
                    waiting_time_avg = tl_state["step_waiting_time"] / 60

                    self.history["travel_speed"][tl_id].append(travel_speed_avg)
                    self.history["travel_time"][tl_id].append(travel_time_avg)
                    self.history["density"][tl_id].append(density_avg)
                    self.history["outflow"][tl_id].append(outflow_avg)
                    self.history["queue_length"][tl_id].append(queue_length_avg)
                    self.history["waiting_time"][tl_id].append(waiting_time_avg)

                    # Reset step metrics
                    tl_state["step_travel_speed_sum"] = 0
                    tl_state["step_travel_time_sum"] = 0
                    tl_state["step_density_sum"] = 0
                    tl_state["step_outflow"] = 0
                    tl_state["step_queue_length"] = 0
                    tl_state["step_waiting_time"] = 0

                if tl_state["green_time_remaining"] > 0:
                    phase = tl_state["phase"]
                    new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                    outflow = sum(
                        1
                        for vid in tl_state["old_vehicle_ids"]
                        if vid not in new_vehicle_ids
                    )
                    tl_state["outflow"] += outflow
                    tl_state["travel_speed_sum"] += self.get_avg_speed(tl)
                    tl_state["travel_time_sum"] += self.get_avg_travel_time(tl)
                    tl_state["density_sum"] += self.get_avg_density(tl)
                    tl_state["old_vehicle_ids"] = new_vehicle_ids
                    tl_state["green_time_remaining"] -= 1

                    if tl_state["green_time_remaining"] <= 0:
                        green_time = tl_state["green_time"]

                        self.outflow_rate[tl_id] = tl_state["outflow"] / green_time
                        self.travel_speed[tl_id] = (
                            tl_state["travel_speed_sum"] / green_time
                        )
                        self.travel_time[tl_id] = (
                            tl_state["travel_time_sum"] / green_time
                        )
                        self.density[tl_id] = tl_state["density_sum"] / green_time

                        reward = self.get_reward(tl_id)
                        next_state, _ = self.get_state(tl)
                        done = self.step >= self.max_steps

                        self.agent_memory[tl_id].push(
                            (
                                tl_state["state"],
                                tl_state["action_idx"],
                                tl_state["green_time"],
                            ),
                            reward,
                            (next_state, None),
                            done,
                        )
                        self.history["agent_reward"][tl_id].append(reward)
                        self.green_time[tl_id] = green_time
                # step state metrics
                step_new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                step_outflow = sum(
                    1
                    for vid in tl_state["step_old_vehicle_ids"]
                    if vid not in step_new_vehicle_ids
                )
                tl_state["step_travel_speed_sum"] += self.get_avg_speed(tl)
                tl_state["step_travel_time_sum"] += self.get_avg_travel_time(tl)
                tl_state["step_density_sum"] += self.get_avg_density(tl)
                tl_state["step_outflow"] += step_outflow
                tl_state["step_queue_length"] += self.get_avg_queue_length(tl)
                tl_state["step_waiting_time"] += self.get_avg_waiting_time(tl)
                tl_state["step_old_vehicle_ids"] = step_new_vehicle_ids

        simulation_time = time.time() - start

        traci.close()

        print("Simulation ended")
        print("Number of vehicles:", num_vehicles)
        print("---------------------------------------")

        print("Training...")
        start_time = time.time()

        if episode % 10 == 0:
            self.target_net.load_state_dict(self.agent.state_dict())

        for _ in range(self.epoch):
            self.training()
        training_time = time.time() - start_time

        print("Training ended")
        print("---------------------------------------")

        self.save_plot(episode=episode)
        self.step = 0

        total_reward = 0
        for traffic_light in self.traffic_lights:
            total_reward += np.sum(self.history["agent_reward"][traffic_light["id"]])

        print("Total reward:", total_reward, "- Epsilon:", epsilon)
        self.reset_history()
        return simulation_time, training_time

    def training(self):
        """
        Retrieve a batch from each traffic light memory and train the agent.
        """
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            batch = self.agent_memory[traffic_light_id].get_samples(
                self.agent.batch_size
            )

            if len(batch) > 0:
                state_data, rewards, next_state_data, dones = zip(*batch)
                states, actions, green_times = zip(*state_data)
                next_states, _ = zip(*next_state_data)

                metrics = self.agent.train_batch(
                    states,
                    actions,
                    rewards,
                    next_states,
                    output_dim=self.num_actions[traffic_light_id],
                    done=dones,
                    green_targets=green_times,
                    target_net=self.target_net,
                )

                self.history["q_value"][traffic_light_id].append(metrics["avg_q_value"])
                self.history["max_next_q_value"][traffic_light_id].append(
                    metrics["avg_max_next_q_value"]
                )
                self.history["target"][traffic_light_id].append(metrics["avg_target"])
                self.history["loss"][traffic_light_id].append(metrics["loss"])

    def get_reward(self, traffic_light_id):
        return (
            # self.weight["green_time"]
            # * self.green_time_normalizer.normalize(
            #     self.green_time_old[traffic_light_id]
            #     - self.green_time[traffic_light_id]
            # )
            self.weight["outflow_rate"]
            * self.outflow_rate_normalizer.normalize(
                self.outflow_rate[traffic_light_id]
                - self.old_outflow_rate[traffic_light_id]
            )
            + self.weight["travel_speed"]
            * self.travel_speed_normalizer.normalize(
                self.travel_speed[traffic_light_id]
                - self.old_travel_speed[traffic_light_id]
            )
            + self.weight["travel_time"]
            * self.travel_time_normalizer.normalize(
                self.old_travel_time[traffic_light_id]
                - self.travel_time[traffic_light_id]
            )
            + self.weight["density"]
            * self.density_normalizer.normalize(
                self.old_density[traffic_light_id] - self.density[traffic_light_id]
            )
        )

    def save_plot(self, episode):

        # Average history over all traffic lights
        avg_history = {}
        for metric, data_per_tls in self.history.items():
            data_lists = [data for data in data_per_tls.values() if len(data) > 0]
            if not data_lists:
                continue
            min_length = min(len(data) for data in data_lists)
            data_lists = [data[:min_length] for data in data_lists]
            avg_data = [
                sum(step_vals) / len(step_vals) for step_vals in zip(*data_lists)
            ]
            avg_history[metric] = avg_data

        # Save and plot averaged metrics
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data(
                    data=data,
                    filename=f"dqn_{self.loss_type}_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                )
            print("Plots at episode", episode, "generated")
            print("---------------------------------------")
            # Reset history after saving plots

    def get_yellow_phase(self, green_phase):
        """
        Convert a green phase string to a yellow phase string by replacing all 'G' with 'y'.
        """
        return green_phase.replace("G", "y")

    def set_yellow_phase(self, tlsId, green_phase):
        """
        Set the traffic light to yellow phase by converting green phase string to yellow.
        """
        yellow_phase = self.get_yellow_phase(green_phase)
        traci.trafficlight.setPhaseDuration(tlsId, 3)
        traci.trafficlight.setRedYellowGreenState(tlsId, yellow_phase)

    def set_green_phase(self, tlsId, duration, new_phase):
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)

    def update_phase(self, phase_idxs, num_phases):
        """
        Update the phase of the traffic light
        """
        phase_str = num_phases * ["r"]
        return self.replace_chars(phase_str, phase_idxs, "G")

    def replace_chars(self, s, indexes, replacements):
        s_list = list(s)

        if isinstance(replacements, str):
            # One replacement character for all indices
            for i in indexes:
                if 0 <= i < len(s_list):
                    s_list[i] = replacements
        else:
            # Each index has a corresponding replacement
            for i, c in zip(indexes, replacements):
                if 0 <= i < len(s_list):
                    s_list[i] = c

        return "".join(s_list)

    def select_action(
        self,
        traffic_light_id: str,
        agent: DQN,
        base_state: np.ndarray,
        desra_green_times: list,
        epsilon: float,
    ):
        """
        Select an action using epsilon-greedy policy with DESRA green time hints.

        Args:
            traffic_light_id (str): ID of the traffic light
            agent (DQN): The DQN model
            base_state (np.ndarray): Base traffic state (e.g. queue, arrivals)
            desra_green_times (list[float]): DESRA suggested green times per phase
            epsilon (float): Epsilon-greedy exploration factor

        Returns:
            (int, float): Tuple of selected phase index and predicted green time
        """
        num_actions = self.num_actions[traffic_light_id]

        # Add DESRA green hints to the state
        state_t = torch.from_numpy(base_state).to(self.device, dtype=torch.float32)

        if random.random() < epsilon:
            action_idx = random.randint(0, num_actions - 1)
            predicted_green = desra_green_times[action_idx]  # fallback to DESRA
            return action_idx, predicted_green

        with torch.no_grad():
            q_values, green_times = agent.predict_one(state_t, output_dim=num_actions)
            if agent.loss_type == "qr":
                q_values = q_values.mean(2)  # [1, A]

            best_action_idx = q_values.squeeze(0).argmax().item()
            predicted_green_time = green_times.squeeze(0)[best_action_idx].item()

            return best_action_idx, predicted_green_time

    def get_state(self, traffic_light):
        """
        Get the current state at a specific traffic light in the simulation.

        Returns:
            np.ndarray: 1D array representing the full input state
            int: green time
        """
        state_vector = []

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            # Skip phases with no movements (safety check)
            if not movements:
                continue

            free_capacity_sum = 0
            density_sum = 0
            max_waiting_time = 0
            max_queue_length = 0

            for detector_id in movements:
                free_capacity_sum += self.get_free_capacity(detector_id)
                density_sum += self.get_density(detector_id)
                max_waiting_time = max(
                    max_waiting_time, self.get_waiting_time(detector_id)
                )
                max_queue_length = max(
                    max_queue_length, self.get_queue_length(detector_id)
                )

            movement_count = len(movements)
            avg_free_capacity = free_capacity_sum / movement_count
            avg_density = density_sum / movement_count

            # Append per-phase state: [free_capacity, density, waiting_time, queue_length]
            state_vector.extend(
                [avg_free_capacity, avg_density, max_waiting_time, max_queue_length]
            )

        # DESRA recommended phase and green time
        best_phase, desra_green, desra_green_list = (
            self.desra.select_phase_with_desra_hints(traffic_light)
        )

        padding = self.agent._input_dim - len(state_vector) - len(desra_green_list) - 2
        state_vector.extend([0] * padding)

        # Convert phase to index
        desra_phase_idx = phase_to_index(
            best_phase, self.actions_map[traffic_light["id"]], 0
        )

        # Append DESRA guidance
        state_vector.extend(desra_green_list)
        state_vector.extend([desra_phase_idx, desra_green])

        return np.array(state_vector, dtype=np.float32), desra_green_list

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_free_capacity(self, detector_id):
        """
        Get the free capacity of a lane.
        """
        lane_length = traci.lanearea.getLength(detector_id)
        occupied_length = (
            traci.lanearea.getLastStepOccupancy(detector_id) / 100 * lane_length
        )

        return (lane_length - occupied_length) / lane_length

    def get_density(self, detector_id):
        """
        Get the density of vehicles on a lane.
        """
        return traci.lanearea.getLastStepOccupancy(detector_id) / 100

    def get_waiting_time(self, detector_id):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    def get_avg_speed(self, traffic_light):
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                speeds.append(speed)
            except:
                pass
        return np.mean(speeds) if speeds else 0.0

    def get_avg_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.mean(travel_times) if travel_times else 0.0

    def get_avg_density(self, traffic_light):
        densities = []
        for detector in traffic_light["detectors"]:
            try:
                densities.append(traci.lanearea.getLastStepOccupancy(detector["id"]))
            except:
                pass
        return np.mean(densities) / 100 if densities else 0.0

    def get_num_phases(self, traffic_light):
        """
        Returns the number of custom-defined traffic light phases.
        """
        return len(traffic_light["phase"])

    def get_movements_from_phase(self, traffic_light, phase_str):
        detectors = [det["id"] for det in traffic_light["detectors"]]
        active_detectors = [
            detectors[i]
            for i, state in enumerate(phase_str)
            if state.upper() == "G" and i < len(detectors)
        ]
        return active_detectors

    def get_avg_queue_length(self, traffic_light):
        """
        Get the average queue length of a lane.
        """
        queue_lengths = []
        for detector in traffic_light["detectors"]:
            try:
                queue_lengths.append(
                    traci.lanearea.getLastStepHaltingNumber(detector["id"])
                )
            except:
                pass
        return np.mean(queue_lengths) if queue_lengths else 0.0

    def get_avg_waiting_time(self, traffic_light):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = []
        for detector in traffic_light["detectors"]:
            vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(detector["id"]))

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
