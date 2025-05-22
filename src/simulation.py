from src.model import DQN
from src.memory import ReplayMemory

import traci
import numpy as np
import random
import torch

GREEN_ACTION = 0
RED_ACTION = 1


class Simulation:
    def __init__(
        self,
        selector_phase_agent: DQN,
        green_duration_agent: DQN,
        green_duration_agent_memory: ReplayMemory,
        selector_phase_agent_memory: ReplayMemory,
        green_duration_agent_cfg,
        selector_phase_agent_cfg,
        max_steps,
        traffic_lights,
        interphase_duration=3,
        epoch=1000,
    ):
        self.selector_phase_agent = selector_phase_agent
        self.green_duration_agent = green_duration_agent
        self.green_duration_agent_memory = green_duration_agent_memory
        self.selector_phase_agent_memory = selector_phase_agent_memory
        self.green_duration_agent_cfg = green_duration_agent_cfg
        self.selector_phase_agent_cfg = selector_phase_agent_cfg
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.interphase_duration = interphase_duration
        self.epoch = epoch

        self.green_duration_agent_action = {}
        self.green_duration_agent_old_action = {}
        self.green_duration_agent_reward = 0
        self.green_duration_agent_state = {}
        self.green_duration_agent_old_state = {}

        self.selector_phase_agent_action = -1
        self.selector_phase_agent_old_action = -1
        self.selector_phase_agent_reward = 0
        self.selector_phase_agent_state = {}
        self.selector_phase_agent_old_state = {}

        self.outflow_rate = 0
        self.old_outflow_rate = 0
        self.step = 0
        self.green_time = 0
        self.green_time_old = 0

    def run(self, epsilon, episode):
        """
        Run the simulation for a given number of episodes.
        Args:
            epsilon (float): exploration rate for epsilon-greedy policy
            episode (int): current episode number
        """

        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        while self.step < self.max_steps:
            for i in range(len(self.traffic_lights)):
                traffic_light = self.traffic_lights[0]
                # Train the agents
                self.train_green_duration_agent(traffic_light, epsilon)
                # self.train_selector_phase_agent(traffic_light, epsilon)

                # Get the action from the green duration agent and set the green phase
                green_street = random.choice(
                    list(self.green_duration_agent_action[traffic_light["id"]].keys())
                )
                self.green_time_old = self.green_time
                self.green_time = self.green_duration_agent_action[traffic_light["id"]][
                    green_street
                ]

                print("Green street:", green_street, "Green time:", self.green_time)

                # Get number of phases
                num_phases = sum(
                    len(detector["lane_idx"]) for detector in traffic_light["detectors"]
                )

                # Get all phase index with match street name from green_time
                phase_index = [
                    detector["lane_idx"]
                    for detector in traffic_light["detectors"]
                    if detector["street"] == green_street
                ]

                # We flatten the list
                phase_index = [i for sublist in phase_index for i in sublist]

                print("Phase index:", phase_index)

                steps_todo = 0
                # Set the green phase for the traffic light
                self.set_green_phase(
                    traffic_light["id"], self.green_time, phase_index, num_phases
                )

                if (self.step + self.green_time) >= self.max_steps:
                    steps_todo = self.max_steps - self.step
                else:
                    steps_todo = self.green_time

                number_of_vehicles = 0

                vehicle_ids = [
                    traci.lanearea.getLastStepVehicleIDs(detector["id"])
                    for detector in traffic_light["detectors"] if detector["street"] == green_street
                ]

                old_vehicle_ids = vehicle_ids

                while steps_todo > 0:
                    self.step += 1
                    steps_todo -= 1
                    traci.simulationStep()

                    vehicle_ids = [
                        traci.lanearea.getLastStepVehicleIDs(detector["id"])
                        for detector in traffic_light["detectors"] if detector["street"] == green_street
                    ]

                    # Calculate the number of vehicles exit the detector by comparing the old and new vehicle IDs
                    number_of_vehicles += sum(
                        1 for vid in old_vehicle_ids if vid not in vehicle_ids
                    )

                    # Save current vehicles as old vehicles to compare in the next step
                    old_vehicle_ids = vehicle_ids

                print("Number of vehicles:", number_of_vehicles)

                # Get outflow rate
                self.outflow_rate = (
                    number_of_vehicles / self.green_time if self.green_time > 0 else 0
                )

                print("Outflow rate:", self.outflow_rate)

                self.green_duration_agent_reward = self.get_green_duration_reward()

                print("Green duration agent reward:", self.green_duration_agent_reward)

    def train_green_duration_agent(self, traffic_light, epsilon):
        """
        Train the green duration agent using the current state and action.
        We get the queue length of each street and get the action as green light duration from the green duration agent by loop through each detector on each street.
        """
        state = {}
        self.green_duration_agent_old_action = self.green_duration_agent_action
        self.green_duration_agent_old_state = self.green_duration_agent_state

        for detector in traffic_light["detectors"]:
            queue_length = self.get_queue_length(detector["id"])
            if detector["street"] not in state:
                state[detector["street"]] = 0

            # Get the queue length from the detector
            state[detector["street"]] += queue_length

        self.green_duration_agent_state = state

        # Normalize the state
        for street in state:
            # Decide where to perform an exploration or exploitation action
            if random.random() < epsilon:
                self.green_duration_agent_action[traffic_light["id"][detector]] = (
                    queue_length / 5 * 3
                )
            else:
                result: torch.Tensor = self.green_duration_agent.predict(
                    torch.from_numpy(np.array([state[street]]))
                )

                if traffic_light["id"] not in self.green_duration_agent_action:
                    self.green_duration_agent_action[traffic_light["id"]] = {}

                if street not in self.green_duration_agent_action[traffic_light["id"]]:
                    self.green_duration_agent_action[traffic_light["id"]][street] = 0

                self.green_duration_agent_action[traffic_light["id"]][street] = (
                    np.argmax(result.detach().numpy())
                )

    def get_green_duration_reward(self):
        """
        Get the reward for the green duration agent.
        """
        # Calculate the reward based on the outflow rate and the difference between the current and previous actions
        return (
            self.green_duration_agent_reward
            + (self.green_time_old - self.green_time)
            + (self.outflow_rate - self.old_outflow_rate)
        )

    def train_selector_phase_agent(self, traffic_light, epsilon):
        """
        Train the selector phase agent using the current state and action.
        We get the current state of all detectors and merge with the action as green light duration from the selector phase agent by loop through each detector on each street.
        Then select which street to set green light depend on the action with epsilon-greedy.
        """
        state = {}
        self.selector_phase_agent_old_action = self.selector_phase_agent_action
        self.selector_phase_agent_old_state = self.selector_phase_agent_state

        # Get the current state from the simulation
        for detector in traffic_light["detectors"]:
            _state = self.get_state(detector["id"])

            if detector["street"] not in state:
                state[detector["street"]] = _state

            state[detector["street"]] = np.append(state[detector["street"]], _state)

        print("State: ", state)

        self.selector_phase_agent_state = state

        # Decide where to perform an exploration or exploitation action
        if random.random() < epsilon:
            self.selector_phase_agent_action = random.randint(
                0, self.selector_phase_agent_cfg["num_actions"] - 1
            )
        else:
            self.selector_phase_agent_old_action = self.selector_phase_agent_action
            self.selector_phase_agent_action = np.argmax(
                self.selector_phase_agent.predict(np.array(state))
            )

        # Set the green phase for the traffic light
        self.set_green_phase(self.selector_phase_agent_action, traffic_light["id"])

        # Set the yellow phase for the traffic light
        # self.set_yellow_phase(traffic_light['id'])

        # Get the reward for the selector phase agent
        # self.selector_phase_agent_reward = self.get_selector_phase_reward()

    def set_yellow_phase(self, phase):
        """
        Set the traffic light to yellow phase.
        """
        traci.trafficlight.setPhase(self.traffic_light_id, phase)
        traci.trafficlight.setPhaseDuration(
            self.traffic_light_id, self.interphase_duration
        )

    def set_green_phase(self, tlsId, duration, phase, num_phases):
        """
        Activate the correct green light combination in sumo
        """
        new_phase = self.update_phase(phase, num_phases)
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

    def select_action(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state (np.array): current state of the simulation
            epsilon (float): exploration rate for epsilon-greedy policy
        Returns:
            int: selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                q_values: torch.Tensor = self.selector_phase_agent.predict(state)
                return q_values.argmax().tolist()[0]

    def get_state(self, detector_id):
        """
        Get the current state of the simulation.
        Returns:
            np.array: current state of the simulation
        """
        # Get the current state from the simulation
        state = np.zeros((self.selector_phase_agent_cfg["num_states"],))
        min_free_capacity = self.get_min_free_capacity(detector_id)
        density = self.get_density(detector_id)
        waiting_time = self.get_waiting_time(detector_id)
        if state == np.zeros((self.num_states,)):
            state = np.array(
                [
                    density,
                    min_free_capacity,
                    waiting_time,
                ]
            )
        else:
            state = np.append(
                state,
                np.array(
                    [
                        density,
                        min_free_capacity,
                        waiting_time,
                    ]
                ),
                axis=0,
            )
        return state

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_min_free_capacity(self, detector_id):
        """
        Get the minimum free capacity of a lane.
        """
        lane_length = traci.lanearea.getLength(detector_id)
        occupied_length = traci.lanearea.getLastStepOccupancy(detector_id) * lane_length

        return lane_length - occupied_length

    def get_density(self, detector_id):
        """
        Get the density of vehicles on a lane.
        """
        return traci.lanearea.getLastStepOccupancy(detector_id)

    def get_waiting_time(self, detector_id):
        """
        Get the waiting time of vehicles on a lane.
        """
        return None

    def get_travel_speed(self, edge_id):
        """
        Get the travel speed of vehicles on a lane.
        """
        return traci.edge.getLastStepMeanSpeed(edge_id)

    def get_travel_time(self, edge_id):
        """
        Get the travel time of vehicles on a lane.
        """
        return traci.edge.getTraveltime(edge_id)
