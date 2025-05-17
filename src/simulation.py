from src.model import DQN
from src.memory_nstep import NStepReplayMemory
from src.intersection import Intersection

import traci
import numpy as np
import os
import random
import time

class Simulation:
    def __init__(self, Model: DQN, NStepReplayMemory: NStepReplayMemory, intersection, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions, epoch):
        self.model = Model
        self.memory = NStepReplayMemory
        self.sumo_cmd = sumo_cmd
        self.gamma = gamma
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.red_duration = red_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self.epoch = epoch
    
    def run(self, epsilon, episode):
        """
        Run the simulation for a given number of episodes.
        Args:
            epsilon (float): exploration rate for epsilon-greedy policy
            episode (int): current episode number
        """
        
        # Run the build routes file command
        for intersection in Intersection.get_all_intersections_with_sumo():
            Intersection.generate_routes(intersection, enable_bicycle=True, enable_pedestrian=True, enable_motorcycle=True, enable_passenger=True)

        traci.start(self.sumo_cmd)
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        #init
        self.step = 0
        self.waiting_times = {}
        self.reward = 0
        self.queue_length = 0

        old_total_waiting_time = 0
        old_state = -1
        old_action = -1

    def get_state(self):
        """
        Get the current state of the simulation.
        Returns:
            np.array: current state of the simulation
        """
        # Get the current state from the simulation
        state = np.zeros((self.num_states,))
        for i in range(self.num_states):
            state[i] = traci.lane.getLastStepOccupancy(i)
        return state
    
    def get_queue_length(self):
        """
        Get the current queue length of every incoming lane in specific junction.
        Returns:
            int: current queue length of every incoming lane in specific junction
        """
        # Get the current queue length from the simulation
        queue_length = 0
        for i in range(self.num_states):
            queue_length += traci.lane.getLastStepOccupancy(i)
        return queue_length