# import libsumo as traci
import os
import traci

from sim_utils.traffic_metrics import TrafficMetrics
class SUMO:
    def __init__(self, port):
        if 'LIBSUMO_AS_TRACI' not in os.environ and 'LIBTRACI_AS_TRACI' not in os.environ:
            self.simulation_conn = traci.getConnection(f"master-{port}")
        else:
            self.simulation_conn = traci
        self.traffic_metrics = TrafficMetrics(port)

    def get_vehicles_in_phase(self, traffic_light, phase_str):
        """
        Returns the vehicle IDs on lanes with a green signal in the specified phase.
        """

        lane_idxs = []

        # Group all lane_idx in detectors
        lane_idxs = [
            detector["id"]
            for detector in traffic_light["detectors"]
        ]

        green_lanes = [
            lane_idxs[i]
            for i, light_state in enumerate(phase_str)
            if light_state.upper() == "G" and i < len(lane_idxs)
        ]

        vehicle_ids = []
        for lane in green_lanes:
            try:
                vehicle_ids.extend(self.simulation_conn.lanearea.getLastStepVehicleIDs(lane))
            except:
                pass  # Skip any invalid or unavailable lanes

        return vehicle_ids