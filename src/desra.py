from src.sumo import SUMO

import traci
import numpy as np
from collections import defaultdict, deque
import math


class DESRA(SUMO):
    def __init__(self, interphase_duration=3, buffer_size=100):
        self.interphase_duration = interphase_duration
        self.buffer_size = buffer_size

        # Rolling buffers for real-time stats (per detector)
        self.vehicle_counts = defaultdict(lambda: deque(maxlen=buffer_size))
        self.occupancies = defaultdict(lambda: deque(maxlen=buffer_size))
        self.flow_rates = defaultdict(lambda: deque(maxlen=buffer_size))
        self.densities = defaultdict(lambda: deque(maxlen=buffer_size))
        
        # Time stamps for rate calculations
        self.last_update_time = defaultdict(float)
        
        # Default fallback parameters (used when insufficient data)
        self.default_saturation_flow = 0.5    # veh/s
        self.default_critical_density = 0.08  # veh/m
        self.default_jam_density = 0.18       # veh/m
        
        # Dynamically estimated parameters (per detector)
        self.estimated_saturation_flow = defaultdict(lambda: self.default_saturation_flow)
        self.estimated_critical_density = defaultdict(lambda: self.default_critical_density)
        self.estimated_jam_density = defaultdict(lambda: self.default_jam_density)

    def select_phase_with_desra_hints(
        self,
        traffic_light,
        q_arrivals: dict,
        current_time: float = None,
        use_global_params: bool = False,
        global_saturation_flow: float = None,
        global_critical_density: float = None,
        global_jam_density: float = None,
    ):
        """
        Select best phase using DESRA hints based on effective outflow.
        
        Args:
            traffic_light: Traffic light configuration
            q_arrivals: Arrival flow rates per detector
            current_time: Current simulation time for parameter updates
            use_global_params: If True, use provided global parameters for all detectors
            global_*: Global parameters to use when use_global_params=True
        """
        best_phase = traffic_light["phase"][0]
        best_green_time = 0
        best_effective_outflow = -1.0

        # Update traffic parameters if current_time is provided
        if current_time is not None:
            for phase_str in traffic_light["phase"]:
                movements = self.get_movements_from_phase(traffic_light, phase_str)
                for det in movements:
                    self.update_traffic_parameters(det, current_time)

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            # 1) Compute the downstream free space for this phase (xd)
            #    i.e. the most restrictive downstream capacity over all movements.
            xd = float("inf")
            for det in movements:
                lane_len = traci.lanearea.getLength(det)
                free_down = self.get_downstream_queue_length(det)  # in meters
                xd = min(xd, free_down)

            # 2) Find Gsat for each movement, then take the MIN positive → Gmin
            Gmin = float("inf")
            sat_times = {}
            for det in movements:
                x0 = self.get_queue_length(det)  # upstream queue (m)
                q_arr = q_arrivals.get(det, 0.0)  # veh/s
                lane_len = traci.lanearea.getLength(det)  # Δ_j
                
                # Get parameters for this detector
                if use_global_params and all(p is not None for p in [global_saturation_flow, global_critical_density, global_jam_density]):
                    params = {
                        'saturation_flow': global_saturation_flow,
                        'critical_density': global_critical_density,
                        'jam_density': global_jam_density
                    }
                else:
                    params = self.get_detector_parameters(det)
                
                Gsat = self.estimate_saturated_green_time(
                    x0, q_arr, xd, lane_len, 
                    params['saturation_flow'],
                    params['critical_density'], 
                    params['jam_density']
                )

                if Gsat > 0:
                    Gmin = min(Gmin, Gsat)
                sat_times[det] = (Gsat, x0, q_arr, lane_len, params)

            # if no movement had Gsat>0, skip
            if Gmin == float("inf"):
                continue

            # 3) For this Gmin, compute total effective outflow F_total
            F_total = 0.0
            for det, (Gsat, x0, q_arr, lane_len, params) in sat_times.items():
                s = params['saturation_flow']
                # three‐case formula from Eq. (9):
                if Gsat >= Gmin:
                    # full‐capacity discharge
                    F_total += s * Gmin
                else:
                    # Gsat < Gmin → two subcases
                    # we need to know if the queue‐extent was upstream‐bounded or downstream‐bounded.
                    # but the paper simplifies it to:
                    #  - if xM <= xd then we can use the second branch
                    #  - else the third.
                    # we already computed xM inside estimate_saturated — but for clarity:
                    xM = self.estimate_shockwave_extent(
                        x0, q_arr, lane_len,
                        params['saturation_flow'],
                        params['critical_density'],
                        params['jam_density']
                    )
                    if xM <= xd:
                        # Eq. (9) middle case: s*Gsat + q_arr*(Gmin - Gsat)
                        F_total += s * Gsat + q_arr * (Gmin - Gsat)
                    else:
                        # Eq. (9) last case: s*Gsat
                        F_total += s * Gsat

            # 4) Compute effective outflow rate v_i
            v_i = F_total / (Gmin + self.interphase_duration)

            # 5) Track the best
            if v_i > best_effective_outflow:
                best_effective_outflow = v_i
                best_phase = phase_str
                best_green_time = Gmin
            elif v_i == best_effective_outflow and Gmin < best_green_time:
                best_phase = phase_str
                best_green_time = Gmin

        return best_phase, max(0, int(best_green_time))

    def update_traffic_parameters(self, detector_id, current_time):
        """
        Update real-time traffic parameters for a detector based on recent measurements.
        This should be called periodically (e.g., every simulation step) to maintain
        up-to-date parameter estimates.
        """
        try:
            # Get current measurements
            vehicle_count = traci.lanearea.getLastStepVehicleNumber(detector_id)
            occupancy_pct = traci.lanearea.getLastStepOccupancy(detector_id)
            detector_length = traci.lanearea.getLength(detector_id)
            
            # Store measurements
            self.vehicle_counts[detector_id].append(vehicle_count)
            self.occupancies[detector_id].append(occupancy_pct)
            
            # Calculate density (vehicles per meter)
            density = (occupancy_pct / 100.0) * self.estimate_vehicle_density_factor()
            self.densities[detector_id].append(density)
            
            # Calculate flow rate if we have time data
            if detector_id in self.last_update_time:
                time_diff = current_time - self.last_update_time[detector_id]
                if time_diff > 0:
                    # Simple flow calculation (could be improved with vehicle exit counts)
                    flow_rate = vehicle_count / max(time_diff, 1.0)  # veh/s
                    self.flow_rates[detector_id].append(flow_rate)
            
            self.last_update_time[detector_id] = current_time
            
            # Estimate parameters if we have sufficient data
            if len(self.densities[detector_id]) >= 20:  # Minimum data points
                self._estimate_fundamental_diagram_parameters(detector_id)
                
        except Exception as e:
            print(f"Warning: Could not update traffic parameters for {detector_id}: {e}")

    def _estimate_fundamental_diagram_parameters(self, detector_id):
        """
        Estimate saturation flow, critical density, and jam density from traffic data
        using the fundamental diagram relationship: flow = density * speed
        """
        densities = list(self.densities[detector_id])
        flows = list(self.flow_rates[detector_id])
        
        if len(densities) < 20 or len(flows) < 20:
            return  # Insufficient data
        
        try:
            import numpy as np
            
            # Use recent data (last 50 measurements)
            recent_densities = np.array(densities[-50:])
            recent_flows = np.array(flows[-50:])
            
            # Remove outliers (beyond 2 standard deviations)
            flow_mean, flow_std = np.mean(recent_flows), np.std(recent_flows)
            density_mean, density_std = np.mean(recent_densities), np.std(recent_densities)
            
            valid_mask = (
                (np.abs(recent_flows - flow_mean) <= 2 * flow_std) &
                (np.abs(recent_densities - density_mean) <= 2 * density_std) &
                (recent_densities > 0)
            )
            
            if np.sum(valid_mask) < 10:
                return  # Not enough valid data
            
            clean_densities = recent_densities[valid_mask]
            clean_flows = recent_flows[valid_mask]
            
            # Estimate jam density (maximum observed density + buffer)
            jam_density = min(np.max(clean_densities) * 1.2, 0.25)  # Cap at reasonable value
            
            # Estimate critical density (density at maximum flow)
            max_flow_idx = np.argmax(clean_flows)
            critical_density = clean_densities[max_flow_idx]
            
            # Ensure critical < jam density
            if critical_density >= jam_density:
                critical_density = jam_density * 0.6
            
            # Estimate saturation flow (maximum observed flow + buffer)
            saturation_flow = np.max(clean_flows) * 1.1
            
            # Update estimates with smoothing (exponential moving average)
            alpha = 0.1  # Smoothing factor
            self.estimated_saturation_flow[detector_id] = (
                alpha * saturation_flow + 
                (1 - alpha) * self.estimated_saturation_flow[detector_id]
            )
            self.estimated_critical_density[detector_id] = (
                alpha * critical_density + 
                (1 - alpha) * self.estimated_critical_density[detector_id]
            )
            self.estimated_jam_density[detector_id] = (
                alpha * jam_density + 
                (1 - alpha) * self.estimated_jam_density[detector_id]
            )
            
        except ImportError:
            # Fallback to simple estimation without numpy
            self._simple_parameter_estimation(detector_id)
        except Exception as e:
            print(f"Warning: Parameter estimation failed for {detector_id}: {e}")

    def _simple_parameter_estimation(self, detector_id):
        """
        Simple parameter estimation when numpy is not available.
        """
        densities = list(self.densities[detector_id])
        flows = list(self.flow_rates[detector_id])
        
        # Simple statistics
        max_density = max(densities[-50:])  # Last 50 measurements
        max_flow = max(flows[-50:])
        
        # Rough estimates
        jam_density = min(max_density * 1.2, 0.25)
        critical_density = max_density * 0.7  # Assume critical is ~70% of observed max
        saturation_flow = max_flow * 1.1
        
        # Ensure physical constraints
        if critical_density >= jam_density:
            critical_density = jam_density * 0.6
        
        # Smooth updates
        alpha = 0.1
        self.estimated_saturation_flow[detector_id] = (
            alpha * saturation_flow + 
            (1 - alpha) * self.estimated_saturation_flow[detector_id]
        )
        self.estimated_critical_density[detector_id] = (
            alpha * critical_density + 
            (1 - alpha) * self.estimated_critical_density[detector_id]
        )
        self.estimated_jam_density[detector_id] = (
            alpha * jam_density + 
            (1 - alpha) * self.estimated_jam_density[detector_id]
        )

    def estimate_vehicle_density_factor(self):
        """
        Estimate the conversion factor from occupancy percentage to vehicle density.
        This depends on average vehicle length and detector characteristics.
        """
        # Typical values: passenger car ~5m, detector sensitivity, spacing
        # This is a simplified model - in practice, this should be calibrated
        avg_vehicle_length = 5.0  # meters
        detection_factor = 1.2    # accounts for detection gaps
        return 1.0 / (avg_vehicle_length * detection_factor)

    def get_detector_parameters(self, detector_id):
        """
        Get the current parameter estimates for a specific detector.
        Returns default values if insufficient data is available.
        """
        return {
            'saturation_flow': self.estimated_saturation_flow[detector_id],
            'critical_density': self.estimated_critical_density[detector_id], 
            'jam_density': self.estimated_jam_density[detector_id]
        }

    def estimate_shockwave_extent(self, x0, q_arr, link_length, saturation_flow=None, critical_density=None, jam_density=None):
        """
        Estimate the extent of the shockwave using the shockwave theory.
        
        Based on Eq. (3): xM = x0 * (kj * s - q_arr * kc) / (kj * (s - q_arr))
        
        Args:
            x0 (float): Upstream queue length **in meters**
            q_arr (float): Arrival flow rate (veh/s)
            link_length (float): Length of upstream link (m)
            saturation_flow (float): Override saturation flow rate (veh/s)
            critical_density (float): Override critical density (veh/m)
            jam_density (float): Override jam density (veh/m)
        Returns:
            float: Shockwave extent (meters), bounded to [0, link_length]
        """
        # Use provided parameters or defaults
        s = saturation_flow if saturation_flow is not None else self.default_saturation_flow
        kc = critical_density if critical_density is not None else self.default_critical_density
        kj = jam_density if jam_density is not None else self.default_jam_density
        
        # Input validation
        if s <= 0 or kc <= 0 or kj <= 0:
            return 0.0
        if x0 < 0:
            return 0.0
        if q_arr < 0:
            return x0  # No arrivals, shockwave doesn't extend

        # If arrival rate meets or exceeds saturation, queue grows indefinitely
        if q_arr >= s:
            return link_length
            
        # Handle zero queue case
        if x0 == 0:
            return 0.0

        # Calculate shockwave extent per Eq. 3
        # xM = x0 * (kj * s - q_arr * kc) / (kj * (s - q_arr))
        numerator = kj * s - q_arr * kc
        denom = kj * (s - q_arr)
        
        if abs(denom) < 1e-9:
            return link_length
            
        xM = x0 * numerator / denom
        
        # Physical bounds: shockwave extent must be non-negative and within link
        return max(0.0, min(xM, link_length))

    def estimate_saturated_green_time(self, x0, q_arr, xd, link_length, saturation_flow=None, critical_density=None, jam_density=None):
        """
        Estimate saturated green time using spillback-aware shockwave theory.
        
        Based on Eq. (7) three-case piecewise formula.
        
        Args:
            x0 (float): Upstream queue length **in meters**
            q_arr (float): Arrival flow rate (veh/s)
            xd (float): **Min** downstream **free space** in meters
            link_length (float): Length of upstream link (m)
            saturation_flow (float): Override saturation flow rate (veh/s)
            critical_density (float): Override critical density (veh/m)
            jam_density (float): Override jam density (veh/m)
        Returns:
            float: Saturated green time (seconds)
        """
        # Use provided parameters or defaults
        s = saturation_flow if saturation_flow is not None else self.default_saturation_flow
        kc = critical_density if critical_density is not None else self.default_critical_density
        kj = jam_density if jam_density is not None else self.default_jam_density
        
        # Input validation
        if s <= 0 or kc <= 0 or kj <= 0:
            return 0.0
        if x0 < 0 or xd < 0 or link_length <= 0:
            return 0.0
        if q_arr < 0:
            q_arr = 0.0

        # Step 1: Calculate shockwave extent using improved method
        xM = self.estimate_shockwave_extent(x0, q_arr, link_length, s, kc, kj)

        # Step 2: Apply three-case piecewise green time formula (Eq. 7)
        # Case 1: Shockwave doesn't reach downstream spillback point or link end
        if xM <= min(xd, link_length):
            if xM <= 0:
                return 0.0
            Gsat = xM * kj / s
        # Case 2: Link is shorter than both shockwave and downstream capacity  
        elif link_length <= min(xM, xd):
            Gsat = link_length * kj / s
        # Case 3: Downstream capacity is the limiting factor
        elif xd <= min(xM, link_length):
            if xd <= 0:
                return 0.0
            Gsat = xd * kj / s
        else:
            Gsat = 0.0

        return max(0.0, Gsat)

    def get_movements_from_phase(self, traffic_light, phase_str):
        """
        Get detector IDs whose street is active (green) in the given phase string.
        """
        phase_index = traffic_light["phase"].index(phase_str)  # Find index of phase_str
        active_street = str(
            phase_index + 1
        )  # Assuming street "1" is for phase 0, "2" is for phase 1, etc.

        # Collect detector IDs belonging to the active street
        active_detectors = [
            det["id"]
            for det in traffic_light["detectors"]
            if det["street"] == active_street
        ]

        return active_detectors

    def get_queue_length(self, detector_id):
        return (
            traci.lanearea.getLastStepOccupancy(detector_id)
            / 100
            * traci.lanearea.getLength(detector_id)
        )

    def get_downstream_queue_length(self, detector_id):
        """
        Returns the available storage (free space in meters) of the downstream link(s)
        for the given area‑detector. Uses the minimum over all downstream lanes.
        """
        lane_id = traci.lanearea.getLaneID(detector_id)
        links = traci.lane.getLinks(lane_id)  # list of (toLane, viaEdge, ...)
        free_spaces = []

        for link in links:
            downstream_lane = link[0]
            # length = traci.lane.getLength(downstream_lane)
            length = 100
            occupancy_pct = traci.lane.getLastStepOccupancy(downstream_lane)
            # occupancy_pct is 0–100%, so queue_length_m = length * (occupancy_pct/100)
            queue_m = length * (occupancy_pct / 100.0)
            free_spaces.append(max(0.0, length - queue_m))

        if free_spaces:
            return min(free_spaces)
        else:
            # no downstream links → assume full storage available
            return traci.lanearea.getLength(detector_id)
