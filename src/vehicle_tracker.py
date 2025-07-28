"""
Vehicle Tracker for SUMO Traffic Simulation
Tracks vehicle statistics by type (bike, car, truck) during simulation.
"""

import traci
import numpy as np
from collections import defaultdict
import json
import csv
import os

class VehicleTracker:
    """Track vehicle statistics by type (bike, car, truck) during simulation."""
    
    def __init__(self, path, output_dir="logs"):
        """
        Initialize VehicleTracker with path for saving logs.
        
        Args:
            path (str): Base path where model folders are located (e.g., "models/model_11")
            output_dir (str): Subdirectory name for vehicle logs (default: "logs")
        """
        self.path = path
        self.output_dir = os.path.join(path, output_dir) if path else output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Vehicle type mapping based on SUMO vehicle types
        self.vehicle_types = {
            'bike': ['bicycle', 'bike', 'motorcycle', 'moped'],
            'car': ['passenger', 'car', 'taxi', 'private'],
            'truck': ['truck', 'bus', 'delivery', 'trailer', 'coach', 'emergency']
        }
        
        # Statistics tracking
        self.stats = {
            'departed': defaultdict(int),  # {vehicle_type: count}
            'arrived': defaultdict(int),   # {vehicle_type: count}
            'running': defaultdict(int),   # {vehicle_type: count}
            'total_departed': 0,
            'total_arrived': 0,
            'total_running': 0
        }
        
        # Step-by-step tracking for detailed analysis
        self.step_stats = []
        
        # Vehicle journey tracking
        self.vehicle_journeys = {}  # {vehicle_id: {type, depart_time, arrive_time}}
        
        # Initialize counters for all vehicle types
        for vtype in ['bike', 'car', 'truck']:
            self.stats['departed'][vtype] = 0
            self.stats['arrived'][vtype] = 0
            self.stats['running'][vtype] = 0
        
    def get_vehicle_type(self, vehicle_id):
        """Determine vehicle type from SUMO vehicle ID or type."""
        try:
            veh_type = traci.vehicle.getTypeID(vehicle_id).lower()
        except:
            # If vehicle not found, try to infer from ID
            veh_type = vehicle_id.lower()
        
        # Classify vehicle type
        for category, keywords in self.vehicle_types.items():
            if any(keyword in veh_type for keyword in keywords):
                return category
        
        # Default to car if unknown
        return 'car'
    
    def update_stats(self, step):
        """Update vehicle statistics for current simulation step."""
        try:
            # Get current vehicle lists
            departed_ids = traci.simulation.getDepartedIDList()
            arrived_ids = traci.simulation.getArrivedIDList()
            running_ids = traci.vehicle.getIDList()
            
            # Process departed vehicles
            for veh_id in departed_ids:
                veh_type = self.get_vehicle_type(veh_id)
                self.stats['departed'][veh_type] += 1
                self.stats['total_departed'] += 1
                
                # Track journey start
                self.vehicle_journeys[veh_id] = {
                    'type': veh_type,
                    'depart_time': step,
                    'arrive_time': None
                }
            
            # Process arrived vehicles
            for veh_id in arrived_ids:
                if veh_id in self.vehicle_journeys:
                    veh_type = self.vehicle_journeys[veh_id]['type']
                    self.vehicle_journeys[veh_id]['arrive_time'] = step
                else:
                    veh_type = 'car'  # Default if not tracked
                
                self.stats['arrived'][veh_type] += 1
                self.stats['total_arrived'] += 1
            
            # Count running vehicles by type
            running_by_type = defaultdict(int)
            for veh_id in running_ids:
                veh_type = self.get_vehicle_type(veh_id)
                running_by_type[veh_type] += 1
            
            # Update running counts
            self.stats['running'] = dict(running_by_type)
            # Ensure all vehicle types are present
            for vtype in ['bike', 'car', 'truck']:
                if vtype not in self.stats['running']:
                    self.stats['running'][vtype] = 0
            
            self.stats['total_running'] = len(running_ids)
            
            # Store step statistics for every step (detailed tracking for entire episode)
            step_data = {
                'step': step,
                'departed_bike': self.stats['departed']['bike'],
                'departed_car': self.stats['departed']['car'],
                'departed_truck': self.stats['departed']['truck'],
                'arrived_bike': self.stats['arrived']['bike'],
                'arrived_car': self.stats['arrived']['car'],
                'arrived_truck': self.stats['arrived']['truck'],
                'running_bike': self.stats['running']['bike'],
                'running_car': self.stats['running']['car'],
                'running_truck': self.stats['running']['truck'],
                'total_departed': self.stats['total_departed'],
                'total_arrived': self.stats['total_arrived'],
                'total_running': self.stats['total_running']
            }
            self.step_stats.append(step_data)
            
        except Exception as e:
            print(f"Error updating vehicle stats at step {step}: {e}")
    
    def get_current_stats(self):
        """Get current vehicle statistics."""
        return {
            'departed': dict(self.stats['departed']),
            'arrived': dict(self.stats['arrived']),
            'running': dict(self.stats['running']),
            'total_departed': self.stats['total_departed'],
            'total_arrived': self.stats['total_arrived'],
            'total_running': self.stats['total_running']
        }
    
    def get_journey_stats(self):
        """Calculate journey statistics."""
        journey_stats = {
            'bike': {'count': 0, 'avg_time': 0, 'completed': 0},
            'car': {'count': 0, 'avg_time': 0, 'completed': 0},
            'truck': {'count': 0, 'avg_time': 0, 'completed': 0}
        }
        
        for veh_id, journey in self.vehicle_journeys.items():
            veh_type = journey['type']
            journey_stats[veh_type]['count'] += 1
            
            if journey['arrive_time'] is not None:
                journey_stats[veh_type]['completed'] += 1
                travel_time = journey['arrive_time'] - journey['depart_time']
                # Update running average
                current_avg = journey_stats[veh_type]['avg_time']
                completed = journey_stats[veh_type]['completed']
                journey_stats[veh_type]['avg_time'] = (
                    (current_avg * (completed - 1) + travel_time) / completed
                )
        
        return journey_stats
    
    def save_logs(self, episode, simulation_type):
        """Save vehicle logs to files."""
        timestamp = str(episode).zfill(3)
        
        # Save summary statistics
        summary_file = os.path.join(
            self.output_dir, 
            f"vehicle_summary_{simulation_type}_ep{timestamp}.json"
        )
        
        summary_data = {
            'episode': episode,
            'simulation_type': simulation_type,
            'final_stats': self.get_current_stats(),
            'journey_stats': self.get_journey_stats()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed step-by-step data
        csv_file = os.path.join(
            self.output_dir,
            f"vehicle_details_{simulation_type}_ep{timestamp}.csv"
        )
        
        if self.step_stats:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.step_stats[0].keys())
                writer.writeheader()
                writer.writerows(self.step_stats)
    
    def print_summary(self, simulation_type):
        """Print vehicle statistics summary."""
        stats = self.get_current_stats()
        journey_stats = self.get_journey_stats()
        
        print(f"\n🚗 Vehicle Statistics - {simulation_type.upper()}")
        print("=" * 60)
        
        print("📊 Final Counts:")
        for veh_type in ['bike', 'car', 'truck']:
            departed = stats['departed'].get(veh_type, 0)
            arrived = stats['arrived'].get(veh_type, 0)
            running = stats['running'].get(veh_type, 0)
            completion_rate = (arrived / departed * 100) if departed > 0 else 0
            
            print(f"  {veh_type.capitalize():<8}: "
                  f"Departed={departed:>4}, "
                  f"Arrived={arrived:>4}, "
                  f"Running={running:>3}, "
                  f"Completion={completion_rate:>5.1f}%")
        
        print(f"\n📈 Totals: "
              f"Departed={stats['total_departed']}, "
              f"Arrived={stats['total_arrived']}, "
              f"Running={stats['total_running']}")
        
        print("\n⏱️  Average Journey Times:")
        for veh_type in ['bike', 'car', 'truck']:
            avg_time = journey_stats[veh_type]['avg_time']
            completed = journey_stats[veh_type]['completed']
            print(f"  {veh_type.capitalize():<8}: {avg_time:>6.1f} steps ({completed} completed)")
    
    def reset(self):
        """Reset all statistics for new episode."""
        self.stats = {
            'departed': defaultdict(int),
            'arrived': defaultdict(int),
            'running': defaultdict(int),
            'total_departed': 0,
            'total_arrived': 0,
            'total_running': 0
        }
        
        # Initialize counters for all vehicle types
        for vtype in ['bike', 'car', 'truck']:
            self.stats['departed'][vtype] = 0
            self.stats['arrived'][vtype] = 0
            self.stats['running'][vtype] = 0
            
        self.step_stats = []
        self.vehicle_journeys = {}
    
    def get_aggregated_stats(self, interval=100):
        """
        Get aggregated statistics at specified intervals.
        
        Args:
            interval (int): Aggregation interval in steps (default: 100)
            
        Returns:
            list: Aggregated statistics for each interval
        """
        if not self.step_stats:
            return []
        
        aggregated = []
        max_steps = max(stat['step'] for stat in self.step_stats)
        
        for start_step in range(0, max_steps + 1, interval):
            end_step = min(start_step + interval - 1, max_steps)
            
            # Find stats within this interval
            interval_stats = [
                stat for stat in self.step_stats 
                if start_step <= stat['step'] <= end_step
            ]
            
            if interval_stats:
                # Get the last (most recent) stats in this interval
                latest_stat = max(interval_stats, key=lambda x: x['step'])
                
                # Calculate rates for this interval
                interval_data = {
                    'start_step': start_step,
                    'end_step': end_step,
                    'interval': f"{start_step}-{end_step}",
                    'departed_bike': latest_stat['departed_bike'],
                    'departed_car': latest_stat['departed_car'],
                    'departed_truck': latest_stat['departed_truck'],
                    'arrived_bike': latest_stat['arrived_bike'],
                    'arrived_car': latest_stat['arrived_car'],
                    'arrived_truck': latest_stat['arrived_truck'],
                    'running_bike': latest_stat['running_bike'],
                    'running_car': latest_stat['running_car'],
                    'running_truck': latest_stat['running_truck'],
                    'total_departed': latest_stat['total_departed'],
                    'total_arrived': latest_stat['total_arrived'],
                    'total_running': latest_stat['total_running']
                }
                
                aggregated.append(interval_data)
        
        return aggregated
    
    def save_aggregated_stats(self, episode, simulation_type, intervals=[100, 300, 600]):
        """
        Save aggregated statistics at different time intervals.
        
        Args:
            episode (int): Episode number
            simulation_type (str): Type of simulation
            intervals (list): List of intervals to aggregate by
        """
        timestamp = str(episode).zfill(3)
        
        for interval in intervals:
            aggregated_data = self.get_aggregated_stats(interval)
            
            if aggregated_data:
                csv_file = os.path.join(
                    self.output_dir,
                    f"vehicle_aggregated_{interval}s_{simulation_type}_ep{timestamp}.csv"
                )
                
                with open(csv_file, 'w', newline='') as f:
                    if aggregated_data:
                        writer = csv.DictWriter(f, fieldnames=aggregated_data[0].keys())
                        writer.writeheader()
                        writer.writerows(aggregated_data)
