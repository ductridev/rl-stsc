"""
Vehicle Tracker for SUMO Traffic Simulation
Tracks vehicle statistics by type (bike, car, truck) during simulation.
"""

import libsumo as traci
import numpy as np
from collections import defaultdict
import json
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd


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
            "bike": ["bicycle", "bike", "motorcycle", "moped"],
            "car": ["passenger", "car", "taxi", "private"],
            "truck": ["truck", "bus", "delivery", "trailer", "coach", "emergency"],
        }

        # Statistics tracking
        self.stats = {
            "departed": defaultdict(int),  # {vehicle_type: count}
            "arrived": defaultdict(int),  # {vehicle_type: count}
            "running": defaultdict(int),  # {vehicle_type: count}
            "total_departed": 0,
            "total_arrived": 0,
            "total_running": 0,
        }

        # Step-by-step tracking for detailed analysis
        self.step_stats = []

        # Vehicle journey tracking
        self.vehicle_journeys = {}  # {vehicle_id: {type, depart_time, arrive_time}}

        # Initialize counters for all vehicle types
        for vtype in ["bike", "car", "truck"]:
            self.stats["departed"][vtype] = 0
            self.stats["arrived"][vtype] = 0
            self.stats["running"][vtype] = 0

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
        return "car"

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
                self.stats["departed"][veh_type] += 1
                self.stats["total_departed"] += 1

                # Track journey start
                self.vehicle_journeys[veh_id] = {
                    "type": veh_type,
                    "depart_time": step,
                    "arrive_time": None,
                }

            # Process arrived vehicles
            for veh_id in arrived_ids:
                if veh_id in self.vehicle_journeys:
                    veh_type = self.vehicle_journeys[veh_id]["type"]
                    self.vehicle_journeys[veh_id]["arrive_time"] = step
                else:
                    veh_type = "car"  # Default if not tracked

                self.stats["arrived"][veh_type] += 1
                self.stats["total_arrived"] += 1

            # Count running vehicles by type
            running_by_type = defaultdict(int)
            for veh_id in running_ids:
                veh_type = self.get_vehicle_type(veh_id)
                running_by_type[veh_type] += 1

            # Update running counts
            self.stats["running"] = dict(running_by_type)
            # Ensure all vehicle types are present
            for vtype in ["bike", "car", "truck"]:
                if vtype not in self.stats["running"]:
                    self.stats["running"][vtype] = 0

            self.stats["total_running"] = len(running_ids)

            # Store step statistics for every step (detailed tracking for entire episode)
            step_data = {
                "step": step,
                "departed_bike": self.stats["departed"]["bike"],
                "departed_car": self.stats["departed"]["car"],
                "departed_truck": self.stats["departed"]["truck"],
                "arrived_bike": self.stats["arrived"]["bike"],
                "arrived_car": self.stats["arrived"]["car"],
                "arrived_truck": self.stats["arrived"]["truck"],
                "running_bike": self.stats["running"]["bike"],
                "running_car": self.stats["running"]["car"],
                "running_truck": self.stats["running"]["truck"],
                "total_departed": self.stats["total_departed"],
                "total_arrived": self.stats["total_arrived"],
                "total_running": self.stats["total_running"],
            }
            self.step_stats.append(step_data)

        except Exception as e:
            print(f"Error updating vehicle stats at step {step}: {e}")

    def get_current_stats(self):
        """Get current vehicle statistics."""
        return {
            "departed": dict(self.stats["departed"]),
            "arrived": dict(self.stats["arrived"]),
            "running": dict(self.stats["running"]),
            "total_departed": self.stats["total_departed"],
            "total_arrived": self.stats["total_arrived"],
            "total_running": self.stats["total_running"],
        }

    def get_journey_stats(self):
        """Calculate journey statistics."""
        journey_stats = {
            "bike": {"count": 0, "avg_time": 0, "completed": 0},
            "car": {"count": 0, "avg_time": 0, "completed": 0},
            "truck": {"count": 0, "avg_time": 0, "completed": 0},
        }

        for veh_id, journey in self.vehicle_journeys.items():
            veh_type = journey["type"]
            journey_stats[veh_type]["count"] += 1

            if journey["arrive_time"] is not None:
                journey_stats[veh_type]["completed"] += 1
                travel_time = journey["arrive_time"] - journey["depart_time"]
                # Update running average
                current_avg = journey_stats[veh_type]["avg_time"]
                completed = journey_stats[veh_type]["completed"]
                journey_stats[veh_type]["avg_time"] = (
                    current_avg * (completed - 1) + travel_time
                ) / completed

        return journey_stats

    def save_logs(self, episode, simulation_type, create_plots=True):
        """Save vehicle logs to files and optionally create plots."""
        timestamp = str(episode).zfill(3)

        # Save summary statistics
        summary_file = os.path.join(
            self.output_dir, f"vehicle_summary_{simulation_type}_ep{timestamp}.json"
        )

        summary_data = {
            "episode": episode,
            "simulation_type": simulation_type,
            "final_stats": self.get_current_stats(),
            "journey_stats": self.get_journey_stats(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save detailed step-by-step data
        csv_file = os.path.join(
            self.output_dir, f"vehicle_details_{simulation_type}_ep{timestamp}.csv"
        )

        if self.step_stats:
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.step_stats[0].keys())
                writer.writeheader()
                writer.writerows(self.step_stats)

        # Create plots if requested
        if create_plots:
            self.create_all_vehicle_plots(episode, simulation_type, save_plots=True)

        print(f"Vehicle logs saved for {simulation_type.upper()} Episode {episode}")
        if create_plots:
            print(
                f"Vehicle plots created for {simulation_type.upper()} Episode {episode}"
            )

    def print_summary(self, simulation_type):
        """Print vehicle statistics summary."""
        stats = self.get_current_stats()
        journey_stats = self.get_journey_stats()

        print(f"\n🚗 Vehicle Statistics - {simulation_type.upper()}")
        print("=" * 60)

        print("Final Counts:")
        for veh_type in ["bike", "car", "truck"]:
            departed = stats["departed"].get(veh_type, 0)
            arrived = stats["arrived"].get(veh_type, 0)
            running = stats["running"].get(veh_type, 0)
            completion_rate = (arrived / departed * 100) if departed > 0 else 0

            print(
                f"  {veh_type.capitalize():<8}: "
                f"Departed={departed:>4}, "
                f"Arrived={arrived:>4}, "
                f"Running={running:>3}, "
                f"Completion={completion_rate:>5.1f}%"
            )

        print(
            f"\n📈 Totals: "
            f"Departed={stats['total_departed']}, "
            f"Arrived={stats['total_arrived']}, "
            f"Running={stats['total_running']}"
        )

        print("\n⏱️  Average Journey Times:")
        for veh_type in ["bike", "car", "truck"]:
            avg_time = journey_stats[veh_type]["avg_time"]
            completed = journey_stats[veh_type]["completed"]
            print(
                f"  {veh_type.capitalize():<8}: {avg_time:>6.1f} steps ({completed} completed)"
            )

    def reset(self):
        """Reset all statistics for new episode."""
        self.stats = {
            "departed": defaultdict(int),
            "arrived": defaultdict(int),
            "running": defaultdict(int),
            "total_departed": 0,
            "total_arrived": 0,
            "total_running": 0,
        }

        # Initialize counters for all vehicle types
        for vtype in ["bike", "car", "truck"]:
            self.stats["departed"][vtype] = 0
            self.stats["arrived"][vtype] = 0
            self.stats["running"][vtype] = 0

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
        max_steps = max(stat["step"] for stat in self.step_stats)

        for start_step in range(0, max_steps + 1, interval):
            end_step = min(start_step + interval - 1, max_steps)

            # Find stats within this interval
            interval_stats = [
                stat
                for stat in self.step_stats
                if start_step <= stat["step"] <= end_step
            ]

            if interval_stats:
                # Get the last (most recent) stats in this interval
                latest_stat = max(interval_stats, key=lambda x: x["step"])

                # Calculate rates for this interval
                interval_data = {
                    "start_step": start_step,
                    "end_step": end_step,
                    "interval": f"{start_step}-{end_step}",
                    "departed_bike": latest_stat["departed_bike"],
                    "departed_car": latest_stat["departed_car"],
                    "departed_truck": latest_stat["departed_truck"],
                    "arrived_bike": latest_stat["arrived_bike"],
                    "arrived_car": latest_stat["arrived_car"],
                    "arrived_truck": latest_stat["arrived_truck"],
                    "running_bike": latest_stat["running_bike"],
                    "running_car": latest_stat["running_car"],
                    "running_truck": latest_stat["running_truck"],
                    "total_departed": latest_stat["total_departed"],
                    "total_arrived": latest_stat["total_arrived"],
                    "total_running": latest_stat["total_running"],
                }

                aggregated.append(interval_data)

        return aggregated

    def save_aggregated_stats(
        self, episode, simulation_type, intervals=[100, 300, 600]
    ):
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
                    f"vehicle_aggregated_{interval}s_{simulation_type}_ep{timestamp}.csv",
                )

                with open(csv_file, "w", newline="") as f:
                    if aggregated_data:
                        writer = csv.DictWriter(f, fieldnames=aggregated_data[0].keys())
                        writer.writeheader()
                        writer.writerows(aggregated_data)

    def plot_vehicle_flow_over_time(self, episode, simulation_type, save_plot=True):
        """Plot vehicle counts over simulation time."""
        if not self.step_stats:
            print("No step statistics available for plotting")
            return

        df = pd.DataFrame(self.step_stats)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Vehicle Flow Analysis - {simulation_type.upper()} Episode {episode}",
            fontsize=16,
        )

        # Plot 1: Cumulative Departed Vehicles
        ax1.plot(
            df["step"], df["departed_bike"], label="Bike", color="green", linewidth=2
        )
        ax1.plot(df["step"], df["departed_car"], label="Car", color="blue", linewidth=2)
        ax1.plot(
            df["step"], df["departed_truck"], label="Truck", color="red", linewidth=2
        )
        ax1.plot(
            df["step"],
            df["total_departed"],
            label="Total",
            color="black",
            linewidth=3,
            linestyle="--",
        )
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Cumulative Departed Vehicles")
        ax1.set_title("Vehicles Entering Network")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative Arrived Vehicles
        ax2.plot(
            df["step"], df["arrived_bike"], label="Bike", color="green", linewidth=2
        )
        ax2.plot(df["step"], df["arrived_car"], label="Car", color="blue", linewidth=2)
        ax2.plot(
            df["step"], df["arrived_truck"], label="Truck", color="red", linewidth=2
        )
        ax2.plot(
            df["step"],
            df["total_arrived"],
            label="Total",
            color="black",
            linewidth=3,
            linestyle="--",
        )
        ax2.set_xlabel("Simulation Step")
        ax2.set_ylabel("Cumulative Arrived Vehicles")
        ax2.set_title("Vehicles Exiting Network")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Running Vehicles (Network Load)
        ax3.plot(
            df["step"], df["running_bike"], label="Bike", color="green", linewidth=2
        )
        ax3.plot(df["step"], df["running_car"], label="Car", color="blue", linewidth=2)
        ax3.plot(
            df["step"], df["running_truck"], label="Truck", color="red", linewidth=2
        )
        ax3.plot(
            df["step"],
            df["total_running"],
            label="Total",
            color="black",
            linewidth=3,
            linestyle="--",
        )
        ax3.set_xlabel("Simulation Step")
        ax3.set_ylabel("Vehicles Currently in Network")
        ax3.set_title("Network Load (Running Vehicles)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Completion Rate Over Time
        completion_rate = (
            df["total_arrived"] / df["total_departed"].replace(0, 1)
        ) * 100
        ax4.plot(df["step"], completion_rate, color="purple", linewidth=3)
        ax4.set_xlabel("Simulation Step")
        ax4.set_ylabel("Completion Rate (%)")
        ax4.set_title("Vehicle Completion Rate")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        plt.tight_layout()

        if save_plot:
            plot_file = os.path.join(
                self.output_dir, f"vehicle_flow_{simulation_type}_ep{episode:03d}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"Vehicle flow plot saved: {plot_file}")
            plt.close()
        else:
            plt.show()

    def plot_journey_analysis(self, episode, simulation_type, save_plot=True):
        """Plot journey time analysis from current statistics."""
        stats = self.get_current_stats()
        journey_stats = self.get_journey_stats()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Journey Analysis - {simulation_type.upper()} Episode {episode}",
            fontsize=16,
        )

        vehicle_types = ["bike", "car", "truck"]
        colors = ["green", "blue", "red"]

        # Plot 1: Average Journey Times by Vehicle Type
        avg_times = [journey_stats[vtype]["avg_time"] for vtype in vehicle_types]
        bars1 = ax1.bar(vehicle_types, avg_times, color=colors, alpha=0.7)
        ax1.set_xlabel("Vehicle Type")
        ax1.set_ylabel("Average Journey Time (steps)")
        ax1.set_title("Average Journey Times by Vehicle Type")

        # Add value labels on bars
        for bar, time in zip(bars1, avg_times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{time:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 2: Completion Rates by Vehicle Type
        completion_rates = []
        for vtype in vehicle_types:
            total = journey_stats[vtype]["count"]
            completed = journey_stats[vtype]["completed"]
            rate = (completed / total * 100) if total > 0 else 0
            completion_rates.append(rate)

        bars2 = ax2.bar(vehicle_types, completion_rates, color=colors, alpha=0.7)
        ax2.set_xlabel("Vehicle Type")
        ax2.set_ylabel("Completion Rate (%)")
        ax2.set_title("Completion Rates by Vehicle Type")
        ax2.set_ylim(0, 100)

        # Add value labels on bars
        for bar, rate in zip(bars2, completion_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 3: Vehicle Count Distribution
        departed_counts = [stats["departed"][vtype] for vtype in vehicle_types]
        arrived_counts = [stats["arrived"][vtype] for vtype in vehicle_types]

        x_pos = np.arange(len(vehicle_types))
        width = 0.35

        ax3.bar(
            x_pos - width / 2,
            departed_counts,
            width,
            label="Departed",
            alpha=0.8,
            color="lightblue",
        )
        ax3.bar(
            x_pos + width / 2,
            arrived_counts,
            width,
            label="Arrived",
            alpha=0.8,
            color="lightgreen",
        )

        ax3.set_xlabel("Vehicle Type")
        ax3.set_ylabel("Number of Vehicles")
        ax3.set_title("Vehicle Count Distribution")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([vt.capitalize() for vt in vehicle_types])
        ax3.legend()

        # Plot 4: Network Efficiency
        total_departed = stats["total_departed"]
        total_arrived = stats["total_arrived"]
        overall_efficiency = (
            (total_arrived / total_departed * 100) if total_departed > 0 else 0
        )

        # Create a gauge-like plot
        ax4.pie(
            [overall_efficiency, 100 - overall_efficiency],
            labels=["Completed", "Incomplete"],
            colors=["lightgreen", "lightcoral"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax4.set_title(
            f"Overall Network Efficiency\n{total_arrived}/{total_departed} vehicles"
        )

        plt.tight_layout()

        if save_plot:
            plot_file = os.path.join(
                self.output_dir,
                f"journey_analysis_{simulation_type}_ep{episode:03d}.png",
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"Journey analysis plot saved: {plot_file}")
            plt.close()
        else:
            plt.show()

    def plot_vehicle_type_breakdown(self, episode, simulation_type, save_plot=True):
        """Plot detailed breakdown of vehicle types over time."""
        if not self.step_stats:
            print("No step statistics available for plotting")
            return

        df = pd.DataFrame(self.step_stats)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f"Vehicle Type Breakdown - {simulation_type.upper()} Episode {episode}",
            fontsize=16,
        )

        # Plot 1: Stacked Area Chart - Departed Vehicles
        ax1.fill_between(
            df["step"], 0, df["departed_bike"], alpha=0.7, color="green", label="Bike"
        )
        ax1.fill_between(
            df["step"],
            df["departed_bike"],
            df["departed_bike"] + df["departed_car"],
            alpha=0.7,
            color="blue",
            label="Car",
        )
        ax1.fill_between(
            df["step"],
            df["departed_bike"] + df["departed_car"],
            df["departed_bike"] + df["departed_car"] + df["departed_truck"],
            alpha=0.7,
            color="red",
            label="Truck",
        )
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Cumulative Departed Vehicles")
        ax1.set_title("Departed Vehicles by Type (Stacked)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Stacked Area Chart - Arrived Vehicles
        ax2.fill_between(
            df["step"], 0, df["arrived_bike"], alpha=0.7, color="green", label="Bike"
        )
        ax2.fill_between(
            df["step"],
            df["arrived_bike"],
            df["arrived_bike"] + df["arrived_car"],
            alpha=0.7,
            color="blue",
            label="Car",
        )
        ax2.fill_between(
            df["step"],
            df["arrived_bike"] + df["arrived_car"],
            df["arrived_bike"] + df["arrived_car"] + df["arrived_truck"],
            alpha=0.7,
            color="red",
            label="Truck",
        )
        ax2.set_xlabel("Simulation Step")
        ax2.set_ylabel("Cumulative Arrived Vehicles")
        ax2.set_title("Arrived Vehicles by Type (Stacked)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Running Vehicles Percentage
        total_running = df["total_running"].replace(0, 1)  # Avoid division by zero
        bike_pct = (df["running_bike"] / total_running) * 100
        car_pct = (df["running_car"] / total_running) * 100
        truck_pct = (df["running_truck"] / total_running) * 100

        ax3.plot(df["step"], bike_pct, label="Bike %", color="green", linewidth=2)
        ax3.plot(df["step"], car_pct, label="Car %", color="blue", linewidth=2)
        ax3.plot(df["step"], truck_pct, label="Truck %", color="red", linewidth=2)
        ax3.set_xlabel("Simulation Step")
        ax3.set_ylabel("Percentage of Running Vehicles")
        ax3.set_title("Vehicle Type Mix in Network")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        # Plot 4: Throughput Rate by Vehicle Type (moving average)
        window = 60
        bike_throughput = df["arrived_bike"].diff().rolling(window=window).mean()
        car_throughput = df["arrived_car"].diff().rolling(window=window).mean()
        truck_throughput = df["arrived_truck"].diff().rolling(window=window).mean()

        ax4.plot(df["step"], bike_throughput, label="Bike", color="green", linewidth=2)
        ax4.plot(df["step"], car_throughput, label="Car", color="blue", linewidth=2)
        ax4.plot(df["step"], truck_throughput, label="Truck", color="red", linewidth=2)
        ax4.set_xlabel("Simulation Step")
        ax4.set_ylabel("Throughput Rate (vehicles/step)")
        ax4.set_title(f"Vehicle Throughput Rate ({window}-step moving average)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_file = os.path.join(
                self.output_dir,
                f"vehicle_breakdown_{simulation_type}_ep{episode:03d}.png",
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"Vehicle breakdown plot saved: {plot_file}")
            plt.close()
        else:
            plt.show()

    def create_all_vehicle_plots(self, episode, simulation_type, save_plots=True):
        """Create comprehensive vehicle analysis plots."""
        print(
            f"\nCreating vehicle plots for {simulation_type.upper()} Episode {episode}..."
        )

        try:
            # Plot 1: Vehicle flow over time
            self.plot_vehicle_flow_over_time(episode, simulation_type, save_plots)

            # Plot 2: Journey analysis
            self.plot_journey_analysis(episode, simulation_type, save_plots)

            # Plot 3: Vehicle type breakdown
            self.plot_vehicle_type_breakdown(episode, simulation_type, save_plots)

            print(
                f"All vehicle plots created for {simulation_type.upper()} Episode {episode}"
            )

        except Exception as e:
            print(f"Error creating vehicle plots: {e}")
