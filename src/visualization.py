import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class Visualization:
    """
    A class for visualizing training data and saving plots.
    """

    def __init__(self, path, dpi=150):
        """
        Initialize the Visualization class with a path and DPI for saving plots.

        Args:
            path (str): Path to save the plots.
            dpi (int): Dots per inch for the saved plots (default: 150).
        """
        self.path = path
        self.dpi = dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Save the data to a file and create a plot.

        Args:
            data (list): Data to be saved and plotted.
            filename (str): Name of the file to save the data.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        # Save data to a text file
        with open(os.path.join(self.path, filename + ".txt"), "w") as f:
            for item in data:
                f.write("%s\n" % item)

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

        fig = plt.gcf()

        # Save the plot
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self.path, filename + ".png"), dpi=self.dpi)
        plt.close("all")

    def save_data(self, data, filename):
        """
        Save the data to a file.

        Args:
            data (list): Data to be saved.
            filename (str): Name of the file to save the data.
        """
        with open(os.path.join(self.path, filename + ".txt"), "w") as f:
            for item in data:
                f.write("%s\n" % item)

    def save_plot(self, episode=0, metrics=None, names=None):
        """
        Plot and compare multiple metrics (e.g., density_avg, green_time_avg, travel_time_avg)
        for DQN, Q, and Base, based on file naming convention.

        Args:
            path (str): Directory where the metric files are saved.
            episode (int): Episode number to load (default: 0).
            metrics (list): List of metric names to compare (e.g., ["density_avg", "green_time_avg"]).
            names (list): List of experiment names (default: ["dqn", "q", "base"]).
        """
        if metrics is None:
            metrics = [
                "density_avg",
                "green_time_avg",
                "travel_time_avg",
                "outflow_rate_avg",
                "loss_avg",
            ]
        if names is None:
            names = ["dqn_qr", "dqn_mse", "dqn_huber", "dqn_weighted", "q", "base"]

        for metric in metrics:
            data = {}
            for name in names:
                filename = os.path.join(
                    self.path, f"{name}_{metric}_episode_{episode}.txt"
                )
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        data[name] = [float(line.strip()) for line in f if line.strip()]
                else:
                    print(f"File not found: {filename}")

            plt.figure(figsize=(12, 7))
            for name in names:
                if name in data:
                    plt.plot(data[name], label=name.upper())

            plt.xlabel("Minute")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(
                f"Comparison of {metric.replace('_', ' ').title()} (Episode {episode})"
            )
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.path, f"compare_{metric}_episode_{episode}.png"),
                dpi=150,
            )
            plt.close()

    def save_comparison_plots(self, episode=0, metrics=None, names=None):
        """
        Plot per-traffic-light comparison metrics over episodes
        (density, travel_time, outflow, etc.) for different simulation types.

        Args:
            episode (int): Current episode number to plot up to.
            metrics (list): List of metric names (e.g., ["density", "travel_time"]).
            names (list): Simulation types to compare (e.g., ["baseline", "dqn_qr"]).
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import os

        if metrics is None:
            metrics = ["reward", "queue_length", "travel_delay", "waiting_time", "outflow"]

        if names is None:
            names = ["baseline", "skrl_dqn", "actuacted"]

        for metric in metrics:
            print(f"Generating plot for metric: {metric}")
            plt.ioff()
            fig, ax = plt.subplots()
            episodes = []
            tl_data = {sim_type: {} for sim_type in names}

            # Process data up to the current episode
            for ep in range(0, episode):  
                filename = os.path.join(self.path, f"comparison_per_tl_{metric}_episode_{ep}.csv")
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    episodes.append(ep)
                    for tl_id in df['traffic_light_id']:
                        for sim_type in names:
                            if sim_type in df.columns:
                                value = df.loc[df['traffic_light_id'] == tl_id, sim_type].values
                                if len(value) > 0:
                                    if tl_id not in tl_data[sim_type]:
                                        tl_data[sim_type][tl_id] = []
                                    tl_data[sim_type][tl_id].append(value[0])

            # Plotting per TL per Simulation Type
            has_data = False
            for sim_type in names:
                for tl_id, values in tl_data[sim_type].items():
                    ax.plot(episodes[:len(values)], values, marker='o', label=f'{sim_type} - {tl_id}')
                    has_data = True

            if has_data:
                ax.set_xlabel('Episode')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Over Episodes')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                img_filename = os.path.join(self.path, f"comparison_{metric}_over_episodes.png")
                plt.savefig(img_filename, dpi=150)
                plt.close(fig)
                print(f"Plot saved to {img_filename}")
            else:
                print(f"No data available for metric: {metric}, skipping plot.")


    def create_vehicle_comparison_from_logs(self, episode, simulation_types):
        """
        Create vehicle comparison plots by reading saved CSV log files.

        Args:
            episode (int): Episode number to analyze
            simulation_types (list): List of simulation types to compare
        """
        # Load data from CSV files
        data = {}
        colors = {"qlearning": "blue", "skrl_dqn": "red", "actuated": "green"}

        print(f"Loading vehicle data for episode {episode}...")

        for sim_type in simulation_types:
            csv_file = os.path.join(
                self.path, "logs", f"vehicle_details_{sim_type}_ep{episode:03d}.csv"
            )
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    data[sim_type] = df
                    print(f"Loaded {sim_type}: {len(df)} data points")
                except Exception as e:
                    print(f"Error loading {sim_type}: {e}")
            else:
                print(f"File not found: {csv_file}")

        if not data:
            print("No vehicle data files found for comparison")
            return

        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Vehicle Performance Comparison from Logs - Episode {episode}", fontsize=16
        )

        # Plot 1: Network Load Comparison (Running Vehicles)
        for sim_type, df in data.items():
            color = colors.get(sim_type, "gray")
            label = sim_type.upper().replace("_", " ")
            ax1.plot(
                df["step"], df["total_running"], label=label, color=color, linewidth=2
            )

        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Vehicles in Network")
        ax1.set_title("Network Load Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final Completion Rates (Bar Chart)
        completion_rates = {}
        for sim_type, df in data.items():
            final_departed = df["total_departed"].iloc[-1] if len(df) > 0 else 0
            final_arrived = df["total_arrived"].iloc[-1] if len(df) > 0 else 0
            completion_rate = (
                (final_arrived / final_departed * 100) if final_departed > 0 else 0
            )
            completion_rates[sim_type] = completion_rate

        if completion_rates:
            sim_names = [
                sim_type.upper().replace("_", " ")
                for sim_type in completion_rates.keys()
            ]
            rates = list(completion_rates.values())
            bar_colors = [
                colors.get(sim_type, "gray") for sim_type in completion_rates.keys()
            ]

            bars = ax2.bar(sim_names, rates, color=bar_colors, alpha=0.7)
            ax2.set_ylabel("Completion Rate (%)")
            ax2.set_title("Final Completion Rates")
            ax2.set_ylim(0, 100)

            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # Plot 3: Vehicle Type Distribution (Stacked Bar)
        if data:
            vehicle_types = ["bike", "car", "truck"]
            type_data = {}

            for sim_type, df in data.items():
                if len(df) > 0:
                    type_data[sim_type] = {
                        "bike": df["departed_bike"].iloc[-1],
                        "car": df["departed_car"].iloc[-1],
                        "truck": df["departed_truck"].iloc[-1],
                    }

            if type_data:
                sim_names = [
                    sim_type.upper().replace("_", " ") for sim_type in type_data.keys()
                ]
                x_pos = np.arange(len(sim_names))

                bike_counts = [
                    type_data[sim_type]["bike"] for sim_type in type_data.keys()
                ]
                car_counts = [
                    type_data[sim_type]["car"] for sim_type in type_data.keys()
                ]
                truck_counts = [
                    type_data[sim_type]["truck"] for sim_type in type_data.keys()
                ]

                bottom_car = bike_counts
                bottom_truck = [b + c for b, c in zip(bike_counts, car_counts)]

                ax3.bar(x_pos, bike_counts, label="Bike", color="lightgreen")
                ax3.bar(
                    x_pos, car_counts, bottom=bottom_car, label="Car", color="lightblue"
                )
                ax3.bar(
                    x_pos,
                    truck_counts,
                    bottom=bottom_truck,
                    label="Truck",
                    color="lightcoral",
                )

                ax3.set_xlabel("Simulation Type")
                ax3.set_ylabel("Number of Vehicles")
                ax3.set_title("Vehicle Type Distribution")
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(sim_names, rotation=45, ha="right")
                ax3.legend()

        # Plot 4: Completion Rate Over Time
        for sim_type, df in data.items():
            if len(df) > 0:
                completion_rate_over_time = (
                    df["total_arrived"] / df["total_departed"].replace(0, 1)
                ) * 100
                color = colors.get(sim_type, "gray")
                label = sim_type.upper().replace("_", " ")
                ax4.plot(
                    df["step"],
                    completion_rate_over_time,
                    label=label,
                    color=color,
                    linewidth=2,
                )

        ax4.set_xlabel("Simulation Step")
        ax4.set_ylabel("Completion Rate (%)")
        ax4.set_title("Completion Rate Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(
            self.path, "logs", f"vehicle_comparison_from_logs_ep{episode:03d}.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Vehicle comparison plot saved: {plot_file}")

        # Print summary statistics
        print(f"Vehicle Performance Summary - Episode {episode}")
        print("=" * 60)
        for sim_type, df in data.items():
            if len(df) > 0:
                final_departed = df["total_departed"].iloc[-1]
                final_arrived = df["total_arrived"].iloc[-1]
                completion_rate = (
                    (final_arrived / final_departed * 100) if final_departed > 0 else 0
                )
                max_running = df["total_running"].max()

                print(
                    f"{sim_type.upper():<12}: "
                    f"Departed={final_departed:>4}, "
                    f"Arrived={final_arrived:>4}, "
                    f"Completion={completion_rate:>5.1f}%, "
                    f"Max Load={max_running:>3}"
                )

    def plot_completed_travel_time_comparison(self, results, episode=None, save_path=None):
        """
        Create comparison plots for average total travel time of completed vehicles.
        
        Args:
            results (dict): Results dictionary with completion tracker data for each method
            episode (int): Episode number for labeling
            save_path (str): Optional custom save path, defaults to self.path
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if save_path is None:
            save_path = self.path
        
        # Extract data from results
        methods = []
        completed_counts = []
        avg_travel_times = []
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
        
        method_mapping = {
            'actuated': 'Research Actuated',
            'baseline': 'Baseline (Fixed)', 
            'dqn': 'SKRL DQN'
        }
        
        for method_key in ['actuated', 'baseline', 'dqn']:
            if method_key in results and 'completion_tracker' in results[method_key]:
                completion_data = results[method_key]['completion_tracker']
                methods.append(method_mapping[method_key])
                completed_counts.append(completion_data['completed_count'])
                avg_travel_times.append(completion_data['avg_travel_time'])
        
        if not methods:
            print("No completion tracker data available for plotting")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Completed Vehicle Count
        bars1 = ax1.bar(methods, completed_counts, color=colors[:len(methods)], alpha=0.7)
        ax1.set_title('Number of Completed Vehicles', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Completed Vehicle Count (number vehicles)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        max_count = max(completed_counts) if completed_counts else 0
        count_offset = max_count * 0.01 if max_count > 0 else 1.0  # Default offset if all zero
        
        for bar, count in zip(bars1, completed_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count_offset,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Average Total Travel Time
        bars2 = ax2.bar(methods, avg_travel_times, color=colors[:len(methods)], alpha=0.7)
        ax2.set_title('Average Total Travel Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Travel Time (seconds)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        max_time = max(avg_travel_times) if avg_travel_times else 0
        label_offset = max_time * 0.01 if max_time > 0 else 1.0  # Default offset if all zero
        
        for bar, time_val in zip(bars2, avg_travel_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best performance (only if we have positive travel times)
        if avg_travel_times:
            positive_times = [t for t in avg_travel_times if t > 0]
            if positive_times:  # Only highlight if there are positive travel times
                # Find the index of the minimum positive time in the original list
                min_positive_time = min(positive_times)
                best_idx = avg_travel_times.index(min_positive_time)
                bars2[best_idx].set_edgecolor('gold')
                bars2[best_idx].set_linewidth(3)
        
        # Overall title
        episode_text = f" - Episode {episode}" if episode is not None else ""
        fig.suptitle(f'Completed Vehicle Travel Time Comparison{episode_text}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"completed_travel_time_comparison_ep{episode}.png" if episode else "completed_travel_time_comparison.png"
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Completion tracker comparison plot saved: {plot_path}")
        
        return plot_path
    
    def plot_travel_time_improvements(self, results, episode=None, save_path=None):
        """
        Create a plot showing travel time improvements relative to baseline.
        
        Args:
            results (dict): Results dictionary with completion tracker data
            episode (int): Episode number for labeling
            save_path (str): Optional custom save path
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if save_path is None:
            save_path = self.path
            
        if 'baseline' not in results or 'completion_tracker' not in results['baseline']:
            print("No baseline data available for improvement calculation")
            return
            
        baseline_time = results['baseline']['completion_tracker']['avg_travel_time']
        if baseline_time <= 0:
            print("Invalid baseline travel time for improvement calculation")
            return
        
        # Calculate improvements
        methods = []
        improvements = []
        colors = []
        
        method_data = {
            'actuated': ('Research Actuated', '#A23B72'),
            'dqn': ('SKRL DQN', '#2E86AB')
        }
        
        for method_key, (method_name, color) in method_data.items():
            if method_key in results and 'completion_tracker' in results[method_key]:
                method_time = results[method_key]['completion_tracker']['avg_travel_time']
                if method_time > 0:
                    improvement = ((baseline_time - method_time) / baseline_time) * 100
                    methods.append(method_name)
                    improvements.append(improvement)
                    colors.append(color if improvement > 0 else '#FF6B6B')  # Red for negative
        
        if not methods:
            print("No improvement data available for plotting")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(methods, improvements, color=colors, alpha=0.7)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize plot
        ax.set_title('Travel Time Improvement vs Baseline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        if improvements:
            improvement_range = max(improvements) - min(improvements)
            offset_factor = improvement_range * 0.01 if improvement_range > 0 else 1.0
        else:
            offset_factor = 1.0
            
        for bar, improvement in zip(bars, improvements):
            y_pos = bar.get_height() + offset_factor
            if improvement < 0:
                y_pos = bar.get_height() - offset_factor
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{improvement:+.1f}%', ha='center', va='bottom' if improvement >= 0 else 'top',
                   fontweight='bold')
        
        # Overall title
        episode_text = f" - Episode {episode}" if episode is not None else ""
        ax.set_title(f'Travel Time Improvement vs Baseline{episode_text}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"travel_time_improvements_ep{episode}.png" if episode else "travel_time_improvements.png"
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Travel time improvements plot saved: {plot_path}")
        
        return plot_path
