import matplotlib.pyplot as plt
import os

class Visualization:
    """
    A class for visualizing training data and saving plots.
    """

    def __init__(self, path, dpi):
        """
        Initialize the Visualization class with a path and DPI for saving plots.

        Args:
            path (str): Path to save the plots.
            dpi (int): Dots per inch for the saved plots.
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
        with open(os.path.join(self.path, filename + '.txt'), 'w') as f:
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
        fig.savefig(os.path.join(self.path, filename + '.png'), dpi=self.dpi)
        plt.close("all")
    
    def save_data(self, data, filename):
        """
        Save the data to a file.

        Args:
            data (list): Data to be saved.
            filename (str): Name of the file to save the data.
        """
        with open(os.path.join(self.path, filename + '.txt'), 'w') as f:
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
            metrics = ["density_avg", "green_time_avg", "travel_time_avg", "outflow_rate_avg", "loss_avg"]
        if names is None:
            names = ["dqn_qr", "dqn_mse","dqn_huber", "dqn_weighted", "q", "base"]

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

            plt.xlabel("60 seconds")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Comparison of {metric.replace('_', ' ').title()} (Episode {episode})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, f"compare_{metric}_episode_{episode}.png"), dpi=150)
            plt.close()