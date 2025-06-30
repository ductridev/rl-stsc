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