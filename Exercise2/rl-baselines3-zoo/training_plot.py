import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def plot_monitor_files(directory, window_size=10):
    # Find all 0.monitor.csv files in the directory
    file_paths = glob.glob(os.path.join(directory, "**", "0.monitor.csv"), recursive=True)

    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        # Read the CSV file, skipping the first row which is a comment
        df = pd.read_csv(file_path, skiprows=1)
        df["r"] = pd.to_numeric(df["r"], errors="coerce")  # Convert 'r' column to numeric

        # Calculate the running average
        df["r_avg"] = df["r"].rolling(window=window_size).mean()

        # Plot the running average with the line number as the x-axis
        plt.plot(df.index, df["r_avg"], label=os.path.basename(os.path.dirname(file_path)))

    plt.xlabel("Line Number")
    plt.ylabel("Reward")
    plt.title("Reward over Line Number with Running Average")
    plt.legend()
    plt.show()


# Example usage
directory = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/ddpg_train"
plot_monitor_files(directory)
directory = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/td3_train"
plot_monitor_files(directory)
directory = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/sac_train"
plot_monitor_files(directory)
