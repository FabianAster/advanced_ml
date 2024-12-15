import pandas as pd
import matplotlib.pyplot as plt


def plot_barchart(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert 'value' column to numeric, handling missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df["number"], df["value"], color="blue")
    plt.xlabel("Number")
    plt.ylabel("Value")
    plt.title("Bar Chart of Values")
    plt.show()


# Example usage
file_path = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/ddpg/ddpg/report_Pendulum-v1_50-trials-25000-tpe-median_1733875694.csv"
plot_barchart(file_path)
file_path = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/td3/td3/report_Pendulum-v1_50-trials-25000-tpe-median_1733889639.csv"
plot_barchart(file_path)
file_path = "/home/fabian/github/advancedML/Exercise2/rl-baselines3-zoo/zac/zac/report_Pendulum-v1_50-trials-25000-tpe-median_1733960856.csv"
plot_barchart(file_path)
