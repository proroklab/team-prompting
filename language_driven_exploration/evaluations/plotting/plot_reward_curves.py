# Script to load csv files and plot mean reward curves for my MARL training
# Files have the following structure: First Collumn = Steps, Second Collumn = Mean Reward
# All Files are found in evaluations/data/training_mean_rewards/

import pandas as pd
import matplotlib.pyplot as plt
import os   

def plot_reward_curves(filenames):      
    for filename in filenames:
        df = pd.read_csv(filename, header=None)
        filename = os.path.basename(filename)[18:-4]
        plt.plot(df[0], df[1], label=filename)
    plt.xlabel("Steps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()

# Filenames are loaded from the data directory
directory = "evaluations/data/training_mean_rewards/"
# Read all the files in the directory
filenames = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.endswith(".csv")
]

plot_reward_curves(filenames)