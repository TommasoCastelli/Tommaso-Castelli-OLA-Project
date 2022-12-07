import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tsoptimizer import TSOptimizer
from UCB_Optimizer import UCBOptimizer
from optimization import Greedy_Optimizer


def ts_optimizer_mc(runs, days):
    data = pd.DataFrame()
    for i in range(runs):
        print("Run nr " + str(i))
        ts = TSOptimizer(False, False, False, True)
        result = ts.run(days)
        data[i] = result

    print(data)

    means = np.array([])
    stds = np.array([])
    for i in range(days):
        means = np.append(means, np.mean(data.iloc[i, :].values))
        stds = np.append(stds, np.std(data.iloc[i, :].values))

    fig, axs = plt.subplots(2)
    fig.suptitle("UCB algorithm, uncertain conversion rate. 50 runs")
    axs[0].plot(range(days), means)
    axs[0].set_title("Mean")
    axs[1].plot(range(days), stds, color="red")
    axs[1].set_title("Standard deviation")
    for ax in axs.flat:
        ax.set(ylabel='Regret')
    axs.flat[-1].set(xlabel='Day')
    plt.show()


def ucb_optimizer_mc(runs, days):
    data = pd.DataFrame()
    for i in range(runs):
        print("Run nr " + str(i))
        ucb = UCBOptimizer(False, False, False, True)
        result = ucb.run(days)
        data[i] = result

    print(data)

    means = np.array([])
    stds = np.array([])
    for i in range(days):
        means = np.append(means, np.mean(data.iloc[i, :].values))
        stds = np.append(stds, np.std(data.iloc[i, :].values))

    fig, axs = plt.subplots(2)
    fig.suptitle("UCB algorithm, uncertain conversion rate. 50 runs")
    axs[0].plot(range(days), means)
    axs[0].set_title("Mean")
    axs[1].plot(range(days), stds)
    axs[1].set_title("Standard deviation")
    for ax in axs.flat:
        ax.set(xlabel='Days', ylabel='Regret')
    plt.show()


def greedy_optimizer_mc(runs, days):
    data = pd.DataFrame()
    for i in range(runs):
        print("Run nr " + str(i))
        gr = Greedy_Optimizer()
        result = gr.run(days)
        data[i] = result

    print(data)

    means = np.array([])
    stds = np.array([])
    for i in range(days):
        means = np.append(means, np.mean(data.iloc[i, :].values))
        stds = np.append(stds, np.std(data.iloc[i, :].values))

    fig, axs = plt.subplots(2)
    fig.suptitle("UCB algorithm, uncertain conversion rate. 50 runs")
    axs[0].plot(range(days), means)
    axs[0].set_title("Mean")
    axs[1].plot(range(days), stds)
    axs[1].set_title("Standard deviation")
    for ax in axs.flat:
        ax.set(xlabel='Days', ylabel='Regret')
    plt.show()
