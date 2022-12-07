# This is the thompson sampling solution

# Inspired by: https://www.youtube.com/watch?v=Zgwfw3bzSmQ

from pricing import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def run_ts_test():
    ts = TSOptimizer(True, True, False)
    ts.test(50)


class TSOptimizer:

    def __init__(self, random_alphas=True, random_daily_users=False, uncertain_graph_weights=False, binarized=True):
        self.ps = Pricing_Simulator(random_alphas, random_daily_users, uncertain_graph_weights)
        self.binarized = binarized
        self.posteriors = []
        self.regrets = pd.DataFrame(columns=['P1', 'P2', 'P3', 'P4', 'P5'])
        for i in range(5):
            self.posteriors.append([Posterior(), Posterior(), Posterior(), Posterior()])

    def pull_arm(self):
        picks = []
        for i in range(len(self.posteriors)):
            posteriors = self.posteriors[i]

            values = []
            for posterior in posteriors:
                values.append(posterior.get_value())

            pick = values.index(max(values))
            self.ps.set_price(i, pick)
            picks.append(pick)

        info = self.ps.run_day(self.binarized, info=True)
        #print("day: " + str(self.ps.get_days()))
        #print("regret: " + str(info.regrets))
        #print("prices: " + str(self.ps.price_indices))
        #print("correct prices: " + str(info.correct_prices))
        #print("-----------------------------")
        self.regrets.loc[self.ps.get_days()] = info.regrets

        for i in range(len(self.posteriors)):
            self.posteriors[i][picks[i]].add_data(info.cems[i])

    def test(self, runs=50):
        for i in range(runs):
            self.pull_arm(True)

        sumdf = self.regrets.sum(axis=1)
        plt.plot(range(len(sumdf)), sumdf)
        plt.title("Thompson sampling, uncertain conversion rates, alpha values and number of items sold")
        plt.xlabel("Runs")
        plt.ylabel("Regret")
        plt.show()

    def run(self, runs=50):
        for i in range(runs):
            self.pull_arm()

        sumdf = self.regrets.sum(axis=1)
        return sumdf

    def print_posteriors(self, index):
        for i in range(len(self.posteriors[index])):
            print("mean, std and n from posterior: " + str(i))
            print(np.mean(self.posteriors[index][i].data))
            print(self.posteriors[index][i].std)
            print(self.posteriors[0][i].data.size)

    def plot_regret(self):
        t = np.arange(0, self.regrets.size)
        regrets = np.zeros(self.regrets.size)
        for i in range(self.regrets.size):
            regrets[i] = np.sum(regrets[i])

        fig, ax = plt.subplots()
        ax.plot(t, regrets)
        ax.set(xlabel='days', ylabel='regret', title='Total regret each day')
        ax.grid()
        plt.show()


class Posterior:

    def __init__(self):
        self.data = np.array([])
        self.std = self.calc_std()

    def get_value(self):
        mean = self.get_mean()
        return np.random.normal(mean, self.std)

    def get_mean(self):
        if self.data.size > 0:
            return np.mean(self.data)
        else:
            return 100000

    def add_data(self, value):
        self.data = np.append(self.data, value)
        self.std = self.calc_std()

    def calc_std(self):
        if self.data.size > 1:
            return np.std(self.data)
        else:
            return 10000
