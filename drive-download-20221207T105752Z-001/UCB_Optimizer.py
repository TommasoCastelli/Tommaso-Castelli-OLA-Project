from pricing import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def run_ucb_test():
    u = UCBOptimizer(False, False, True, True)
    u.test(50)


class UCBOptimizer:

    def __init__(self, random_alphas=True, random_daily_users=False, uncertain_graph_weights=False, binarized=True,
                 change_detection=None):
        self.ps = Pricing_Simulator(random_alphas, random_daily_users, uncertain_graph_weights)
        self.binarized = binarized
        self.posteriors = []
        self.regrets = pd.DataFrame(columns=['P1', 'P2', 'P3', 'P4', 'P5'])
        if change_detection is None:
            for i in range(5):
                self.posteriors.append([UCBPosterior(), UCBPosterior(), UCBPosterior(), UCBPosterior()])
        elif change_detection == "sw":
            for i in range(5):
                self.posteriors.append([UCBPosteriorSW(10), UCBPosteriorSW(10), UCBPosteriorSW(10), UCBPosteriorSW(10)])
        elif change_detection == "cd":
            self.commons = []
            for i in range(5):
                common = CommonCD()
                self.commons.append(common)
                self.posteriors.append(
                    [UCBPosteriorCD(10, common), UCBPosteriorCD(10, common), UCBPosteriorCD(10, common),
                     UCBPosteriorCD(10, common)])

    def pull_arm(self):

        # Check all the posteriors to get the price values
        picks = []
        for i in range(len(self.posteriors)):
            posteriors = self.posteriors[i]

            values = []
            for posterior in posteriors:
                values.append(posterior.get_ucb())

            pick = values.index(max(values))
            self.ps.set_price(i, pick)
            picks.append(pick)

        info = self.ps.run_day(self.binarized, info=True)
        print("day: " + str(self.ps.get_days()))
        print("regret: " + str(info.regrets))
        print("prices: " + str(self.ps.price_indices))
        print("correct prices: " + str(info.correct_prices))
        print("-----------------------------")
        self.regrets.loc[self.ps.get_days()] = info.regrets

        for i in range(len(self.posteriors)):
            UCBPosterior.increment_total_data_size()
            self.posteriors[i][picks[i]].add_data(info.cems[i])

    def test(self, runs=50):
        for i in range(runs):
            self.pull_arm()

        UCBPosterior.reset_totals()

        sumdf = self.regrets.sum(axis=1)
        plt.plot(range(len(sumdf)), sumdf)
        plt.title("UCB algorithm, uncertain conversion rates and graph weights")
        plt.xlabel("Runs")
        plt.ylabel("Regret")
        plt.show()

    def run(self, runs=50):
        for i in range(runs):
            self.pull_arm()

        UCBPosterior.reset_totals()

        sumdf = self.regrets.sum(axis=1)
        return sumdf


class UCBPosterior:
    total_data_size = 0

    @classmethod
    def reset_totals(cls):
        UCBPosterior.total_data_size = 0

    @classmethod
    def increment_total_data_size(cls):
        UCBPosterior.total_data_size += 1

    def __init__(self, constant=1):
        self._data = np.array([])
        self._ucb = 10000
        self._constant = constant

    def get_ucb(self):
        return self._ucb

    def get_mean(self):
        if self._data.size > 0:
            return np.mean(self._data)
        else:
            return 10000

    def add_data(self, value):
        self._data = np.append(self._data, value)
        self._ucb = self.calc_ucb()

    def calc_ucb(self):
        if self._data.size > 0:
            return np.mean(self._data) + self._constant * math.sqrt(
                math.log(UCBPosterior.total_data_size) / self._data.size)


class UCBPosteriorSW(UCBPosterior):
    instances = []

    def __init__(self, constant, window_size):
        super(UCBPosteriorSW, self).__init__(constant)
        self._sliding_window = np.array([])
        self._window_size = window_size
        UCBPosteriorSW.instances.append(self)

    def add_data(self, value):
        super().add_data(value)
        self._sliding_window = np.append(self._sliding_window, UCBPosterior.total_data_size - 1)
        for instance in UCBPosteriorSW.instances:
            instance.slide_window()

    def calc_ucb(self):
        total = self._data.size
        if total > self._window_size:
            total = self._window_size

        if self._data.size > 0:
            return np.mean(self._data) + self._constant * math.sqrt(
                math.log(total) / self._data.size)

    def slide_window(self):
        if UCBPosterior.total_data_size < self._window_size - 1:
            return

        if UCBPosterior.total_data_size - self._window_size in self._data:
            self._data = self._data[1:]
            self._sliding_window = self._sliding_window[1:]


# Gör datan som granskas för förändringar till en klassvariabel.
class UCBPosteriorCD(UCBPosterior):

    def __init__(self, constant, common):
        super(UCBPosteriorCD, self).__init__(constant)
        self.common = common
        self.common.add_parent(self)

    def add_data(self, value):
        super(UCBPosteriorCD, self).add_data(value)
        self.common.add_value(value)

    def reset(self):
        self._data = np.array([])
        self._ucb = 10000


class CommonCD:

    def __init__(self):
        self.data = np.array([])
        self.parents = []

    def reset(self):
        self.data = np.array([])
        for parent in self.parents:
            parent.reset()

    def is_outlier(self, value):
        std = np.std(self.data)
        mean = np.mean(self.data)

        if mean - 2 * std < value < mean + 2 * std:
            return False
        else:
            return True

    def add_value(self, value):
        self.data = np.append(self.data, value)
        if self.is_change():
            print("Change occured!!")
            self.reset()

    def add_parent(self, parent):
        self.parents.append(parent)

    def is_change(self):
        if self.data.size < 10:
            return False

        old = self.data[:-2]
        new = self.data[-2:]

        for value in new:
            if not self.is_outlier(value):
                return False
        return True


def test_cd(runs, change):
    common = CommonCD()
    post = UCBPosteriorCD(1, common)
    post2 = UCBPosteriorCD(2, common)
    post3 = UCBPosteriorCD(2, common)

    data = np.array([])
    mean = np.array([])
    ucb = np.array([])

    for i in range(runs):
        if i < change:
            value = np.random.normal(10, 1)
            post.add_data(value)
        else:
            value = np.random.normal(14, 1)
            post.add_data(value)

        data = np.append(data, value)
        if post._data.size == 0:
            mean_value = 0
        else:
            mean_value = np.mean(post._data)
        mean = np.append(mean, mean_value)
        ucb = np.append(ucb, post.get_ucb())
        x = range(runs)

    plt.plot(x, data, label="data")
    plt.plot(x, mean, label="Posterior data mean")
    plt.plot(x, ucb, label="UCB")
    plt.legend()
    plt.title("Change detection algorithm test")
    ax = plt.gca()
    ax.set_ylim([0, 20])
    plt.show()
