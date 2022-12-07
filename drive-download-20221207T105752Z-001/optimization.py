# This is where step 2 in the pricing project is handled
import numpy as np

# Imports
from pricing import *
import matplotlib.pyplot as plt


class Greedy_Optimizer:

    def __init__(self):
        self.ps = Pricing_Simulator()
        self.runs = []

    def greedy_optimize_prices(self):

        info = self.ps.run_day(binarized=True, info=True)
        self.runs.append(info)
        current_cem = info.cem  # cem is for cumulative expected margin

        change = False
        while not change:
            change = True
            for i in range(self.ps.number_of_products):
                if self.ps.pi(i) < 3:
                    self.ps.increase_price(i)
                    info = self.ps.run_day(binarized=True, info=True)
                    self.runs.append(info)
                    new_cem = info.cem
                    if new_cem > current_cem:
                        current_cem = new_cem
                        change = False
                        continue
                    else:
                        self.ps.decrease_price(i)

        return self.ps.price_indices

    def run(self, days):
        optimal_prices = self.greedy_optimize_prices()

        if len(self.runs) < days:
            days_left = days - len(self.runs)
            for i in range(days_left):
                info = self.ps.run_day(binarized=True, info=True)
                self.runs.append(info)
        elif len(self.runs) > days:
            self.runs = self.runs[:days]

        regrets = np.array([])
        for run in self.runs:
            regrets = np.append(regrets, sum(run.regrets))

        return regrets
