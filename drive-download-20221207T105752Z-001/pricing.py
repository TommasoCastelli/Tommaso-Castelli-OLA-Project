
from simulator import *
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm


class User_Class:

	def __init__(self, price_ranges, random_alphas=True, random_daily_users=False):
		self.price_ranges = price_ranges
		self.number_of_products_sold = np.zeros(5)
		self.daily_users = 0
		self.new_day(random_daily_users)
		self.alphas = self.generate_alphas(5, self.daily_users, random_alphas)
		self.reservation_price_distributions = []
		self.generate_reservation_prices(self.price_ranges)
		self.graph = Graph()
		self.graph.setup(self.alphas, np.zeros(5))

	def generate_reservation_prices(self, price_ranges):
		
		# Weight

		for i in range(len(price_ranges)):
			mean = np.random.uniform(price_ranges[i].r(0), price_ranges[i].r(-1))
			std = np.random.exponential(np.log(mean))
			distribution = Reservation_Price_Distribution(mean,std)
			self.reservation_price_distributions.append(distribution)

	def new_day(self, random=False):
		daily_users = 0
		if random:
			daily_users = np.random.normal(1000, 100)
		else:
			daily_users = 1000
		self.daily_users = daily_users

	def generate_user(self):
		user = User()

		reservation_prices = []
		for i in range(5):
			reservation_prices.append(self.reservation_price_distributions[i].get_reservation_price())
		user.set_reservation_price(reservation_prices)

		return user

	# Generates alpha values. n is the number of products.
	def generate_alphas(self, n, n_MAX_customers, random=True):
		if not random:
			alphas = np.array([3, 1, 1, 1, 1, 1])
			alphas = alphas / 8
			return alphas

		size = n+1
		alpha = np.zeros(size) # ?? aplha 0 => competitor website, alpha 1 => product 1, ...

		alpha_ratio = np.zeros(size)
		for i in range(size):
			if i == 0:
				alpha_ratio[i] = np.random.poisson(5) * 2 + 1
			else:
				alpha_ratio[i] = np.random.poisson(5) + 1
		alpha_ratio = alpha_ratio / np.sum(alpha_ratio)

		alpha = np.random.multinomial(n_MAX_customers, alpha_ratio)

		alpha_noise = np.random.dirichlet(alpha,size=None)
		return alpha_noise

class Reservation_Price_Distribution:

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	# Maybe fix sometimes everyone is in range
	def get_reservation_price(self):
		return np.random.normal(self.mean, self.std)

	def set_mean(self, mean):
		self.mean = mean

	def set_std(self, std):
		self.std = std

class Price_Range:

	def __init__(self):
		self.price_range = self.generate_price_range()
		self.price = np.mean(self.price_range)
		self.cost = self.generate_cost(self.price_range)

	def generate_price_range(self):
		price_scalars = np.array([0.7, 0.9, 1.1, 1.3])
		price = np.random.uniform(10, 100)
		price_range = price * price_scalars
		return price_range

	def generate_cost(self, price_range):
		center = price_range[0]
		std = np.mean(price_range) * 0.1
		return np.random.normal(center, std)

	# Returns the margin of the product
	def margin(self, index):
		return self.price_range[index] - self.cost

	# Returns the price in the range at an index
	def r(self, index):
		return self.price_range[index]


class Pricing_Simulator:

	def __init__(self, random_alphas=True, random_daily_users=False, uncertain_graph_weights=False):
		self.day = 0
		self.simulator = Simulator()
		self.number_of_products = 5
		self.simulator.setup(self.number_of_products, uncertain_graph_weights)
		self.price_ranges = self.generate_price_ranges()
		self.user_classes = [User_Class(self.price_ranges, random_alphas, random_daily_users), User_Class(self.price_ranges, random_alphas, random_daily_users), User_Class(self.price_ranges, random_alphas, random_daily_users)]
		self.price_indices = [0] * self.number_of_products

	def generate_price_ranges(self):
		price_ranges = []
		for i in range(self.number_of_products):
			price_ranges.append(Price_Range())
		return price_ranges

	def run_day(self, binarized=False, verbose=0, info=False, random_daily_users=False):
		self.day += 1

		# GENERATE NEW PRICES
		prices = np.zeros(5)
		for i in range(5):
			prices[i] = self.price_ranges[i].r(self.price_indices[i])
		self.simulator.set_prices(prices)

		# New users
		for i in range(len(self.user_classes)):
			self.user_classes[i].new_day(random_daily_users)

		total_products = np.zeros(5)
		for i in range(len(self.user_classes)):
			self.simulator.set_alphas(self.user_classes[i].alphas)
			for j in range(self.user_classes[i].daily_users):
				user = self.user_classes[i].generate_user()
				cart = self.simulator.run(user)
				if binarized:
					total_products = total_products + self.bin_array(cart)
				else:
					total_products = total_products + cart

		if verbose == 1:
			print("UC 1 number of users: " + str(self.user_classes[0].daily_users))
			print("UC 1 alphas: " + str(self.user_classes[0].alphas))

			print("UC 2 number of users: " + str(self.user_classes[1].daily_users))
			print("UC 2 alphas: " + str(self.user_classes[1].alphas))

			print("UC 3 number of users: " + str(self.user_classes[2].daily_users))
			print("UC 3 alphas: " + str(self.user_classes[2].alphas))

			print("Prices are: " + str(prices))

			print("conversion rate for the products are: " + str(total_products / (self.user_classes[0].daily_users + self.user_classes[1].daily_users + self.user_classes[2].daily_users)))

		visitors = self.user_classes[0].daily_users + self.user_classes[1].daily_users + self.user_classes[2].daily_users

		if info:
			cems = []
			regrets = []
			correct_prices = []
			for i in range(5):
				clairvoyance = self.clairvoyance(i, binarized)
				cems.append(self.price_ranges[i].margin(self.price_indices[i]) * total_products[i] / visitors)
				regrets.append(max(clairvoyance) - cems[i])
				correct_prices.append(clairvoyance.index(max(clairvoyance)))
			return Info(sum(cems), total_products, cems, regrets, correct_prices)
		else:
			return self.cem(total_products, visitors)

	def cem(self, sales, visitors):
		margin = 0
		for i in range(self.number_of_products):
			margin += (sales[i] * self.price_ranges[i].margin(self.price_indices[i])) / visitors

		if margin < 0:
			prices = np.zeros(5)
			for i in range(self.number_of_products):
				prices[i] = self.price_ranges[i].margin(self.price_indices[i])

			#print("CEM is negative and these are the prices: " + np.array2string(prices))

		return margin

	def bin_array(self, array):
		a = np.copy(array)
		for i in range(len(a)):
			if a[i] > 0:
				a[i] = 1
		return a

	def get_days(self):
		return self.day

	def set_price(self, index, price_index):
		if price_index > -1 and price_index < 4:
			self.price_indices[index] = price_index
		else:
			raise Exception("Price index out of bounds in set_price()")

	# short for price_indices
	def pi(self, index=None):
		if index != None:
			return self.price_indices[index]
		else:
			return self.price_indices

	def increase_price(self, index):
		if self.price_indices[index] != 3:
			self.price_indices[index] += 1

	def decrease_price(self, index):
		if self.price_indices[index] != 0:
			self.price_indices[index] -= 1

	def clairvoyance(self, product, binarized):
		# To scale the reward of the sales by the expected number of products sold.
		scalar = 0
		if binarized:
			scalar = 1
		else:
			scalar = 2

		rewards = []
		price_range = self.price_ranges[product].price_range
		for i in range(len(price_range)):

			probs = np.array([])
			for j in range(len(self.user_classes)):
				mean = self.user_classes[j].reservation_price_distributions[product].mean
				std = self.user_classes[j].reservation_price_distributions[product].std
				probs = np.append(probs, 1 - norm.cdf(price_range[i], mean, std))
			prob = np.mean(probs)
			rewards.append(prob*self.price_ranges[product].margin(i) * scalar)
		
		return rewards

class Info:

	def __init__(self, cem, sales, cems, regrets=[], correct_prices=[]):
		self.cem = cem
		self.sales = sales
		self.cems = cems
		self.regrets = regrets
		self.correct_prices = correct_prices
		



	