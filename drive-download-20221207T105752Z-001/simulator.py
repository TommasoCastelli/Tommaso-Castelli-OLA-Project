# This is the file for the simulator. Its intended to be imported into other files for use.
# Vertex and Graph would be private but python does not allow such. They are not intended to be used outside this file.

# Inspired by https://www.bogotobogo.com/python/python_graph_data_structures.php

## Libraries
import numpy as np
import math
import random

class Vertex:
	def __init__(self, node, alpha):
		self.id = node
		self.alpha = alpha
		self.adjacent = {}
		self.price = 0

	def add_neighbor(self, neighbor, beta=0):
		self.adjacent[neighbor] = beta

	# This function draws numbers from a beta distribution based on the alpha values of the connections.
	# The largest draw is then the product shown as the first secondary, then the second is picked in the same way
	# among the remaining connections.
	# Also the random variables are normalized and returned to be used as probabilities
	# NOTE: if there are only one connections this will probably crash
	def pick_display(self):
		edges = self.get_connections()

		if len(edges) == 0:
			return "exit"
		elif len(edges) == 1:
			return [edges[0], np.random.beta(self.get_beta(edges[0]), self.alpha, size=None)]

		rolls = []
		for edge in edges:
			rolls.append(np.random.beta(self.get_beta(edge), self.alpha, size=None))
		x1 = max(rolls)
		first = edges[rolls.index(x1)]
		rolls[rolls.index(x1)] = 0
		x2 = max(rolls)
		second = edges[rolls.index(x2)]
		x2 = x2 / (sum(rolls) + x1) 
		x1 = x1 / (sum(rolls) + x1)
		return [first, x1, second, x2]

	def get_connections(self):
		return list(self.adjacent.keys())
	
	def get_id(self):
		return self.id

	def get_alpha(self):
		return self.alpha
	
	def get_beta(self, neighbor):
		return self.adjacent[neighbor]

	def set_price(self, price):
		self.price = price

	def remove_connection(self, connection):
		if connection in self.adjacent.keys():
			del self.adjacent[connection]

class Graph:
	def __init__(self, nodes=0):
		self.vertices = {}
		self.lamb = 0.5
		self.picked = []

	def add_vertex(self, id, alpha):
		self.vertices[id] = Vertex(id,alpha)

	def set_price(self, id, price):
		self.vertices[id].set_price(price)
	
	def add_edge(self, frm, to, uncertain=False, fully_connected=True):
		alpha = self.vertices[to].get_alpha()
		if not fully_connected and np.random.uniform() < 0.2:
			beta = 0
		elif uncertain:
			beta = np.random.normal(alpha, alpha / 6)
		else:
			beta = alpha

		self.vertices[frm].add_neighbor(to, beta)

	def setup(self, alphas, prices, uncertain=False):
		for i in range(1,len(alphas)):
			self.add_vertex("P" + str(i), alphas[i])
			self.set_price("P" + str(i), prices[i-1])

		for i in range(1,len(alphas)):
			for j in range(1,len(alphas)):
				if i != j:
					self.add_edge("P" + str(i), "P" + str(j), uncertain)

	# Resets graph to initial condition
	def reset(self):
		self.picked = []
	
	# This function wipes the Graph
	def clear(self):
		self.vertices = {}
		self.picked = []

	def remove_vertex(self, vertex):
		keys = list(self.vertices.keys())[:]
		for key in keys:
			if key == vertex:
				del self.vertices[key]
			else:
				self.vertices[key].remove_connection(vertex)

	# Recently changed to be able to display multiple times but not to pick same twice
	def pick_new_product(self, start):
		self.set_picked(start)
		display = self.vertices[start].pick_display()

		new_tabs = []
		if display == "exit":
			return new_tabs

		if (np.random.uniform() < display[1] and display[0] not in self.picked):
			new_tabs.append(display[0])
			self.set_picked(display[0])

		if len(display) == 4:
			if (np.random.uniform() < display[3] * self.lamb and display[2] not in self.picked):
				new_tabs.append(display[2])
				self.set_picked(display[2])

		return new_tabs


	def set_picked(self, product):
		self.picked.append(product)


	# A test function to test the picking based on beta distributed values
	def test(self):
		self.clear()
		self.add_vertex("P1", 60)
		self.add_vertex("P2", 30)
		self.add_vertex("P3", 10)
		self.add_vertex("P4", 5)
		self.add_edge("P1","P2")
		self.add_edge("P1","P3")
		self.add_edge("P1","P4")
		self.add_edge("P2","P1")
		self.add_edge("P2","P3")
		self.add_edge("P2","P4")
		self.add_edge("P3","P1")
		self.add_edge("P3","P2")
		self.add_edge("P3","P4")
		return self.vertices["P1"].pick_display()

	def testMC(self, n):
		primaries = []
		secondaries = []
		for i in range(n):
			print("progress: " + str(i/n))
			result = self.test()
			primaries.append(result[0])
			secondaries.append(result[2])

		print("primaries:")
		print(primaries.count("P2"))
		print(primaries.count("P3"))
		print(primaries.count("P4"))
		print("secondaries:")
		print(secondaries.count("P2"))
		print(secondaries.count("P3"))
		print(secondaries.count("P4"))

	def test_pick(self):
		self.test()
		picks = []
		pick = "P1"
		self.picked.append("P1")
		picks.append("P1")
		while pick != "exit":
			pick = self.pick_new_product(pick)
			picks.append(pick)
		print(picks)

class User:

	def __init__(self):
		self.reservation_price = []
		self.f1 = 0
		self.f2 = 0

	def set_reservation_price(self, prices):
		self.reservation_price = prices

class Simulator:

	def __init__(self):
		self.g = Graph()
		self.alphas = np.array([])
		self.prices = np.array([])

	def simulate_user(self, user):
		roll = np.random.uniform()
		purchases = np.zeros(len(self.prices))

		condition = self.alphas[0]
		if roll < condition:
			return purchases

		for i in range(1,len(self.alphas)):
			condition = condition + self.alphas[i]
			if roll < condition:
				tabs = ["P" + str(i)]
				while tabs != []:				# Sätt in ett stoppvillkor här
					purchases[self.get_index(tabs[0])] = self.check_purchase(user, tabs[0])
					tabs = tabs + self.g.pick_new_product(tabs[0])
					tabs.remove(tabs[0])
				self.g.reset()
				return purchases
		raise Exception("alphas is broken")

	def info(self):
		print("alphas are: " + np.array2string(self.alphas))
		print("The prices are: " + np.array2string(self.prices))

	def set_prices(self, prices):
		self.prices = prices

	def set_alphas(self, alphas):
		self.alphas = alphas

	def run(self, user):
		purchases = self.simulate_user(user)
		return purchases

	def check_purchase(self, user, product):
		index = self.get_index(product)
		if self.prices[index] < user.reservation_price[index]:
			units = np.random.exponential(2)
			units = math.ceil(units)
			return units
		else:
			return 0

	def get_index(self, product):
		return int(product[-1]) - 1

	def setup(self, number_of_products, uncertain=False):
		n_MAX_customers = self.generate_max_customers()
		self.alphas = self.generate_alphas(number_of_products, n_MAX_customers)
		self.prices = self.generate_prices(number_of_products)
		self.g.setup(self.alphas, self.prices, uncertain)


	def generate_user(self):
		user = User()
		user.generate_reservation_price()
		return user

	def generate_prices(self, n):
		prices = np.zeros(n)
		for i in range(n):
			prices[i] = np.random.exponential(10) + 2

		return prices

	# this function could use some work. 20000 is an arbitrary number.
	def generate_max_customers(self):
		return np.random.uniform(0,20000)

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






