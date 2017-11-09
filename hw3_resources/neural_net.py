import numpy as np
import sys
import math
import time

class NeuralNet:
	def __init__(self, layers):
		self.layers = layers #layers has 1 input + L hidden + 1 output layer
		L = len(layers) - 2
		self.weight = []
		self.bias = []
		self.activation = []
		self.z = []
		self.gradient = []
		for i in range(L + 1):
			self.weight.append(np.random.randn(layers[i], layers[i+1]) * sqrt(2.0/layers[i]))
			self.gradient.append(np.zeros((layers[i], layers[i+1])))
			self.bias.append(np.zeros(layers[i+1]))
			self.activation.append(np.zeros(layers[i]))
			self.z.append(np.zeros(layers[i+1]))
		self.softMax = np.zeros(layers[-1])

	def feedForward(self, x):
		self.activation[0] = x
		self.activation[0].flatten()
		L = len(self.layers) - 2
		for stage in range(L):
			self.z[stage] = np.dot(self.activation[stage], self.weight[stage]) + self.bias[stage]
			self.activation[stage + 1] = self.z[stage] * (self.z[stage] > 0)
		self.z[L] = np.dot(self.activation[L], self.weight[L]) + self.bias[L]
		self.softMax = np.exp(self.z[L])
		self.softMax = self.softMax / np.sum(self.softMax)

	def backPropagate(self, y):
		L = len(self.layers) - 2
		gradient[L] = (self.softMax - 1) * y
		gradient[L].flatten()
		for stage in reversed(range(L)):
			gradient[stage] = np.dot(self.weight[stage + 1], gradient[stage + 1])