"""
Created by Sanjay at 10/8/2021

Feature: Enter feature name here
Enter feature description here
"""
import numpy as np


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.output = 0
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
