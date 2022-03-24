"""
Created by Sanjay at 10/8/2021

Feature: Enter feature name here
Enter feature description here
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from basics.Layers import LayerDense

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print(X.shape, y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

dense1 = LayerDense(n_inputs=2, n_neurons=3)
dense1.forward(X)

# print(dense1.output[:5])
print(dense1.output.shape)