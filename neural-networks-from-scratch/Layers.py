"""
Created by Sanjay at 11/17/2018

Feature: Enter feature name here
Enter feature description here
"""
import numpy as np


class FullyConnectedLayer:
    def __init__(self, lr=1e-3, W=[], b=[]):
        # Initial Weights and Biases
        self.W = W
        self.b = b

        # Gradients
        self.gW = None  # TODO Complete
        self.gb = None  # TODO Complete
        self.gI = None  # TODO Complete

        # Input and Output data
        self.input = None  # TODO Complete
        self.output = None  # TODO Complete

        # Learning Rate
        self.lr = lr

    def forward(self, input):
        self.input = input
        self.output = np.dot (self.W, self.input) + self.b
        return self.output

    def backward(self, gradients):
        self.gW = None  # TODO Complete
        self.gW = None  # TODO Complete
        self.gW = None  # TODO Complete

        # Update Weights and Biases
        self.W = self.W - self.lr * self.gW
        self.b = self.b - self.lr * self.gb
        return self.gI


class ReLULayer:
    def __init__(self):
        self.input = None  # TODO Complete
        self.output = None  # TODO Complete

    def forward(self, input):
        self.input = input
        self.output = np.clip(input, 0, input)
        return self.output

    def backward(self, gradients):
        return gradients * (self.input >= 0)  # If input < 0 gradient = 0


class SoftMaxCrossEntropyLossLayer():
    def __init__(self):
        self.output = None
        self.true_output = None
        self.gI = None

        # For Softmax with Cross Entropy, we calculate delta of loss straightaway without
        # calculating loss and its derivative individually
        self.d_loss = Nonex`

    def calculate(self, input):
        self.input = input
        self.d_loss = np.subtract(self.true_output, self.output)
        return self.d_loss