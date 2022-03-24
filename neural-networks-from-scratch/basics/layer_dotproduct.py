"""
    Creates a simple layer of neurons, with 4 inputs.
    Associated YT NNFS tutorial: https://www.youtube.com/watch?v=lGLto9Xd7bU
"""
import numpy as np

inputs_0 = [1.0, 2.0, 3.0, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

"""
    Part 2: Multiplication of individual weights and 
"""
out_0 = [inputs_0[0] * weights1[0] + inputs_0[1] * weights1[1] + inputs_0[2] * weights1[2] + inputs_0[3] * weights1[3] + bias1,
         inputs_0[0] * weights2[0] + inputs_0[1] * weights2[1] + inputs_0[2] * weights2[2] + inputs_0[3] * weights2[3] + bias2,
         inputs_0[0] * weights3[0] + inputs_0[1] * weights3[1] + inputs_0[2] * weights3[2] + inputs_0[3] * weights3[3] + bias3]
print(out_0)

"""
    Part 3: Do the same thing as previous section with numpy dot function
            First use of matrix
"""
weights = [weights1, weights2, weights3]
biases = [bias1, bias2, bias3]

out_1 = np.dot(weights, inputs_0) + biases
print(out_1)

"""
    Part 4.1: Batches
"""
inputs_0 = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]
out_2 = np.dot(inputs_0, np.array(weights).T) + biases
print(out_2, '\n')

"""
    Part 4.2: Multiple Layers
"""
weights_l1 = weights  # layer 1 weights
weights_l2 = [[0.1, -0.14, 0.5],  # layer 2 weights
              [-0.5, 0.12, -0.33],
              [-0.44, 0.73, -0.13]]
biases_l1 = biases
biases_l2 = [-1, 2, -0.5]

l1_outputs = np.dot(inputs_0, np.array(weights_l1).T) + biases_l1
l2_outputs = np.dot(l1_outputs, np.array(weights_l2).T) + biases_l2

print(f'{l2_outputs = }', '\n')

