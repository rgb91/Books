from NeuralNetwork import NeuralNetwork
from Layers import FullyConnectedLayer, SoftMaxCrossEntropyLossLayer, ReLULayer

if __name__ == '__main__':
    model = NeuralNetwork (input_size=14)
    model.add(FullyConnectedLayer(100))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(40))
    model.add (ReLULayer ())
    model.add(FullyConnectedLayer(4))  # Fully Connected
    model.calculate_output(SoftMaxCrossEntropyLossLayer())

    history = model.fit()
    predicted_values = model.predict()

