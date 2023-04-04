import random

from color import Color
from functions import *


class Network:
    def __init__(
        self, sizes, activation_function=sigmoid, derivative_function=sigmoid_derivative
    ):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.saved_activations = []
        self.saved_derivatives = []

    def dataset_testing(self, test_name, dataset):
        iteration = 1
        color = Color.BOLD
        print(color, test_name, "\n")
        correct_count = 0
        for data in dataset:
            if data[1] == np.argmax(self.forward_pass(data[0])[0]):
                color = Color.OKGREEN
                result = "✅"
                correct_count += 1
            else:
                color = Color.FAIL
                result = "❌"
            print(color, "#", iteration, "\t", result)
            iteration += 1
        color = Color.ENDC
        print(color, "\n Total:\t", correct_count, " / ", len(dataset), "\n\n")

    def sgd(self, dataset, batch_size, epochs_count, learning_rate):
        for epoch in range(epochs_count):
            random.shuffle(dataset)
            batches = [
                dataset[k:k + batch_size] for k in range(0, len(dataset), batch_size)
            ]
            for batch in batches:
                for sample in batch:
                    self.back_propagation(
                        input_values=sample[0],
                        correct_output=sample[1],
                        learning_rate=learning_rate,
                    )

    def forward_pass(self, input_values):
        self.saved_activations.clear()
        self.saved_derivatives.clear()
        activation = input_values
        self.saved_activations.append(activation)
        self.saved_derivatives.append(activation)
        input_sums = []
        for layer in range(len(self.sizes[1:])):
            input_sums = self.weights[layer] @ np.array(activation) + self.biases[layer].T[0]
            activation = [self.activation_function(input_sum) for input_sum in input_sums]
            derivative = [self.derivative_function(input_sum) for input_sum in input_sums]
            for i in range(len(derivative)):
                derivative[i] = [derivative[i]]
            self.saved_activations.append(activation)
            self.saved_derivatives.append(derivative)
        return activation, input_sums

    def back_propagation(self, input_values, correct_output, learning_rate):
        sum_gradient = np.array(
            self.get_error(input_values, correct_output)
        ).T

        for layer in range(len(self.sizes) - 1, 0, -1):
            weight_gradient = (
                np.array(self.saved_activations[layer - 1]).reshape(
                    len(self.saved_activations[layer - 1]), 1
                )
                @ np.array(sum_gradient).reshape(1, len(sum_gradient))
            ).T
            bias_gradient = np.array(sum_gradient).T

            activations_gradient = sum_gradient.T @ self.weights[layer - 1]
            sum_gradient = (
                activations_gradient * np.array(self.saved_derivatives[layer - 1]).T
            )[0]

            self.weights[layer - 1] = (
                self.weights[layer - 1] - learning_rate * weight_gradient
            )
            self.biases[layer - 1] = (
                self.biases[layer - 1] - learning_rate * bias_gradient
            )

        error = np.array(self.get_error(input_values, correct_output)).T
        return error

    def get_error(self, input_values, correct_output):
        predict = self.forward_pass(input_values)[0]
        return cross_entropy(y=correct_output, p=softmax(predict))
