from random import uniform


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [uniform(0, 10) for _ in range(n_inputs + 1)]

    def activate(self, inputs):
        potential = self.weights[-1]

        for n_input, input_value in enumerate(inputs):
            potential += input_value * self.weights[n_input]

        return potential

    def transfer(self, potential):
        if potential > 0:
            return 1
        else:
            return 0


