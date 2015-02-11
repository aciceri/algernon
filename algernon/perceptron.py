from algernon.neuron import Neuron


class Perceptron:
    def __init__(self, n_inputs):
        self.neuron = Neuron(n_inputs)

    def train(self, in_set, des_out_set, max_epoch=500, des_error=0.1, learning_rate=0.1):
        for epoch in range(max_epoch):
            error = 0
            error_weights = [0] * len(self.neuron.weights)

            for x_set, des_out in zip(in_set, des_out_set):
                y = self.neuron.transfer(self.neuron.activate(x_set))
                y_error = -(y - des_out)

                for i, x in enumerate(x_set):
                    error_weights[i] += (x * y_error)

                error_weights[i + 1] += (y_error)

                error += (y_error ** 2) / 2.0

            for i, error_weight in enumerate(error_weights):
                self.neuron.weights[i] += (error_weight * learning_rate)

            self.log(epoch, error, error_weights, self.neuron.weights)

            if error <= des_error:
                break

    def go(self, inputs):
        return self.neuron.transfer(self.neuron.activate(inputs))

    def log(self, epoch, error, error_weights, weights):
        print("EPOCH #%d" % epoch)
        print("  Error: %f" % error)
        print("  Error weights: (last is bias)")
        for e in error_weights:
            print("    %f" % e)
        print("  Weights: (last is bias)")
        for e in weights:
            print("    %f" % e)

