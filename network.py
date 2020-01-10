import random
import numpy as np
import net_helper as nh
import copy


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # a list of np arrays of one-float lists
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # a list of np arrays of multiple-float lists
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost_influences = [1] * self.num_layers

    # returns only output activations
    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # returns activations of all layers (including input)
    def feed_forward_complete(self, a):
        out = [a]
        for w, b in zip(self.weights, self.biases):
            out.append(sigmoid(np.dot(w, out[-1]) + b))
        return out

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None, to_save_failures=False):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
        if to_save_failures:
            self.save_failures(test_data, 'failures', 'failure')

    def learn_fool_sgd(self, training_data, epochs, mini_batch_size, eta, morph_iterations, morph_eta):
        n = len(training_data)
        empty_output_half = np.full_like(training_data[0][1], 0)
        doubled_td = copy.deepcopy(training_data)
        for i in range(len(doubled_td)):
            doubled_td[i] = (doubled_td[i][0], np.append(doubled_td[i][1], empty_output_half, axis=0))
        for j in range(epochs):
            random.shuffle(doubled_td)
            mini_batches = [
                doubled_td[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                morphed_data = self.random_morphed(training_data, mini_batch_size * 2, empty_output_half,
                                                   morph_iterations, morph_eta)
                combined_data = np.append(mini_batch, morphed_data, axis=0)
                random.shuffle(combined_data)
                self.update_mini_batch(combined_data, eta)

                file = open("learn_fool_network.txt", "w+")
                file.write(self.__str__())
                file.close()
            print("Epoch {0} complete".format(j))

    def random_morphed(self, data, num_needed, empty_output_half, iterations, eta):
        out = []
        i = 0
        while len(out) < num_needed:
            if i == 0:
                random.shuffle(data)
            target_digit = random.randint(0, 8)
            if target_digit >= np.argmax(data[i][1]):
                target_digit += 1
            doubled_target = np.append(nh.output_vector(target_digit), empty_output_half, axis=0)

            # actual_output = np.argmax(self.feed_forward(self.morph_data(data[i][0], doubled_target, iterations, eta)))
            # if actual_output > 9:
            #     print(np.argmax(data[i][1]), "->", target_digit, actual_output)

            temp = self.morph_data(data[i][0], doubled_target,
                                   iterations, eta, stop_when_fooled=True)
            if temp is not None:
                out.append((temp, np.append(empty_output_half, data[i][1], axis=0)))
                nh.vector_to_png(out[-1][0], 28, 10).save('morphed_image.png')
            i = (i + 1) % len(data)
        return out

    def sgd_stop_when_stable(self, training_data, epoch_memory, mini_batch_size, eta, test_data):
        success_rates = []
        n = len(training_data)
        while True:
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            num_correct = self.evaluate(test_data)
            if len(success_rates) >= epoch_memory:
                if success_rates.pop() >= num_correct:
                    return self.__str__(), num_correct
            success_rates.insert(0, num_correct)

    def morph_data(self, original, target_out, iterations, eta, stop_when_fooled=False):
        target = self.feed_forward_complete(original)
        target[-1] = target_out
        morphed = copy.deepcopy(original)

        disregard_output = False

        for j in range(iterations):
            neuron_influences = self.cost_influences / self.sizes / np.sum(self.cost_influences)
            if disregard_output:
                neuron_influences = neuron_influences * 300
                neuron_influences[-1] = 0
            nabla_m = self.morph_prop(morphed, target)
            # there's probably a more efficient way to do this
            for l in range(len(neuron_influences)):
                nabla_m[l] = nabla_m[l] * neuron_influences[l]
            nabla_m = np.sum(nabla_m, axis=0)
            morphed = np.clip(morphed - nabla_m * eta, 0, 1)
            result = np.argmax(self.feed_forward(morphed))
            if result == np.argmax(target_out):
                if stop_when_fooled:
                    # print("Stopped on", j)
                    return morphed
                else:
                    disregard_output = True
            else:
                disregard_output = False
            # print('Iter{0}: {1}'.format(j, result))
            # print('   Cost: {0}'.format(self.morph_cost(morphed, target)))
        if stop_when_fooled:
            return None
        return morphed

    def morph_prop(self, x, target):
        # for each item in the list nabla_m, there is an array of values,
        # each corresponding to an input layer activation
        # each value represents the rate at which that activation affects the cost of that layer
        nabla_m = [np.full((len(x), 1), 1)] * self.num_layers

        # feed forward
        activation = x
        activations = [x]
        zs = [[]]  # extra array to offset index and make it easier to work with
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # forward pass
        delta = np.identity(len(x))
        nabla_m[0] = np.array(activations[0] - target[0])

        for l in range(1, self.num_layers):
            z = zs[l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, self.weights[l - 1].transpose())
            for n in range(len(sp)):
                delta[n] = delta[n] * sp[n][0]
            nabla_m[l] = np.dot(delta, activations[l] - target[l])

        # print(np.shape(nabla_m))
        return nabla_m

    # this whole function is pretty inefficient, but I don't know the numpy stuff well enough
    def morph_cost(self, actual_output, target):
        actual = self.feed_forward_complete(actual_output)
        target = target
        ds = np.subtract(actual, target)
        neuron_influences = (self.cost_influences / self.sizes) / np.sum(self.cost_influences)
        out = 0
        for l in range(len(ds)):
            for i in range(len(ds[l])):
                d = ds[l][i]
                out += d * d * neuron_influences[l]
        return out[0]

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # note that the result is being subtracted from, not added to, the biases/weights
        self.biases = [b - (nb * eta / len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (nw * eta / len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # delta ends up with the last dimension in weights,
            # then the last dimension in delta (which is 1).
            # the first in weights and delta must match
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save_failures(self, test_data, folder_name, file_prefix):
        fails = 0
        for (x, y) in test_data:
            if np.argmax(self.feed_forward(x)) != y:
                nh.vector_to_png(x, 28).save('{0}/{1}{2}.png'.format(folder_name, file_prefix, fails))
                fails = fails + 1

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def __str__(self):
        out = str(self.num_layers) + "\n"
        for size in self.sizes:
            out = out + str(size) + " "
        out = out + "\n"
        for layer in self.biases:
            for bias in layer:
                out = out + str(bias[0]) + " "
            out = out + "\n"
        for layer in self.weights:
            for neuron in layer:
                for weight in neuron:
                    out = out + str(weight) + " "
                out = out + "\n"
        return out


def from_string(s):
    out = Network([])
    lines = s.splitlines()
    out.num_layers = int(lines[0])
    out.cost_influences = np.full(out.num_layers, 1)
    # "if s" ensures that s is not empty (as in end of line)
    out.sizes = [int(s) for s in lines[1].split(' ') if s]
    out.biases = [np.array([[float(s)] for s in l.split(' ') if s]) for l in lines[2: out.num_layers + 1]]
    layer_start = out.num_layers + 1
    for size in out.sizes[1:]:
        out.weights.append(np.array([[float(s) for s in layer.split(' ') if s]
                                     for layer in lines[layer_start: layer_start + size]]))
        layer_start = layer_start + size
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
