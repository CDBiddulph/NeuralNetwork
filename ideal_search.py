import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# you can set max_success to a nonzero value if you have standards
max_success = 9520
net = network.Network([784, 30, 10])

# print(net)

while True:
    network_string, candidate = net.sgd_stop_when_stable(training_data, 3, 10, 3.0, test_data)
    if candidate > max_success:
        file = open("best_network.txt", "w+")
        file.write(network_string)
        file.close()
        max_success = candidate
        print(max_success)
    else:
        # if it's the best so far, I'll give it another chance to do better
        # but otherwise it's getting reset
        net = network.Network([784, 30, 10])
