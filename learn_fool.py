import network
import mnist_loader
import net_helper as nh
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 20])

file = open("learn_fool_network.txt", "r")
net = network.from_string(file.read())
file.close()
net.cost_influences = np.array([0, 0, 1])

net.learn_fool_sgd(training_data, 1, 10, 3.0, 100, 0.02)

original_data = test_data[1][0]

morphed_data = net.morph_data(original_data, nh.output_vector(5), 500, .02, stop_when_fooled=True)

print("Final evaluation:", np.argmax(net.feed_forward(morphed_data)))

file = open("learn_fool_network.txt", "w+")
file.write(net.__str__())
file.close()

difference_data = np.clip(morphed_data - original_data + np.full(morphed_data.shape, .5), 0, 1)

nh.vector_to_png(original_data, 28, 10).save('original_image.png')
nh.vector_to_png(morphed_data, 28, 10).save('morphed_image.png')
nh.vector_to_png(difference_data, 28, 10).save('difference_image.png')
