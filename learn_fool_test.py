import network
import mnist_loader
import net_helper as nh
import numpy as np

target_digit = 8

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

file = open("learn_fool_network.txt", "r")
net = network.from_string(file.read())
file.close()
net.cost_influences = np.array([20, 0, 1])

original_data = test_data[0][0]

empty_output_half = np.full_like(training_data[0][1], 0)

morphed_data = net.morph_data(original_data, np.append(nh.output_vector(target_digit), empty_output_half, axis=0),
                              100, .001)

if morphed_data is not None:
    print("Final evaluation:", np.argmax(net.feed_forward(morphed_data)))

    difference_data = np.clip(morphed_data - original_data + np.full(morphed_data.shape, .5), 0, 1)
    combined = np.append(np.append(original_data, difference_data, axis=0), morphed_data, axis=0)

    nh.vector_to_png(combined, 28, 10).save('combined_image.png')

    # nh.vector_to_png(difference_data, 28, 10).save('difference_image.png')
    # nh.vector_to_png(morphed_data, 28, 10).save('morphed_image.png')
    # nh.vector_to_png(original_data, 28, 10).save('original_image.png')
else:
    print("Unsuccessful")

