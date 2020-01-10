import network
import mnist_loader
import net_helper as nh
import numpy as np
import random

#

#

#

#

# CHANGE THE BLUE NUMBERS BELOW AND PRESS SHIFT+F10
# TO GENERATE YOUR OWN MODIFIED IMAGE OF A DIGIT
# THEN OPEN "combined_image" IN THE FILE EXPLORER

# this is the digit that the original image will be of
original_digit = 8
# this is what the computer will think the image is of
target_digit = 3

#

#

#

#

#

# DON'T CHANGE ANYTHING BELOW THIS LINE

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

file = open("best_network.txt", "r")
net = network.from_string(file.read())
file.close()
net.cost_influences = np.array([10, 10, 1])

random.shuffle(test_data)

for data in test_data:
    if data[1] == original_digit:
        morph = net.morph_data(data[0], nh.output_vector(target_digit),
                               500, .01, stop_when_fooled=True)
        if morph is not None:
            difference = np.clip(morph - data[0] + np.full(morph.shape, .5), 0, 1)
            combined = np.append(np.append(data[0], difference, axis=0), morph, axis=0)
            nh.vector_to_png(combined, 28, 10).save('showcase\\combined_image.png')
            break

print("Final evaluation:", np.argmax(net.feed_forward(morph)))

"""
for data in test_data:
    if data[1] == original_digit:
        stop = False
        for digit in range(0, 10):
            morph = net.morph_data(data[0], nh.output_vector(digit),
                                   500, .02, stop_when_fooled=True)
            if morph is not None:
                difference = np.clip(morph - data[0] + np.full(morph.shape, .5), 0, 1)
                # combined = np.append(np.append(data[0], difference, axis=0), morph, axis=0)
                nh.vector_to_png(morph, 28, 10).save('all_digits\\to_{0}.png'.format(digit))
            else:
                stop = True
                break
        if not stop:
            nh.vector_to_png(data[0], 28, 10).save('all_digits\\original.png'.format(digit))
            break
"""

"""
original_data = test_data[2][0]

empty_output_half = np.full_like(training_data[0][1], 0)

morphed_data = net.morph_data(original_data, np.append(nh.output_vector(target_digit), empty_output_half, axis=0),
                              500, .01)

# morphed_data = net.morph_data(original_data, nh.output_vector(target_digit),
#                               1000, .001, stop_when_fooled=True)

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
"""
