import png
import numpy as np


def vector_to_png(arr, width, scale):
    output = []
    col = 0
    row_arr = []
    for x in arr:
        int_x = int(x * 256)
        if int_x > 255:
            int_x = 255
        for _ in range(scale):
            row_arr.append(int_x)
        col = (col + scale) % (width * scale)
        if col == 0:
            for _ in range(scale):
                output.append(row_arr)
            row_arr = []
    return png.from_array(output, 'L')


def output_vector(digit):
    out = np.full((10, 1), 0)
    out[digit] = [1]
    return out
