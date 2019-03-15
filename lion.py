from __future__ import division
import numpy as np
from PIL import Image
from scipy import misc
from skimage import data
from skimage.color import rgb2gray
from io import BytesIO
import matplotlib.pyplot as plt
import requests
from urllib3.connectionpool import xrange

response = requests.get('http://vignette2.wikia.nocookie.net/grayscale/images/4/47/Lion.png/revision/latest?cb=20130926182831')
lion_arr = np.array(Image.open(BytesIO(response.content)))

# if all the color channels are actually the same then get only one
if np.array_equal(lion_arr[:, :, 0], lion_arr[:, :, 1]) and np.array_equal(lion_arr[:, :, 1], lion_arr[:, :, 2]):
    lion_arr = lion_arr[:, :, 0]
# print(lion_arr[:10, :10])

# We need to apply our kernel to patches of the image with the same shape as the kernel.
# We’ll multiply our kernel/filter and the patch and then take the sum of the resulting output array.
padded_array = np.pad(lion_arr, (1, 1), 'constant')  # array with one line zeros all around

# print(padded_array[293:, 487:])

# https://it.wikipedia.org/wiki/Matrice_di_convoluzione
kernel = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])  # 3x3 array

# output_array = np.zeros(lion_arr.shape)  # array of lion_arr dim filled by zeros
# # For any given patch in the image, our convolution is just outputting 1 * the middle element of the patch.
# # Every other element-to-element multiplication becomes 0 due to the kernel. For this reason, we call this
# # kernel the identity kernel.
# for i in xrange(padded_array.shape[0]-2):
#     for j in xrange(padded_array.shape[1]-2):
#         temp_array = padded_array[i:i+3, j:j+3]
#         # plt.imshow(temp_array, cmap=plt.get_cmap('gray'))
#         # plt.show()
#         output_array[i, j] = np.sum(temp_array*kernel)


# Since we’re calculating sums, our output values can be greater than 255 or less than 0 so we need to squash them.
def squash_pixel_value(value):
    if value < 0:
        return 0
    elif value < 255:
        return value
    else:
        return 255


def conv_2d_kernel(image_array_2d, kernel, squash_pixels=True):
    padded_array = np.pad(image_array_2d, (1, 1), 'constant')

    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]

    transformed_array = np.zeros(image_array_2d.shape)

    for i in xrange(padded_array.shape[0] - kernel_width + 1):
        for j in xrange(padded_array.shape[1] - kernel_height + 1):
            temp_array = padded_array[i:i + kernel_width, j:j + kernel_height]
            # print temp_array.shape
            if squash_pixels:
                transformed_array[i, j] = squash_pixel_value(np.sum(temp_array * kernel))
            else:
                transformed_array[i, j] = np.sum(temp_array * kernel)
    return transformed_array


# Edge Detection

edge_kernel_1 = np.array([[1, 0, -1],
                          [0, 0, 0],
                          [-1, 0, 1]])

edge_kernel_2 = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

edge_kernel_3 = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])

lion_transf_edge1 = conv_2d_kernel(lion_arr, kernel = edge_kernel_1, squash_pixels = True)
lion_transf_edge2 = conv_2d_kernel(lion_arr, kernel = edge_kernel_2, squash_pixels = True)
lion_transf_edge3 = conv_2d_kernel(lion_arr, kernel = edge_kernel_3, squash_pixels = True)


f, ax_array = plt.subplots(2, 2)
f.set_figheight(10)
f.set_figwidth(15)
ax_array[0, 0].imshow(lion_arr, cmap = plt.get_cmap('gray'))
ax_array[0, 0].set_title('Original Image')
ax_array[0, 0].axis('off')
ax_array[0, 1].imshow(lion_transf_edge1, cmap = plt.get_cmap('gray'))
ax_array[0, 1].set_title('Edge Kernel 1')
ax_array[0, 1].axis('off')
ax_array[1, 0].imshow(lion_transf_edge2, cmap = plt.get_cmap('gray'))
ax_array[1, 0].set_title('Edge Kernel 2')
ax_array[1, 0].axis('off')
ax_array[1, 1].imshow(lion_transf_edge3, cmap = plt.get_cmap('gray'))
ax_array[1, 1].set_title('Edge Kernel 3')
ax_array[1, 1].axis('off')
plt.show()
