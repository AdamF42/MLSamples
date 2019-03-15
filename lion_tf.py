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
import tensorflow as tf


response = requests.get('http://vignette2.wikia.nocookie.net/grayscale/images/4/47/Lion.png/revision/latest?cb=20130926182831')
lion_arr = np.array(Image.open(BytesIO(response.content)))

# if all the color channels are actually the same then get only one
if np.array_equal(lion_arr[:, :, 0], lion_arr[:, :, 1]) and np.array_equal(lion_arr[:, :, 1], lion_arr[:, :, 2]):
    lion_arr = lion_arr[:, :, 0]

blur_box_kernel = np.ones((3, 3)) / 9
blur_gaussian_kernel = np.array([[1,2,1],
                                 [2,4,2],
                                 [1,2,1]]) / 16

lion_array_4d = lion_arr.reshape(-1, 303, 497, 1)
blur_kernel_4d = blur_box_kernel.reshape(3, 3, 1, 1)
print(blur_kernel_4d.shape)

graph = tf.Graph()
with graph.as_default():
    # A variable maintains state in the graph across calls
    tf_input_image = tf.Variable(np.array(lion_array_4d, dtype = np.float32))
    print(tf_input_image)
    tf_blur_kernel = tf.Variable(np.array(blur_kernel_4d, dtype = np.float32))
    # strides = [1, 1, 1, 1] results in a convolution on every pixel
    # padding = 'SAME' is standard zero padding that results in an output array with the same shape as the input array.
    tf_convolution_output = tf.nn.conv2d(tf_input_image, tf_blur_kernel, strides = [1, 1, 1, 1], padding = 'SAME')
    tf_max_pooling = tf.nn.max_pool(tf_input_image, ksize=[1, 2, 2, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    print(tf_convolution_output.shape)
    print(tf_max_pooling.shape)

with tf.Session(graph = graph) as sess:
    tf.global_variables_initializer().run()
    transformed_image = tf_max_pooling.eval()
    transformed_image = transformed_image[0, :, :, 0]

f, ax_array = plt.subplots(2, 1)
f.set_figheight(15)
f.set_figwidth(12)
ax_array[0].imshow(lion_arr, cmap = plt.get_cmap('gray'))
ax_array[0].axis('off')
ax_array[1].imshow(transformed_image, cmap = plt.get_cmap('gray'))
ax_array[1].axis('off')

plt.show()
