from __future__ import division
import numpy as np
from PIL import Image
from scipy import misc
from skimage import data
from skimage.color import rgb2gray
from io import BytesIO
import matplotlib.pyplot as plt
import requests


response = requests.get('http://vignette2.wikia.nocookie.net/grayscale/images/4/47/Lion.png/revision/latest?cb=20130926182831')
lion_arr = np.array(Image.open(BytesIO(response.content)))

plt.imshow(lion_arr)

plt.show()


print (lion_arr.shape)
print (np.array_equal(lion_arr[:, :, 0], lion_arr[:, :, 1]))
print (np.array_equal(lion_arr[:, :, 1], lion_arr[:, :, 2]))

lion_arr = lion_arr[:, :, 0]


print(lion_arr[:10, :10])
