import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

############################# PREPARE DATA #############################

# Generate random houses
num_house = 200
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# plot data
# plt.plot(house_size,house_price, "bo")  # bx = blue x
# plt.ylabel("Price")
# plt.xlabel("Size")

# plt.show()

# data normalization to avoid under/overflow
def normalize(array):
    return (array - array.mean()) / array.std()

num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)  # Why???
train_price_norm = normalize(train_price)  #Why???

# define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)  # Why???
test_price_norm = normalize(test_price)  #Why???

# Set tensorflow placeholders
tf_house_size = tf.placeholder("float", name= "house_size")
tf_house_price = tf.placeholder("float", name="house_price")

# Set tensorflow Variables
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# define operations
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# define the Loss Function
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price, 2))/(2*num_train_samples)

# define optimizer learning rate
learning_rate = 0.1

# define a Gradient descent optimizer to minimize cost
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


# Inizialize tensor variables
init =  tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):

        for(x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size: x, tf_house_price: y})

            # Display current status
        if(iteration+1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration +1), "cost=", "{:.9f}".format(c),\
                  "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_price_norm})
    print("Trained cost =", training_cost, "size_factor =", sess.run(tf_size_factor), "price_offset =", sess.run(tf_price_offset), '\n')


    # Plot
    plt.rcParams["figure.figsize"]=(10, 8)
    plt.figure()
    plt.xlabel("Size")
    plt.ylabel("Price")

    plt.plot(train_house_size, train_price, "go", label="Training data")
    plt.plot(test_house_size, test_price, "mo", label="Testing data")
   # plt.plot(train_house_size_norm * train_house_size_std + )
    plt.legend(loc='upper left')
    plt.show()
