import tensorflow as tf
import matplotlib.pyplot as plt
import math

tf.reset_default_graph()

x_data = tf.placeholder(dtype=tf.float32, shape=None)
y_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(.5, dtype=tf.float32)
intercept = tf.Variable(.5, dtype=tf.float32)
exponent = tf.Variable(.5, dtype=tf.float32)

model_operation = slope * tf.pow(x_data, exponent) + intercept

error = model_operation - y_data
sqr = tf.square(error)
loss = tf.reduce_mean(sqr)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#y = x^2 + 3
def box(x):
    y = math.pow(x, 2) + 3
    return y

x_values = []
y_values = []


for i in range(0, 5):
    x_values.append(i)
    y_values.append(box(i))

print(x_values)
print(y_values)

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={x_data: x_values, y_data: y_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept, exponent]))
            plt.plot(x_values, sess.run(model_operation, feed_dict={x_data: x_values}))

    print(sess.run(loss, feed_dict={x_data: x_values, y_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(model_operation, feed_dict={x_data: x_values}))

    plt.show()