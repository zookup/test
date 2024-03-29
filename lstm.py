import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot = True)

tf.set_random_seed(777)

# model

input_size = 28
input_steps = 28
hidden_size = 64
n_classes = 10

learning_rate = 0.01
training_epochs = 10
batch_size = 128
display_step = 10


 # input output placeholder
X = tf.placeholder(tf.float32, [None, input_steps, input_size])
Y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([hidden_size, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))
#############################################################################
# 아래 함수로 대체
# # lstm cell
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 1.0)
#
# # unstack해야 한다.
# x = tf.unstack(X, input_steps, axis = 1)
#
#
# # TypeError : inputs must be a sequence => unstack을 하지 않으면 발생
# # output의 shape : batch_size * hidden_size
# outputs1, states1 = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
# pred = tf.matmul(outputs1[-1], W) + b
##############################################################################

def static_rnn(X,w,b):
    x = tf.unstack(X, input_steps, axis = 1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 1.0)
    outputs1, states1 = tf.nn.static_rnn(lstm_cell, x, dtype = tf.float32)
    return tf.matmul(outputs1[-1], W) + b


pred = static_rnn(X, W, b)

# logits를 softmax한 예측값과 실제값의 차이를 cross_entropy를 이용하여 차이를 비교
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))

# AdamOptimizer를 이용하여 cost를 최소화
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


# Evaluation model
# 예측값과 실제값이 다르면 Flase, 같으면 True
corrected_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

# 맞춘 확률
accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))


# training & prediction
sess = tf.Session()
sess.run(tf.global_variables_initializer())

global_step = 0

start_time = time.time()

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, input_steps, input_size))

        c = sess.run(cost, feed_dict={X:batch_x, Y:batch_y})
        _ = sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
        # TypeError : unhashable type : 'list' => 위의 unstack 된 값이 들어가지 않기 때문

        avg_cost += c/total_batch

        global_step += 1

    test_data = mnist.test.images.reshape((-1, input_steps, input_size))
    test_label = mnist.test.labels

    print('Epoch:{:2d}, cost={:9f}'.format((epoch+1), avg_cost))
    print('Accuracy:', accuracy.eval(session = sess, feed_dict={X:test_data, Y:test_label}))

end_time = time.time()

print("execution time :", (end_time - start_time))
