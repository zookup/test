# 필요한 라이브러리를 로드합니다
import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

np.random.seed(144)


# parameter
learning_rate = 0.001
training_epochs = 10
batch_Size = 256


# data
input_size = 28
input_steps = 28
n_hidden = 128
n_classes = 10


# declar placeholder and variable

X = tf.placeholder(tf.float32,[None, input_steps, input_size])
y = tf.placeholder(tf.float32,[None, n_classes])

W = tf.Variable(tf.random.normal([n_hidden*2, n_classes]))
b = tf.Variable(tf.random.normal([n_classes]))

keep_prob = tf.placeholder(tf.float32)


lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden, state_is_tuple= True)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple= True)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype = tf.float32)


# 나온 결과 같을 [batchsize, n_step, n_hidden] -> [n_steps, batch_size, n_hidden]
outputs_fw = tf.transpose(outputs[0], [1,0,2])
outputs_bw = tf.transpose(outputs[1], [1,0,2])

output_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)

pred = tf.matmul(output_concat,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.arg_max(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

global_step = 0

start_time = time.time()

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_Size)

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_Size)
        batch_x = batch_x.reshape((batch_Size, input_steps, input_size )).astype(np.float32)

        c, _ = sess.run([cost, optimizer], feed_dict = {X:batch_x, y:batch_y, keep_prob:0.9})

        avg_cost += c / total_batch

        global_step += 1

    test_data = mnist.test.images.reshape((-1, input_steps, input_size))
    test_label = mnist.test.labels

    print('Epoch:{:2d}, cost={:9f}'.format((epoch+1), avg_cost))
    print('Accuracy:', accuracy.eval(session = sess, feed_dict={X:test_data, y:test_label, keep_prob : 1.0}))

end_time = time.time()

print("execution time :", (end_time - start_time))
