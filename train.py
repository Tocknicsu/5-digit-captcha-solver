import numpy
import tensorflow as tf

import cv2
import input_data

learning_rate = 1e-3
train = True
training_iters = 1000
batch_size = 20
display_step = 1

width = 100
height = 50

n_input = width * height
n_classes = 50  # 5 * 10
dropout = 0.75  # Dropout, probability to keep units


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, sx, sy):
    return tf.nn.conv2d(x, W, strides=[1, sx, sy, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork(_x, _dropout):
    # network weights
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1024, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, n_classes])
    b_fc2 = bias_variable([n_classes])

    # hidden layers
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(_x, W_conv1, 4, 2), b_conv1))
    h_pool1 = max_pool_2x2(h_conv1)
    # h_pool1 = tf.nn.dropout(h_pool1, _dropout)

    h_conv2 = tf.nn.relu(tf.nn.bias_add(
        conv2d(h_pool1, W_conv2, 2, 2), b_conv2))
    # h_conv2 = tf.nn.dropout(h_conv2, _dropout)

    h_conv3 = tf.nn.relu(tf.nn.bias_add(
        conv2d(h_conv2, W_conv3, 2, 2), b_conv3))
    # h_conv3 = tf.nn.dropout(h_conv3, _dropout)

    h_conv4_flat = tf.reshape(h_conv3, [-1, 1024])

    h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_conv4_flat, W_fc1), b_fc1))
    # h_fc1 = tf.nn.dropout(h_fc1, _dropout)

    readout = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2)

    return readout


x = tf.placeholder("float", [None, width, height, 1])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")

pred = createNetwork(x, keep_prob)
pred_digits = tf.split(1, 5, pred)
y_digits = tf.split(1, 5, y)

pred_label = tf.argmax(tf.reshape(pred, [-1, 5, 10]), 2)
y_label = tf.argmax(tf.reshape(y, [-1, 5, 10]), 2)

cost = 0
for i in range(5):
    cost += tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_digits[i], y_digits[i])
    )

print("OK")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(pred_label, y_label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)


def Train(sess):
    curr = 0
    for iters in range(training_iters):
        batch_xs, batch_ys = input_data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_xs,
            y: batch_ys,
            keep_prob: dropout
        })
        current_accuracy, current_cost = sess.run([
            tf.reduce_min(accuracy), tf.reduce_min(cost)
        ], feed_dict={
            x: batch_xs,
            y: batch_ys,
            keep_prob: 1.
        })
        print("Iter: ", iters, current_accuracy, current_cost)
        if current_accuracy >= 0.9:
            curr += 1
        if curr >= 10:
            return


def Test(sess):
    batch_xs, batch_ys = input_data.test.next_batch(len(input_data.test.pool))
    current_accuracy, current_cost = sess.run([
        tf.reduce_min(accuracy), tf.reduce_min(cost)
    ], feed_dict={
        x: batch_xs,
        y: batch_ys,
        keep_prob: 1.
    })
    total_err = 0
    pred_ans_coll = sess.run(pred_label, feed_dict={
        x: batch_xs,
        keep_prob: dropout
    })
    y_ans_coll = sess.run(y_label, feed_dict={
        y: batch_ys,
    })
    for idx in range(len(batch_xs)):
        xs = [batch_xs[idx]]
        ys = [batch_ys[idx]]
        err = 0
        pred_ans = pred_ans_coll[idx]
        y_ans = y_ans_coll[idx]
        for id in range(5):
            err += pred_ans[id] != y_ans[id]
        total_err += err
        print(pred_ans, y_ans, err)
    print(
        current_accuracy,
        current_cost,
        (len(batch_xs) * 5 - total_err) / (len(batch_xs) * 5)
    )

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        if train:
            Train(sess)
            saver.save(sess, 'saved_networks/weight.dat')

        Test(sess)
