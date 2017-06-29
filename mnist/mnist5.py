# Convolutional Neural Net
import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

training_epochs = 15
batch_size = 100
test_size = 256

load_dir = "data"
load_name = os.path.join(load_dir, 'model.ckpt')

TRAIN = 0
TEST = 1

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
	with tf.name_scope("model"):
		with tf.name_scope("input_layer"):
			l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
			l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			l1 = tf.nn.dropout(l1, p_keep_conv)

		with tf.name_scope("hidden_layer1"):
			l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
			l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			l2 = tf.nn.dropout(l2, p_keep_conv)

		with tf.name_scope("hidden_layer2"):
			l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
			l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
			l3 = tf.nn.dropout(l3, p_keep_conv)

		with tf.name_scope("hidden_layer3"):
			l4 = tf.nn.relu(tf.matmul(l3, w4))
			l4 = tf.nn.dropout(l4, p_keep_hidden)

		with tf.name_scope("output_layer"):
			pyx = tf.matmul(l4, w_o)

	return pyx

# argument
mode = TRAIN
if len(sys.argv) > 1 :
	if sys.argv[1] == "test" :
		mode = TEST;

# data set loading
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1], name="input")
Y = tf.placeholder("float", [None, 10], name="output")

w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, 10])

p_keep_conv = tf.placeholder("float", name="keep_conv")
p_keep_hidden = tf.placeholder("float", name="keep_hidden")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost_function"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

with tf.name_scope("optimizer"):
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.cast(tf.argmax(py_x, 1), tf.int32, name="predict_op")

tf.summary.scalar('cost', cost)

# Launch the graph in a session


# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	saver = tf.train.Saver(tf.global_variables())
	ckpt = tf.train.get_checkpoint_state(load_dir)

	if mode == TRAIN:

		train_writer = tf.summary.FileWriter('./summaries/mnist5/', sess.graph)
		summary_op = tf.summary.merge_all()
		summary_step = 0

		for i in range(training_epochs):
			avg_cost = 0.
			training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))

			for start, end in training_batch:
				sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
				avg_cost += sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})/batch_size
				# summary write
#				if summary_step % 100 == 0 :
#					result = sess.run(summary_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
#					summary_str = result
#					train_writer.add_summary(summary_str, summary_step)

				summary_step += 1

			print "Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost)

		print "Optimization Finished!"


		saver.save(sess, load_name)
		print "Model Saved!"

		tf.train.write_graph(sess.graph_def, load_dir, 'trained.pb', as_text=False)
		tf.train.write_graph(sess.graph_def, load_dir, 'trained.txt', as_text=True)
		print "Graph Saved!"

	else :
		print (ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		print "Model Loaded!"

	#	test_indices = np.arange(test_size)
	#	np.random.shuffle(test_indices)
	#	test_indices = test_indices[0:test_size]

#		print "Train Accuracy:", np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
		print "Test Accuracy:", np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_hidden: 1.0}))

