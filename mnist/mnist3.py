# Xavier initialization

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(n_inputs, n_outputs, uniform=True):
	if uniform:
		init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)

# data set loading
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 784], name="input")
Y = tf.placeholder("float", [None, 10], name="label")

# Store layers weight & bias
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 10], initializer=xavier_init(256, 10))

# W1 = tf.Variable(tf.random_normal([784, 256]))
# W2 = tf.Variable(tf.random_normal([256, 256]))
# W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]), name="B1")
B2 = tf.Variable(tf.random_normal([256]), name="B2")
B3 = tf.Variable(tf.random_normal([10]), name="B3")

# Construct model
with tf.name_scope("model"):
	with tf.name_scope("input_layer"):
		L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
	with tf.name_scope("hidden_layer"):
		L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
	with tf.name_scope("output_layer"):
		hypothesis = tf.add(tf.matmul(L2, W3), B3)

# Define loss and optimizer
with tf.name_scope("cost_function"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
with tf.name_scope("optimizer"):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

tf.summary.scalar('cost', cost)

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	#tensorboard summary write
	train_writer = tf.summary.FileWriter('./summaries/mnist3/', sess.graph)
	summary_op = tf.summary.merge_all()
	summary_step = 0

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Fit training using batch data
			sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
			# Compute average loss
			avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch
			# summary write
			if summary_step % 100 == 0:
				result = sess.run(summary_op, feed_dict={X:mnist.train.images, Y:mnist.train.labels})
				summary_str = result
				train_writer.add_summary(summary_str, summary_step)

			summary_step += 1

		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

	print "Optimization Finished!"

	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Train Accuracy:", accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})
	print "Test Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
