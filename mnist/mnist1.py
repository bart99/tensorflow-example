# Softmax classfier

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# data set loading
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
training_epochs = 15
batch_size = 100
display_step = 1

# set up model
X = tf.placeholder(tf.float32, [None, 784], name="input")
Y = tf.placeholder(tf.float32, [None, 10], name="label")

W = tf.Variable(tf.zeros([784, 10]), name="weight")
b = tf.Variable(tf.zeros([10]), name="bias")


with tf.name_scope("model"):
	hypothesis = tf.matmul(X, W) + b

# cost funcition
with tf.name_scope("cost_function"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))

tf.summary.scalar('cost', cost)

# optimization algorithm
with tf.name_scope("optimizer"):
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# tensorboard summary write
	train_writer = tf.summary.FileWriter('./summaries/mnist1/', sess.graph)

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
			result = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})
			avg_cost += result / total_batch

			# summary write
			if summary_step % 100 == 0 :
				result = sess.run(summary_op, feed_dict={X: mnist.train.images, Y: mnist.train.labels})
				summary_str = result
				train_writer.add_summary(summary_str, summary_step)

			summary_step += 1

		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)



	print "Optimization Finished!"


	# Test model
	correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Train Accuracy:", accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})
	print "Test Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
