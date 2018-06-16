''' (Feed forward neural network - straight through)
input -> weight -> hidden layer 1 (activation function) -> weight -> hidden layer2 
(activation function) -> weights -> output layer

compare output to intended output -> cost function (cross entropy)
optimization function (optimizer) -> minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation is adjusted weights assigned by optimization function

feed forward + backprop = epoch  (epoch is a cycle)
'''

import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#10 classes, 0-9 (onehot parameter how it works -- one is on, rest are off)
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]
5 = [0,0,0,0,0,1,0,0,0,0]
6 = [0,0,0,0,0,0,1,0,0,0]
7 = [0,0,0,0,0,0,0,1,0,0]
8 = [0,0,0,0,0,0,0,0,1,0]
9 = [0,0,0,0,0,0,0,0,0,1]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100  #how many images will be processed at the same time and manipulated

#Placeholder variables (height x width)
x = tf.placeholder('float',[None, 784])  #Data
y = tf.placeholder('float',shape = [None, n_classes])              #Label of data

def neural_network_model(data):
	#  -creates bunch of tensors (arrays) that are filled with random nums as your weights
	#  -a bias adds to the neuron to protect it from not being 0 (it will never fire if 0)
	#  -hidden layer 2 weights, first input will not be 784, but rather num nodes
	#   from hidden layer 1 -- similar for hidden layer 3
	#  -output layer variables will be num nodes from hl3 and also number of classes (10)
	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_hl1])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases' : tf.Variable(tf.random_normal([n_classes]))}

	# (input data * weight) + biases -- (model for each layer)
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #relu acts like threshhold function (does it fire or not)
	
	# l1 data going into layer 2, same for l3
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels =  y))
	# minimize cost (difference between output and real output)
	optimizer = tf.train.AdamOptimizer().minimize(cost)  #AdamOptimizer has a param learning_rate , default 0.001
	# cycles feed forward + backprop
	hm_epochs = 20


	#Trains data
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c 
			print('Epoch ' , (epoch+1), ' completed out of ' , hm_epochs, ' loss: ', epoch_loss)

		

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) #argmax return max value in array
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)