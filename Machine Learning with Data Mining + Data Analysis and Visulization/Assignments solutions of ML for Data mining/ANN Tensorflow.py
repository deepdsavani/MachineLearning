import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)

IMAGE_PIXELS = x_train.shape[1]*x_train.shape[2]*x_train.shape[3];
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]);
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]);

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)

x_train=x_train/255.0;
x_test=x_test/255.0;


y_Train = [];
y_Test = [];

for l in y_train:
	t = np.zeros(10);
	t[l]=1;
	y_Train.append(t);

for l in y_test:
	t = np.zeros(10);
	t[l]=1;
	y_Test.append(t);

y_train = np.array(y_Train, dtype=np.float32);
y_test = np.array(y_Test, dtype=np.float32);

n_nodes_hl1 = 512;
n_nodes_hl2 = 1024;
n_nodes_hl3 = 512;
n_nodes_hl4 = 256;
n_nodes_hl5 = 128;


n_classes = 10;
batch_size = 128;


x = tf.placeholder('float32',shape=(None,IMAGE_PIXELS));
y = tf.placeholder('float32');


def neural_network_model(data) :

	hidden_layer_1 = { 'weights': tf.Variable(tf.random_normal([IMAGE_PIXELS, n_nodes_hl1])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl1])) };

	hidden_layer_2 = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl2])) };

	hidden_layer_3 = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl3])) };

	hidden_layer_4 = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl4])) };

	hidden_layer_5 = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
						'biases' : tf.Variable(tf.random_normal([n_nodes_hl5])) };						

	output_layer = 	{ 'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
						'biases' : tf.Variable(tf.random_normal([n_classes])) };

	l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']) , hidden_layer_1['biases'] );
	l1 = tf.nn.tanh(l1);

	l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']) , hidden_layer_2['biases'] );
	l2 = tf.nn.tanh(l2);

	l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']) , hidden_layer_3['biases'] );
	l3 = tf.nn.tanh(l3);

	l4 = tf.add(tf.matmul(l3,hidden_layer_4['weights']) , hidden_layer_4['biases'] );
	l4 = tf.nn.tanh(l4);

	l5 = tf.add(tf.matmul(l4,hidden_layer_5['weights']) , hidden_layer_5['biases'] );
	l5 = tf.nn.tanh(l5);

	output = tf.add(tf.matmul(l5,output_layer['weights']) , output_layer['biases'] );

	return output;

prediction = neural_network_model(x);
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y));

optimizer = tf.train.AdamOptimizer().minimize(cost);

n_epoch = 50;

def get_next_epoch_data():

	out_x = x_train[cur: cur+ batch_size,:];
	out_y = y_train[cur: cur+ batch_size];

	return out_x,out_y;

with tf.Session() as sess :

	sess.run(tf.global_variables_initializer());

	for epoch in range(n_epoch):

		epoch_loss =0;
		cur=0;

		for _ in range(int(len(x_train)/batch_size) )  :

			epoch_x , epoch_y = get_next_epoch_data();

			cur+=batch_size;
			_,c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y });
			epoch_loss+=c;


		print('Epoch', epoch, 'completed out of', n_epoch, 'loss:',epoch_loss);

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1));
		accuracy = tf.reduce_mean(tf.cast(correct,'float32'));

		print('Epoch Accuracy: ',accuracy.eval({x:x_test, y:y_test}) );








