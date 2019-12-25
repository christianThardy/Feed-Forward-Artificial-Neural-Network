# Feed-Forward Neural Network in Tensorflow
# Needs refactoring

# Data preprocessing
# Import data and transform pandas marices to numpy arrays to tensorflow tensors

import pandas as pd
dataset = pd.read_csv('purchase_history.csv')

import numpy as np
X_train = dataset.iloc[:, [2, 3]].as_matrix()
X_data_np = np.asarray(X_train, np.float32)
y_train = dataset.iloc[:, 4].as_matrix()
y_data_np = np.asarray(y_train, np.float32)

import tensorflow as tf
tfX = tf.convert_to_tensor(X_data_np, np.float32)
tfY = tf.convert_to_tensor(y_data_np, np.float32)


# Shows X_train tensor values

with tf.Session() as sess:
	print(sess.run(tfX))

# Shows y_train tensor values

with tf.Session() as sess:
	print(sess.run(tfY))


num_samples = 100 # 100 samples per class - 4 classes
input_dim = 2 # Dimensionality of input 
hidden_layers = 4 # Hidden layer size 
num_classes = 4 # Number of classes 


# Classes represented as 4 gaussian clouds

n1 = np.random.randn(num_samples, 2) + np.array([0, -2]) 
n2 = np.random.randn(num_samples, 2) + np.array([2, 2]) 
n3 = np.random.randn(num_samples, 2) + np.array([-2, 2]) 
n4 = np.random.randn(num_samples, 2) + np.array([0, 0]) 
n = np.vstack([n1, n2, n3, n4])


# Create labels

Y = np.array([0]*num_samples + [1]*num_samples + [2]*num_samples + [3]*num_samples)


# Plotting gaussian clouds

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y, s=100, alpha=0.5)
plt.show()


# Turn Y into an indicator matrix for training

input_label =len(Y)
T = np.zeros((input_label, num_classes))

for i in range(input_label):
	T[i, Y[i]] = 1


# Function for initializing weights

def load_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev = 0.01))

# Forward propagation

def forward(X, W1, b1, W2, b2):
	Z = tf.nn.tanh(tf.matmul(X, W1) + b1)
	# Activation function
	return tf.matmul(Z, W2) + b2

# Create tensorflow placeholders

tfX = tf.placeholder(tf.float32, [None, input_dim])
tfY = tf.placeholder(tf.float32, [None, num_classes])

# Weight initialization

W1 = load_weights([input_dim, hidden_layers])
b1 = load_weights([hidden_layers])
W2 = load_weights([hidden_layers, num_classes])
b2 = load_weights([num_classes])

# Output variable

py_x = forward(tfX, W1, b1, W2, b2)


# Define loss function

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = py_x, labels = tfY))

# Train function tunes optimizers learning rate for desired output

opt = tf.train.GradientDescentOptimizer(1e-1).minimize(loss) 
predict_op = tf.argmax(py_x, 1)      							  
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# Gradient descent
for i in range(1600):

    sess.run(opt, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 10 == 0:

        print('epoch:', np.mean(Y == pred))
