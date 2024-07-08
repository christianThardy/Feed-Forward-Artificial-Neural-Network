Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Function to initialize weights for the network
def load_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev = 0.01))


# Forward propagation
def forward(X, W1, b1, W2, b2):
	Z = tf.nn.tanh(tf.matmul(X, W1) + b1)
	# Activation function
	return tf.matmul(Z, W2) + b2


# Import data
dataset = pd.read_csv('purchase_history.csv')

# Extract features (columns 2 and 3) and convert to numpy array
X_train = dataset.iloc[:, [2, 3]].as_matrix()
X_data_np = np.asarray(X_train, np.float32)

# Extract labels (column 4) and convert to numpy array
y_train = dataset.iloc[:, 4].as_matrix()
y_data_np = np.asarray(y_train, np.float32)

# Convert numpy arrays to TensorFlow tensors
tfX = tf.convert_to_tensor(X_data_np, np.float32)
tfY = tf.convert_to_tensor(y_data_np, np.float32)

# Display X_train tensor values
with tf.Session() as sess:
	print(sess.run(tfX))

# Display y_train tensor values
with tf.Session() as sess:
	print(sess.run(tfY))

# Configuration for sample generation
num_samples = 100 # 100 samples per class - 4 classes
input_dim = 2 # Dimensionality of input features
hidden_layers = 4 # Number of hidden layer neurons 
num_classes = 4 # Number of output classes 

# Generate data for 4 Gaussian clouds representing the 4 classes
n1 = np.random.randn(num_samples, 2) + np.array([0, -2]) 
n2 = np.random.randn(num_samples, 2) + np.array([2, 2]) 
n3 = np.random.randn(num_samples, 2) + np.array([-2, 2]) 
n4 = np.random.randn(num_samples, 2) + np.array([0, 0])
# Stack generated data into a single array
n = np.vstack([n1, n2, n3, n4])

# Create labels for generated data
Y = np.array([0]*num_samples + [1]*num_samples + [2]*num_samples + [3]*num_samples)

# Plotting generated Gaussian clouds with their respective labels
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y, s=100, alpha=0.5)
plt.show()

# Convert class labels Y into a one-hot encoded matrix for training
input_label =len(Y)
T = np.zeros((input_label, num_classes))
for i in range(input_label):
	T[i, Y[i]] = 1

# Create tensorflow placeholders for input features and labels
tfX = tf.placeholder(tf.float32, [None, input_dim])
tfY = tf.placeholder(tf.float32, [None, num_classes])

# Weight initialization for weights and biases
Weight_1 = load_weights([input_dim, hidden_layers])
bias_1 = load_weights([hidden_layers])
Weight_2 = load_weights([hidden_layers, num_classes])
bias_2 = load_weights([num_classes])

# Compute the output of the network
py_x = forward(tfX, W1, b1, W2, b2)

# Define loss function as Softmax cross-entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = py_x, labels = tfY))

# Define the optimizer for training as gradient descent with learning rate of 0.1
opt = tf.train.GradientDescentOptimizer(1e-1).minimize(loss) 
predict_op = tf.argmax(py_x, 1) # Operation to predict class labels

# Initialize TensorFlow session and variables
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Training neural network using gradient descent
for i in range(1600):
    sess.run(opt, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 10 == 0:
	# Print accuracy at every 10th epoch
        print('epoch:', np.mean(Y == pred))

