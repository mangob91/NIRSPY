# -*- coding: utf-8 -*-
"""
hxf
"""
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
import numpy as np
from NIRSPY import NIRSPY_analysis

dataPath = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
#dataPath = r'E:\hxf\Faculty\ColumbiaUniversity\dataprocess\fNIRS\SPM_fnirs\data\SPM12\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'

sampling_rate = 10.416667
NIRSPY = NIRSPY_basic(dataPath, sampling_rate) #object
fileName = 'OxyHb_label.csv'
NIRSPY.set_file_name(fileName)
HbO = NIRSPY.load_Data_csv()
# Setting up basic directory and loading csv file OxyHb_label.csv
#%%
np_analysis = NIRSPY_analysis(5)
X_train, X_test, Y_train, Y_test = np_analysis.time_series_split(HbO)
#%%
np_basic = NIRSPY_basic(dataPath, sampling_rate)
labels_train, features_train = Y_train[3], X_train[3]
labels_valid, features_valid = Y_test[3], X_test[3]
labels_test, features_test = Y_test[4], X_test[4]
np_pre = NIRSPY_preprocessing(sampling_rate)
temp= np_basic.standardize(np_pre.baseline_drift(features_train, 4))
row, col = temp.shape
features_final = np.hstack([temp, labels_train[:,None]])
#%%
features_final_val = np_basic.standardize(np_pre.baseline_drift(features_valid, 4))
features_final_test = np_basic.standardize(np_pre.baseline_drift(features_test, 4))
#%%
seed = 7
np.random.seed(seed)
#%%
# number of neuron in hidden layers = 4409/(2*(16+2)) seems like a good place to start 
import tensorflow as tf
n_inputs = col
n_hidden1 = 200
n_hidden2 = 150
n_outputs = 2

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X') # input layer
y = tf.placeholder(tf.int64, shape = (None), name = 'y') # target layer
#%% functino that creates neuron layer
def neuron_layer(X, n_neurons, name, activation = None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1]) # looking at number of features
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev) # holds all weight information
        # initialized randomly using truncated Gaussian with standard deviation of 2/sqrt(n_input + n_neurons)
        W = tf.Variable(init, name = 'kernel') # weights
        b = tf.Variable(tf.zeros([n_neurons]), name = 'bias') # bias
        Z = tf.matmul(X,W) + b
        if activation is not None: # only if activation parameters are provided, such as tf.nn.relu
            return activation(Z)
        else:
            return Z
#%% model
with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name = 'hidden2', activation = tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name = 'outputs')
#%% cost function
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(cross_entropy, name = 'loss')
#%% parameter tuning
learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
#%%
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#%%
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#%%
num_epoch = 7
batch_size = 240
with tf.Session() as sess:
    init.run()
    for epoch in range(num_epoch):
        for iteration in range(row // batch_size):
            X_batch, y_batch = np_analysis.bootStrapping(features_final, [0.8,0.2], batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict = {X: features_final_val, y: labels_valid})
        print(epoch, "Train accuracy: ", acc_train, "Val_accuracy: ", acc_val)
    save_path = saver.save(sess, "./my_model_final.ckpt")
#%%
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        X_new_scaled = features_final_test
        Z = logits.eval(feed_dict = {X: X_new_scaled})
        y_pred = np.argmax(Z, axis = 1)
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, y_pred))