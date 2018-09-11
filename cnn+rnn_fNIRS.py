# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:17:04 2018

@author: leeyo
"""

from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
from NIRSPY import NIRSPY_analysis
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import time
import tensorflow as tf
#%%
dataPath = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
#dataPath = r'E:\hxf\Faculty\ColumbiaUniversity\dataprocess\fNIRS\SPM_fnirs\data\SPM12\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
sampling_rate = 10.416667
NIRSPY = NIRSPY_basic(dataPath, sampling_rate) #object
fileName = 'OxyHb_label.csv'
NIRSPY.set_file_name(fileName)
HbO = NIRSPY.load_Data_csv()
# Setting up basic directory and loading csv file OxyHb_label.csv
#%%
np_analysis = NIRSPY_analysis(5) #have in total 5 splits, sequencentially
X_train, X_test, Y_train, Y_test = np_analysis.time_series_split(HbO) #hxf: please put a link reference
#%%
np_basic = NIRSPY_basic(dataPath, sampling_rate)
labels_train, features_train = Y_train[3], X_train[3] #first 4 splits for training
labels_valid, features_valid = label_binarize(Y_test[3],[0,1]), X_test[3] #the 4th split for validatoin
labels_test, features_test = label_binarize(Y_test[4],[0,1]), X_test[4] #last split for testing
#hxf: please verify above by using sequentially 60% for training, 20% for validation, and 20% for testing
# verified

np_pre = NIRSPY_preprocessing(sampling_rate) # object for preprocessing
drift_removed= np_basic.standardize(np_pre.baseline_drift(features_train, 4)) #poly degrees, maximum is 4 degrees
row, col = drift_removed.shape
feature_and_label = np.hstack([drift_removed, labels_train[:,None]]) #include label
#%%
features_final_val = np_basic.standardize(np_pre.baseline_drift(features_valid, 4))
row_val, col_val = features_final_val.shape
features_final_val = features_final_val.reshape(row_val,col_val,1).astype(np.float32)
features_final_test = np_basic.standardize(np_pre.baseline_drift(features_test, 4))
row_test, col_test = features_final_test.shape
features_final_test = features_final_test.reshape(row_test,col_test,1).astype(np.float32)
# processing validation and test data set and setting types to be compatible with tensorflow
#%%
bootstrapped_feature, bootstrapped_label = np_analysis.bootStrapping(feature_and_label, [0.8, 0.2], 200)
for i in range(5000):
    print(i)
    temp_feature, temp_label = np_analysis.bootStrapping(feature_and_label, [0.8,0.2], 200)
    bootstrapped_feature, bootstrapped_label = np.vstack((temp_feature, bootstrapped_feature)), np.vstack((temp_label, bootstrapped_label))
#%%
bootstrapped_label = bootstrapped_label.reshape(len(bootstrapped_feature),1)
#%%
combine_both = np.hstack((bootstrapped_feature, bootstrapped_label))
#%%
seed = 7
np.random.seed(seed)
# setting seed
#%%
sess = tf.InteractiveSession()
learning_rate = 1e-4 # learning rate for adam optimizer
size_batch = 200 # mini_batch size
n_epoch = 100 # number of epoch
num_filter = 8 # number of filter. Right now, 3 Convolutional Layers and 16 -> 32 -> 64 feature maps
size_kernel_1st = 3 # always use odd size!!!
size_kernel_2nd = 5
size_kernel_3rd = 7
num_stride = 1
n_class = 2
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # random values from truncated normal or bounded normal distribution
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#Convolution and Pooling
#TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations into functions.
#refer to https://www.tensorflow.org/api_docs/python/tf/nn/conv2d 
def conv1d(x, W):
  return tf.nn.conv1d(x, W, stride = 1, padding='VALID')

#refer to https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
# =============================================================================
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
# 
# =============================================================================
# =============================================================================
# def conv1d(X, num_filter, size_kernel, num_stride):
#   return tf.layers.conv1d(X, filters = num_filter, kernel_size = size_kernel, strides = num_stride, padding = 'SAME', activation = tf.nn.relu)
# 
# =============================================================================

X = tf.placeholder(tf.float32, shape = (None, col, 1), name = 'X')
y = tf.placeholder(tf.int32, shape = (None), name = 'y')

# 1st convolutional layer
W_conv1 = weight_variable([size_kernel_1st, 1, num_filter]) # weight variable. Returns tensor with [1, 3, 1, 16] in our case. 
b_conv1 = bias_variable([num_filter])
h_conv1 = tf.nn.relu(conv1d(X, W_conv1) + b_conv1)
print(h_conv1.shape)
#%%
# 2nd Convolutional Layer
W_conv2 = weight_variable([size_kernel_2nd, num_filter, 2 * num_filter])
b_conv2 = bias_variable([2 * num_filter])
h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)
print(h_conv2.shape)
#%%
# 3rd Convolutional Layer
W_conv3 = weight_variable([size_kernel_3rd, 2 * num_filter, 4 * num_filter])
b_conv3 = bias_variable([4 * num_filter])
h_conv3 = tf.nn.relu(conv1d(h_conv2, W_conv3) + b_conv3)
print(h_conv3.shape)
#%%
# 4th Convolutional Layer
W_conv4 = weight_variable([1, 4 * num_filter, 100])
b_conv4 = bias_variable([1])
h_conv4 = tf.nn.relu(conv1d(h_conv3, W_conv4) + b_conv4)
print(h_conv4.shape)
#%%
#fully connection layer
n_neurons_dense = 100
W_fc1 = weight_variable([col * (4 * num_filter), n_neurons_dense])
b_fc1 = bias_variable([n_neurons_dense])
h_conv4_flat = tf.reshape(h_conv4, [-1, col * (4 * num_filter)]) #flattening
h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
print(h_fc1.shape)
#%%
#dropout
keep_prob = tf.placeholder(tf.float32) # will make it as 0.5
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.shape)
#%%
#Readout Layer
W_fc2 = weight_variable([n_neurons_dense, n_class])
b_fc2 = bias_variable([n_class])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.shape)
#%%
## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
#%%
# hxf 4/29/2018 preparing for saving the mode
result_saved_root = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Aug 20th 2018\result4\CNN_3.txt'
path, filename = os.path.split(result_saved_root)  # refer to https://stackoverflow.com/questions/10507298/splitting-path-strings-into-drive-path-and-file-name-parts
tempFilename = os.path.splitext(filename)  # refer to https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python
sModelPath = path + "/" + tempFilename[0]  # without extension

#%%
for epoch in range(50):
    print("Epoch: ", epoch)
    for iteration in range(row // size_batch):
        X_batch, y_batch = np_analysis.bootStrapping(feature_and_label, [0.8,0.2], size_batch)
        # boot strapping the sample to remedy class imbalance
        X_batch = X_batch.reshape(size_batch, col, 1).astype(np.float32)
        # reshape features according,y
        y_batch = label_binarize(y_batch,[0,1])
        # one hot encoding
        train_step.run(feed_dict={X: X_batch, y: y_batch, keep_prob: 0.5})
        # train our model with 0.5 dropout
        #acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, keep_prob: 0.6})
        #print('Train Accuracy: ', acc_train)
        if iteration % 8 == 0: # at every 8th iteration
            startTime = time.time()
            # batch loss and accuracy
            loss = cross_entropy.eval(feed_dict={X: X_batch, y: y_batch, keep_prob:1.0}) # loss is target function that our model aims to minimize
            accuracy_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, keep_prob: 1.0})
            # validation loss and accuracy
            acc_val = accuracy.eval(feed_dict = {X: features_final_val, y: labels_valid, keep_prob: 1.0})
            #loss_val = cross_entropy.eval(feed_dict={X: features_final_val, y: labels_valid, keep_prob: 1.0})
            print("Train Accuracy: ", accuracy_batch, "Val_accuracy: ", acc_val, "Train Loss: ", loss)
            average_time = (time.time() - startTime) / 1000
            with open(result_saved_root, 'a') as f: # appending mode
                print("iteration: " + str(epoch) + ', totalTrainSize = ' + str(X_batch.shape[0]) + ", Minibatch Loss = " +
                          "{:.4f}".format(loss) + ", Training accuracy= " + "{:.3f}".format(accuracy_batch),
                          ", Validation accuracy= " + "{:.3f}".format(acc_val),
                          ", average validation time = " + "{:.5f}".format(average_time), file=f)
            if accuracy_batch >= 0.80 and acc_val >= 0.80:
                startTime = time.time()
                test_accuracy = accuracy.eval(feed_dict={X: features_final_test, y: labels_test, keep_prob: 1.0})
                average_time = (time.time() - startTime) / 1000
                print("Testing accuracy:", test_accuracy)
                with open(result_saved_root, 'a') as f:
                    print("...Testing accuracy:" + "{:.3f}".format(test_accuracy) + ", average testing time = " + "{:.5f}".format(average_time), file=f)
                #save the model
                try:
                    oTrainSaver
                except NameError:
                    oTrainSaver=tf.train.Saver()
                save_path = sModelPath + '_' + str(epoch)+'.ckpt'
                oTrainSaver.save(sess, save_path)


#%%
''' keras implementation of professor`s idea'''
from keras import Sequential
from keras.layers import Dense, Conv1D, Activation, Flatten, Dropout
from keras.optimizers import adam
import numpy as np
from sklearn.preprocessing import OneHotEncoder

train_X = combine_both[:,:16].reshape(-1, 16, 1)

oh = OneHotEncoder()
train_y = oh.fit_transform(combine_both[:,-1].reshape(-1, 1))

input_shape = (train_X.shape[1], 1)
#%%
model = Sequential()
model.add(Conv1D(16, 3, padding='valid', input_shape=input_shape))
model.add(Activation('elu'))
model.add(Conv1D(32, 5, padding='valid'))
model.add(Activation('elu'))
model.add(Conv1D(64, 7, padding='valid'))
model.add(Activation('elu'))
model.add(Conv1D(100, 1, padding='same'))
model.add(Activation('elu'))
model.add(Flatten())
#model.add(Dense(256))
model.add(Dropout(0.5))
#model.add(Activation('elu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()
#%%
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=adam(0.001))
history = model.fit(train_X, train_y, validation_split=0.2, batch_size=64, epochs=100)
#%%
test_X1, test_X2 = features_final_val, features_final_test
label_X1, label_X2 = oh.fit_transform(labels_valid).reshape(-1,1), oh.fit(labels_test).reshape(-1,1) 
predicted_X1, predicted_X2 = model.predict_classes(test_X1), model.predict_classes(test_X2)

