# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:20:00 2018

@author: leeyo
"""
#%%
"""
Importing required packages and libraries
"""
NIRSPY_path = r'C:\Users\leeyo\OneDrive\Documents\Research\Code'
# define path for NIRSPY
import os
os.chdir(NIRSPY_path)
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import Normalizer
#%%
"""
Here, I am using detrended dataset from Tak's toolbox. Currently only working with ME dataset.
This Chunk will read detrended data into the space
"""
path_detrended_ME = r'C:\Users\leeyo\OneDrive\Documents\Research\Motor Analysis\ME'
fileName_ME = 'HbO_ME.csv'
dataPath_ME = os.path.join(path_detrended_ME, fileName_ME)
sampling_rate = 10.417
np_basic_ME = NIRSPY_basic(dataPath_ME, sampling_rate)
np_basic_ME.set_file_name(fileName_ME)
ME = np_basic_ME.load_Data_csv()
#%%
"""
Splitted data 80% for train and 20% for test. Random state was 27.
"""
random_state = 27
X_train, X_test, y_train, y_test = train_test_split(ME[:,:-1], ME[:,-1], test_size=0.2, random_state=random_state)
#%%
"""
I decided to normalize our dataset based on l2 norm. It seems outliers present in our dataset, 
thus it works better if I normalize.
"""
norm = Normalizer(norm = 'l2')
norm.fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
#%%
"""
This step artificially generates train dataset. You can see that resampled train labels have over 60% positive class
"""
smote_enn = SMOTEENN(random_state=random_state)
X_resampled, y_resampled = smote_enn.fit_sample(X_train_norm, y_train)
print(sum(y_resampled == 1)/len(y_resampled))
#%%
"""
Converting 1D sequence vector to 2D mesh according to our brain map. Produces 4 by 4 mesh.
"""
np_pre = NIRSPY_preprocessing(sampling_rate)
mesh_train = np_pre.temp_1D_2D_transform(X_resampled)
mesh_test = np_pre.temp_1D_2D_transform(X_test_norm)
#%%
"""
Converting our labels to onehot matrix
"""
oh = OneHotEncoder()
train_y = oh.fit_transform(y_resampled.reshape(-1,1))
input_shape = (mesh_train.shape[1], mesh_train.shape[2], 1)
#%%
"""
Importing required Keras functionalities
"""
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.optimizers import adam
#%%
"""
Current architecture has 3 convolutional layers with fully connection.
It seems to benefit our model if we restrict size of the first hidden layer.
I also used tanh as our activation function. It should have similar effect as standardization.
batch_size of 32 seems to be optimal in our case.
"""
kernel_size = 3
kernel_number = 3
drop_out_rate = 0.3
learning_rate = 0.0003
batch_size = 32
epoch = 700
model = Sequential()
model.add(Conv2D(kernel_number, kernel_size, padding='same', input_shape=(input_shape)))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Conv2D(kernel_number, kernel_size + 15, padding='same'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Conv2D(kernel_number, kernel_size + 10, padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Flatten())
model.add(Dropout(drop_out_rate))
model.add(Dense(2, activation='sigmoid'))
model.summary()
#%%
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=adam(learning_rate))
history = model.fit(mesh_train, train_y, batch_size=batch_size, epochs=epoch, validation_split = 0.2, class_weight = 'auto')
#%%
import matplotlib.pyplot as plt
save_Path_fig = r'C:\Users\leeyo\OneDrive\Documents\Research\Log\Nov 27th 2018'
plot_Name1 = 'Accuracy and Losses#4.png'
full_plot_Path1 = os.path.join(save_Path_fig, plot_Name1)
fig1 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='lower left')
plt.show()
#%%
from sklearn.metrics import accuracy_score
test_labels = oh.fit_transform(y_test.reshape(-1,1))
predicted = model.predict_classes(mesh_test)
accuracy = accuracy_score(y_test, predicted)
print(accuracy)
#%%
from sklearn.metrics import recall_score
recall_score = recall_score(y_test,predicted)
print(recall_score)
#%%
"""
Saving our trained model
"""
save_Path_model = r'C:\Users\leeyo\OneDrive\Documents\Research\Log\Nov 27th 2018\Trial\model'
model_name = 'my_model3.h5'
model_dir = os.path.join(save_Path_model, model_name)
model.save(model_dir)
#%%
fig1.savefig(full_plot_Path1)