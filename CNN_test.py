# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:54:27 2018

@author: leeyo
"""
"""
This is CNN_test.py. It loads our trained model and validates our performance.
"""
#%%
"""
Same step as before. This loads NIRSPY and data.
"""
import os
path_NIRSPY = r"C:\Users\leeyo\OneDrive\Documents\Research\Code\Python"
os.chdir(path_NIRSPY)
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
from NIRSPY import NIRSPY_analysis
from sklearn.preprocessing import Normalizer
import numpy as np
#%%
# Load testing fNIRS Data
dataPath = r'C:\Users\leeyo\OneDrive\Documents\Research\Motor Analysis\ME'
fileName_ME = 'HbO_ME.csv'
sampling_rate = 10.417
np_basic_ME = NIRSPY_basic(dataPath, sampling_rate)
np_basic_ME.set_file_name(fileName_ME)
np_analysis = NIRSPY_analysis(5)
ME = np_basic_ME.load_Data_csv()
#%%
"""
L2 Normalize data and then transform to 2D mesh
"""
features = ME[:,:-1]
labels = ME[:,-1]
norm = Normalizer(norm = 'l2')
norm.fit(features)
np_pre = NIRSPY_preprocessing(sampling_rate)
#%%
"""
Loading trained model
"""
from keras.models import load_model
model = load_model(r'C:/Users/leeyo/OneDrive/Documents/Research/Log/Nov 27th 2018/Trial/model/best/my_model.h5')
#%%
import time
print("..wait for 20s.....")
sleepTime=20
#%%
windowSize=20 #i.e., every 1 time point
stepSize=1 #suppose windowsize=1 time points with a step size 1
# Predict every windowSize time points and cacluate the time 
#%%
savePath = r"C:\Users\leeyo\OneDrive\Documents\Research\Demo"
npyFile = "test.txt"
savePath_full = os.path.join(savePath, npyFile)
#%%
predicted_labels = []
with open(savePath_full, "w") as f:
    i,j = 0,1
    n_row = len(features)
    while i < n_row:
        startT = time.time()
        if (i + windowSize) > n_row:
            X = norm.transform(features[i:,:])
            y = labels[i:]
        else:
            X = norm.transform(features[i:i + windowSize,:])
            y = labels[i:i + windowSize]
        predicted_label = model.predict_classes(np_pre.temp_1D_2D_transform(X))
        real_label = np.array2string(y, precision = 1, separator = ',', suppress_small = True)
        predicted_label_str = np.array2string(predicted_label, precision = 1, separator = ',', suppress_small = True)
        stopT = time.time()
        interval = stopT-startT
        minibatch_acc = np_analysis.class_accuracy(predicted_label, y, [0,1])
        S = ['Current Sample #', str(j), ", real_label: ", real_label, ", predicted label: ", predicted_label_str, ', time: ', str(interval), ", accuracy class#1 (rest): ",str(minibatch_acc[0] * 100),", accuracy class#2 (squeeze): ", str(minibatch_acc[1] * 100),"\n"]
        output = ''.join(S)
        predicted_labels = np.append(predicted_labels, predicted_label)
        f.write(output)
        i = i + windowSize
        j+=1
    class_accuracy = np_analysis.class_accuracy(labels, predicted_labels, [0,1])
    last_line = ["Model's performance: for class#1 (rest):", str(class_accuracy[0] * 100), "%; for class#2 (squeeze): ", str(class_accuracy[1] * 100), "%"]

    f.write(''.join(last_line))
    f.close()