# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:57:25 2018

@author: leeyo
"""
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
import numpy as np
from NIRSPY import NIRSPY_analysis
#dataPath = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
dataPath = r'E:\hxf\Faculty\ColumbiaUniversity\dataprocess\fNIRS\SPM_fnirs\data\SPM12\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
sampling_rate = 10.416667
NIRSPY = NIRSPY_basic(dataPath, sampling_rate) #object
fileName = 'OxyHb_label.csv'
NIRSPY.set_file_name(fileName)
HbO = NIRSPY.load_Data_csv()
# Setting up basic directory and loading csv file OxyHb_label.csv
#%%
labels = HbO[:,HbO.shape[1]-1] #last column
features = HbO[:, :HbO.shape[1] -1]
tscv = NIRSPY_analysis(6) #6 means number of splits
X_train, X_test, Y_train, Y_test = tscv.time_series_split(features, labels)  
# Splitting full dataset into test and train
#%% processing train data
np_pre = NIRSPY_preprocessing(sampling_rate) #object
drift_removed = np_pre.baseline_drift(X_train[5], 4) #X_train[5] is accumulative, including all training splits, 4 degrees
extracted_feature = np_pre.feature_extraction(drift_removed) #sum up every second data, and select fetures based on the mututul information
label_final = np_pre.feature_extraction_label(Y_train[5])
features_final = np_pre.feature_selection(extracted_feature, label_final, 12) #baedon MI, select the best 12 features
#%%
stdzed = NIRSPY.standardize(features_final)
temp_features = np.hstack([stdzed, label_final[:,None]])
# Standardizing features for SVM then col bind with labels
#%%
from NIRSPY import NIRSPY_analysis
na = NIRSPY_analysis(5)
temp = na.bootStrapping(temp_features, [0.5,0.5],110,80) #batch size is 110 plus 80, should be: ratio (e.g.,[0.42,0.58], batchSize (e.g., 150)
# boot strapping based on algorithm given from professor Xiaofu He
print(sum(temp[:,12] == 1)/len(temp)) #calculate the ratio of active class, replace 12 with a variable which should be the last column
#%%
from sklearn.model_selection import GridSearchCV
from sklearn import svm as svm
from sklearn.metrics import accuracy_score
parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':[1.5,1.6,1.7,1.8,1.9,2], 
              'gamma':[0.01,0.1,0.5,1,1.1,1.2,1.3,1.5,2], 'degree':[2,3,4]}
svr = svm.SVC(class_weight = {1:15})
grid = GridSearchCV(svr, parameters) #hxf why 5??
grid.fit(temp[:,:-1], temp[:,-1].ravel()) #exclude the last column which is the label column
print(grid.best_params_)
# fitting support vector machine for train dataset 
#%% Moving onto processing test dataset
drift_removed_test = np_pre.baseline_drift(X_test[5], 4) #use an availabe above, be consistent with the training processing
extracted_feature_test = np_pre.feature_extraction(drift_removed_test)
label_final_test = np_pre.feature_extraction_label(Y_test[5])
features_final_test = np_pre.feature_selection(extracted_feature_test, label_final_test, 12)
stdzed_test = NIRSPY.standardize(features_final_test)
temp_features_test = np.hstack([stdzed_test, label_final_test[:,None]])
predicted = grid.predict(temp_features_test[:,:-1])
print(accuracy_score(temp_features_test[:,-1],predicted))
# =============================================================================
#%print the best model which was chosen for the prediction
#%%
#example for comparison
# =============================================================================
# import matplotlib.pyplot as plt
# plt.figure(0)
# plt.plot(drift_removed[:,1])
# plt.figure(1)
# plt.plot(features[:,1])
# =============================================================================
