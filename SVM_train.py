# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:29:55 2018

@author: leeyo
"""

import os
os.chdir(r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Sep 18th 2018')
# 2D CNN + RNN Implementation
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
from NIRSPY import NIRSPY_analysis
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
#%%
# importing HbO with labels for MI data
#dataPath = r'E:\hxf\Faculty\ColumbiaUniversity\dataprocess\fNIRS\SPM_fnirs\data\SPM12\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
save_Path = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Oct 21st 2018'
dataPath_MI = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_MI\Concentrations with Labels'
dataPath_ME = r'C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_ME\Concentrations with Labels'
sampling_rate = 10.416667
np_basic_MI = NIRSPY_basic(dataPath_MI, sampling_rate)
np_basic_ME = NIRSPY_basic(dataPath_ME, sampling_rate)
fileName_MI = 'OxyHb_label(MI).csv'
fileName_ME = 'OxyHb_Label.csv'
np_basic_MI.set_file_name(fileName_MI)
np_basic_ME.set_file_name(fileName_ME)
HbO_MI = np_basic_MI.load_Data_csv()
HbO_ME = np_basic_ME.load_Data_csv()
#%%
np_analysis = NIRSPY_analysis(5)
X_train, X_test, Y_train, Y_test = np_analysis.time_series_split(HbO_ME)
#%%
np_pre = NIRSPY_preprocessing(sampling_rate) # object for preprocessing
X_train_baseline = np_pre.baseline_drift(X_train[4])
X_test_baseline = np_pre.baseline_drift(X_test[4])
X_train_final = np_basic_ME.normalize(X_train_baseline)
X_test_final = np_basic_ME.normalize(X_test_baseline)
#%%
smote_enn = SMOTEENN(random_state = 42)
X_resampled, y_resampled = smote_enn.fit_sample(X_train_final, Y_train[4])
#%%
from sklearn.utils import shuffle
X, y = shuffle(X_resampled, y_resampled, random_state = 42)
#%%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
args = {"C": [4,5,5.1,5.2], "kernel":("linear", "rbf"), "gamma" : [5.5,6,7,8,10]}
model_SVM = SVC()
#model_SVM.fit(X,y)
clf = GridSearchCV(model_SVM, args, cv = 4, scoring = 'f1')
clf.fit(X, y)
#%%
clf.best_params_
#%%
predicted = clf.predict(X_test_final)
#predicted = model_SVM.predict(X_test_final)
accuracy_score(Y_test[4], predicted)
np_analysis.class_accuracy(Y_test[4], predicted)
#%%
import pickle
modelName = 'SVM_model[0.898, 0.296] acc 0.765.sav'
pickle.dump(clf, open(modelName, "wb"))
#%%
svc = SVC(clf.best_estimator_)
svc.fit(X,y)
tempp = svc.predict(X_test_final)
accuracy_score(Y_test[4], tempp)
np_analysis.class_accuracy(Y_test[4], tempp)
#%%
accuracy_score(Y_test[4], predicted)