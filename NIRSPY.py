# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:08:32 2018

@author: leeyo
"""
import os
import numpy as np
import scipy.io as sio
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import mutual_info_classif 
from sklearn.model_selection import TimeSeriesSplit
from MyCircularQueue import MyCircularQueue

def data_sorting(data, num_classes):
    # creating a map sorting by its classes
    range_keys = list(range(0,num_classes))
    sorted_classes = dict.fromkeys(range_keys)
    for i in range_keys:
        sorted_classes[i] = data[data[:,-1] == i]
    return sorted_classes

def data_sampling(data, size):
    # expects data to be an array
    return data[np.random.randint(0, data.shape[0], int(size)), :]

def check_Path(dataPath):
    return os.path.exists(dataPath)

def remove_trend(data, cof, p_degree):
    y = np.zeros(data.shape)
    row = data.shape[0]
    col = data.shape[1]
    for j in range(col):
        for i in range(row):
            y[i,j] = np.dot(calculate_trend(i, p_degree), cof[:,j])
    return data - y
def calculate_trend(time_point, p_degree):
    # p_degree contains polynomials. for example, if degree is 4, then [1,2,3,4]
    temp = [time_point ** x for x in p_degree]
    temp.append(1)
    return temp
        
def poly_fit(data, deg = None):
    if deg is None:
        deg = 4
    return np.polyfit(range(len(data)), data, deg)
 
class NIRSPY_basic:
    _dataPath = ''
    _work_dir = ''
    _file_name = ''
    __sampling_rate = float(0)
    # this field contains information of datapath
    def __init__(self, dataPath, sampling_rate):
        self.change_Path(dataPath)
        temp_head, temp_tail = os.path.split(self._dataPath)
        if(temp_tail.endswith('.mat') or temp_tail.endswith('.csv')):
            self.set_file_name(temp_tail)
            self.set_work_dir(temp_head)
        else:
            self.set_work_dir(dataPath)
        self.__sampling_rate = sampling_rate
        
    def get_dataPath(self):
        return self._dataPath
    
    def get_work_dir(self):
        return self._work_dir
    
    def get_file_name(self):
        return self._file_name
    
    def set_file_name(self, fileName):
        self._file_name = fileName
        
    def set_work_dir(self, workDir):
        self._work_dir = workDir
        
    def change_Path(self, dataPath):
        if check_Path(dataPath):
            self._dataPath = dataPath
        else:
            print('Given data path does not exist')
    
    def load_Data_m(self, dataPath = None):
        # Reading .m datafile potential use for multiple condition file
        if dataPath is None:
            dataPath = os.path.join(self.get_work_dir(), self.get_file_name())
        _, tail = os.path.split(dataPath)
        if tail.endswith('.csv'):
            print('File extension is .csv, load_Data_csv is being called instead')
            self.load_Data_csv(dataPath)
        else:
            return sio.loadmat(dataPath)
    
    def load_Data_csv(self, dataPath = None):
        if dataPath is None:
            dataPath = os.path.join(self.get_work_dir(), self.get_file_name())
        _, tail = os.path.split(dataPath)
        if tail.endswith('.mat'):
            print('File extension is .mat, load_Data_m is being called instead')
            self.load_Data_m(dataPath)
        else:
            return np.genfromtxt(dataPath, delimiter = ',')
        
    def normalize(self, features):
        # This is just standard normalizer
        scaler = Normalizer.fit(features)
        normalized_features = scaler.transform(features)
        return normalized_features
    
    def standardize(self, features):
        scaler = StandardScaler().fit(features)
        standardized_features = scaler.transform(features)
        return standardized_features
    
    #def signal_processing():
        
    #def feature_selection(self):
class NIRSPY_preprocessing:
    _sampling_rate = float(0)
    time = float(0)
    def __init__(self, sampling_rate):
        self._sampling_rate = sampling_rate
        
    def get_Hz(self):
        return self._sampling_rate
    def set_Hz(self, sampling_rate):
        if self.__sampling_rate == 0:
            self.__sampling_rate = sampling_rate
        else:
            raise Exception('Sampling rate is already set')
    def normalize(self, features_orig, features_new, interval, time):
        # this one is based on page 1 (Lapborisuth 2017)
        num_tp = round(interval * self._sampling_rate)
        # num_tp means number of time points
        time_loc = round(time * self._sampling_rate)
        start_loc = time_loc - num_tp
        if num_tp > time_loc:
            raise Exception('Not enough of data')
        else:
            temp = features_orig.iloc[start_loc:time_loc,:]
            avg_prev = temp.mean()
            sigma_prev = temp.std()
            return (features_new - avg_prev)/sigma_prev
    def feature_extraction(self, features):
        # given array, extracts time averages of change
        tp = math.ceil(self.get_Hz())
        row = features.shape[0]
        # Round up the sampling rate, for example 10.416667 is 11
        remainder = row % tp
        new_row = row-remainder
        j = 0
        for i in range(0, new_row, tp):
            time_sum = features[i:i+tp].sum(axis = 0)
            features[j] = time_sum 
            j += 1
        features[j] = features[-remainder:].sum(axis = 0)
        return features[0:j+1]
    def feature_extraction_label(self, labels):
        label = self.feature_extraction(labels)
        label[label > 0] = 1
        return label
    
    def baseline_drift(self, data, degree = None):
        if degree is None:
            degree = 4
        # expects a numpy ndarray. it is faster than dataframe.
        p_degree = list(range(degree, 0, -1))
        cof = np.apply_along_axis(poly_fit, 0, data, degree)
        drift_removed = remove_trend(data, cof, p_degree)
        return drift_removed
    
    def feature_selection(self, data, target_label, num_channels):
        mutual_info = mutual_info_classif(data, target_label)
        sorted_index = mutual_info.argsort()
        return data[:,sorted_index[-num_channels:]]
    
    #def wavelet_MDL(self, data):
        # Wavelet Minimum Description Length technique to detrend signals

class NIRSPY_analysis:
    
    def __init__(self, num_split):
        self.num_split = num_split
        self.s_active = 0
        
    def set_num_Split(self, num_split):
        self.num_split = num_split
        
    def get_num_Split(self, num_split):
        return self.num_split
    def time_series_split(self, data, label):
        # returns a dictionary containing time series splitted data
        tscv = TimeSeriesSplit(n_splits = self.num_split)
        X_train, X_test = {},{}
        Y_train, Y_test = {},{}
        key = 0
        for train_index, test_index in tscv.split(data):
            X_train[key], X_test[key] = data[train_index], data[test_index]
            Y_train[key], Y_test[key] = label[train_index], label[test_index]
            key+=1
        return X_train, X_test, Y_train, Y_test
    def set_active(self, index):
        self.s_active = index
        
    def get_active(self):
        return self.s_active
    
    def bootStrapping(self, data, class_ratio, bs_active, bs_rest):
        #batch_size = bs_active + bs_rest
        # This function is based on bootstrapping given by professor Xiaofu He
        # data and ratio between class, for example, [0.5,0.5] or 0.5 class 0 and 0.5 class 1
        # expects data containing features and label (label assumed to be the last column)
        if sum(class_ratio) != 1:
            raise Exception("Class ratios don't add up to 1")
        num_classes = len(class_ratio) # number of classes
        sorted_data = data_sorting(data, num_classes)
        queue = MyCircularQueue(sorted_data[1].shape[0])
        for k in range(sorted_data[1].shape[0]):
            queue.enqueue(k)
        # sorted_data is a map key ranging from 0 to num_classes. This corresponds to n_active, n_rest etc.
        # calculating S_resting, S_active. etc
        #queue_active = queue.Queue(sorted_data[1].shape[0])
        result = np.array([], dtype = np.float64).reshape(0,data.shape[1])
        for i in range(3):
            # pick S_active sequentially
            s_active = self.pick_active(queue, sorted_data[1], self.get_active(), bs_active)
            s_rest = data_sampling(sorted_data[0], bs_rest)
            combined = np.vstack((s_active, s_rest))
            result = np.vstack((result, combined))
        return result
    
    def pick_active(self, queue, s_active_data, index, s_active_size):
        result = np.zeros((s_active_size, s_active_data.shape[1]), dtype = np.float64)
        for i in range(s_active_size):
            temp = queue.dequeue()
            result[i] = s_active_data[temp]
            queue.enqueue(temp)   
        return result

