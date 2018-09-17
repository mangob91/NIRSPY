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
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_classif 
from sklearn.model_selection import TimeSeriesSplit
from MyCircularQueue import MyCircularQueue

def parsing_optode_pos(dataPath):
    with open(dataPath, 'r') as f:
        source_coord = np.empty((0,3), type = np.float64)
        detect_coord = np.empty((0,3), type = np.float64)
        for i, line in enumerate(f):
            if 'S' in line:
                # then source
                temp_source = line.split(',')
                source_index = int(temp_source[0][1])
                source_coord = np.vstack((source_coord,[list(float(x) for x in temp_source[1:])]))
            elif 'D' in line:
                # then detector
                temp_detector = line.split(',')
                det_index = int(temp_detector[0][1])
                
                
                

def data_sampling(data, size):
    # expects data to be an array
    return data[np.random.randint(0, data.shape[0], int(size)), :]

def check_Path(dataPath):
    return os.path.exists(dataPath)

def remove_trend(data, cof, p_degree):
    y = np.zeros(data.shape)
    row = data.shape[0] #time point
    col = data.shape[1] #channel
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
        normalized_features = normalize(features)
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
        # this one is based on (page 3 of Lapborisuth 2017)
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
        #reference: equation 1 of Robinson2016.pdf
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

    #reference:Baseline drift was modeled and removed using a polynomial of
    #the fourth degree, i.e., page 4 of Lapborisuth2017.pdf
    #for raw time points data
    def baseline_drift(self, data, degree = None):
        if degree is None:
            degree = 4
        # expects a numpy ndarray. it is faster than dataframe.
        p_degree = list(range(degree, 0, -1))
        cof = np.apply_along_axis(poly_fit, 0, data, degree)
        drift_removed = remove_trend(data, cof, p_degree)
        return drift_removed

    #reference: based on mutul informaiton, i.e., equations 3 & 4 of Robinson2016.pdf
    def feature_selection(self, data, target_label, num_channels):
        mutual_info = mutual_info_classif(data, target_label) #for returned value, the larger, the better
        sorted_index = mutual_info.argsort() #asending
        return data[:,sorted_index[-num_channels:]] #starting from the last one
    
    def swap_columns(self, data, order):
        i = 0
        nrow, ncol = data.shape[0], data.shape[1]
        results = np.zeros([nrow, ncol])
        for i in range(16):
            results[:,i] = data[:, order[i]]
        return results
           
    def temp_1D_2D_transform(self, data):
        # this function will transform time points * channel data into time_points, 4, 4, 1
        time_points = data.shape[0]
        order = [2,0,6,4,3,1,7,5,9,8,14,12,10,11,15,13]
        col_swapped_data = self.swap_columns(data, order)
        transformed_data = np.zeros([time_points,4,4,1])
        for i in range(time_points):
            transformed_data[i] = col_swapped_data[i].reshape([1,4,4,1])
        return transformed_data
        
    def transform_1D_2D(self, ch_config, optode_pos):
        # this function reads pos information and reorder 1D vector to 2D mesh
        # reference Zhang 2018
        with open(optode_pos,'r') as f:
            coordinates = {} #result_ch
            for line in f:
                temp = [x.strip() for x in line.split(',')]
                if 'S' in temp[0] or 'D' in temp[0]:
                    # source
                    coordinates[temp[0]] = list(map(float, temp[1:]))  
        with open(ch_config, 'r') as f:
            pair_info = {} #result
            for line2 in f:
                temp2 = [y.strip() for y in line2.split(',')]
                if 'Ch' in temp2[0]:
                    continue
                else:
                    pair_info[int(temp2[0])] = list(temp2[1:])
        channel_coordinates = np.zeros((len(pair_info.keys()),3))
        for key, val in pair_info.items():
            # val[0] = source and val[1] = detector
            source = 'S' + val[0]
            detect = 'D' + val[1]
            temp =  [a+b for a, b in zip(coordinates[source], coordinates[detect])]
            # finding average of given two coordinates (source and detector) which corresponds to channel location
            # MATLAB function spm_fnirs_read_pos.m line 81 ~
            channel_coordinates[key - 1] = [x/2 for x in temp]
            #def wavelet_MDL(self, data):
        # Wavelet Minimum Description Length technique to detrend signals
        return channel_coordinates
    
class NIRSPY_analysis:
    
    def __init__(self, num_split):
        self.num_split = num_split
        self.s_rest = 0
        self.sorted_data = []
        self.sorted = False
        
    def set_num_Split(self, num_split):
        self.num_split = num_split
        
    def get_num_Split(self, num_split):
        return self.num_split

    
    def time_series_split(self, data):
        feature = data[:,:-1]
        label = data[:,-1]
        # returns a dictionary containing time series splitted data
        tscv = TimeSeriesSplit(n_splits = self.num_split)
        X_train, X_test = {},{}
        Y_train, Y_test = {},{}
        key = 0
        for train_index, test_index in tscv.split(data):
            X_train[key], X_test[key] = feature[train_index], feature[test_index]
            Y_train[key], Y_test[key] = label[train_index], label[test_index]
            key+=1
        return X_train, X_test, Y_train, Y_test
    def set_rest(self, index):
        self.s_rest = index
        
    def get_rest(self):
        return self.s_rest
    
    def bootStrapping(self, data, class_ratio, n_min_batch):
        bs_active, bs_rest = round(class_ratio[1] * n_min_batch), round(class_ratio[0] * n_min_batch)
        # batch_size = bs_active + bs_rest
        # This function is based on bootstrapping given by professor Xiaofu He
        # data and ratio between class, for example, [0.5,0.5] or 0.5 class 0 and 0.5 class 1
        # expects data containing features and label (label assumed to be the last column)
        if sum(class_ratio) != 1:
            raise Exception("Class ratios don't add up to 1")
        num_classes = len(class_ratio) # number of classes
        if self.sorted is False:
            self.sorted_data = self.data_sorting(data, num_classes)
        queue = MyCircularQueue(self.sorted_data[0].shape[0])
        for k in range(self.sorted_data[0].shape[0]):
            queue.enqueue(k)
        # sorted_data is a map key ranging from 0 to num_classes. This corresponds to n_active, n_rest etc.
        # calculating S_resting, S_active. etc
        # queue_active = queue.Queue(sorted_data[1].shape[0])
        result = np.array([], dtype = np.float64).reshape(0,data.shape[1])
        # pick S_active sequentially
        s_rest = self.pick_rest(queue, self.sorted_data[0], self.get_rest(), bs_rest)
        s_active = data_sampling(self.sorted_data[1], bs_active)
        combined = np.vstack((s_rest, s_active))
        result = np.vstack((result, combined))
        X = result[:,:-1]
        y = result[:,-1].astype(np.int8)
        return X,y
    
    def pick_rest(self, queue, s_rest_data, index, s_rest_size):
        result = np.zeros((s_rest_size, s_rest_data.shape[1]), dtype = np.float64)
        for i in range(s_rest_size):
            temp = queue.dequeue()
            result[i] = s_rest_data[temp]
            queue.enqueue(temp)   
        return result
    
    def data_sorting(self, data, num_classes):
    # creating a map sorting by its classes
        range_keys = list(range(0,num_classes))
        sorted_classes = dict.fromkeys(range_keys)
        for i in range_keys:
            sorted_classes[i] = data[data[:,-1] == i]
        self.sorted = True
        return sorted_classes
