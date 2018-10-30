import os
path_NIRSPY = r"C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Sep 18th 2018"
os.chdir(path_NIRSPY)
#%%
from NIRSPY import NIRSPY_basic
from NIRSPY import NIRSPY_preprocessing
from NIRSPY import NIRSPY_analysis
# Load testing fNIRS Data
filePath_ME = r"C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_ME\Concentrations with Labels"
filePath_MI = r"C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\fnirs_data\raw_data\nirs_MI\Concentrations with Labels"
fileName_ME = "OxyHb_label.csv"
fileName_MI = "OxyHb_label(MI).csv"  
#%%
fullData_ME = os.path.join(filePath_ME, fileName_ME)
fullData_MI = os.path.join(filePath_MI, fileName_MI)
#%%
sampling_rate = 10.416667
np_basic = NIRSPY_basic(fullData_ME, sampling_rate)
np_pre = NIRSPY_preprocessing(sampling_rate)
testData_full = np_basic.load_Data_csv()
#%%
testData = testData_full[:,:-1] #shape(testData)=[sampleNum,channelNum], e.g., [100, 16], means 100 timepoints, each sample has 16 channels (depends on the real data channel#)
testLabel = testData_full[:,-1]
#%%
# Load trained model from the disk
import pickle
modelPath = r"C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Oct 21st 2018\Model"
modelFile = "./SVM_model[0.898, 0.296] acc 0.765.sav"
modelPath_full = os.path.join(modelPath, modelFile)
svmModel = pickle.load(open(modelPath_full, 'rb'))
#%%
import time
print("..wait for 20s.....")
sleepTime=20 #20 s
#%%
windowSize=20 #i.e., every 1 time point
stepSize=1 #suppose windowsize=1 time points with a step size 1
# Predict every windowSize time points and cacluate the time 
#%%
savePath = r"C:\Users\leeyo\OneDrive\DOCUME~3-DESKTOP-UT3R6M6-266\MATLAB\New_folder\Andrew_Lee\DSI\On Going\Oct 21st 2018\Model"
npyFile = "test.txt"
savePath_full = os.path.join(savePath, npyFile)
#%%
np_analysis = NIRSPY_analysis(5)
predicted_labels = []
with open(savePath_full, "w") as f:

    import numpy as np
    i = 0
    n_row = len(testData)
    j = 1
    while i < n_row:
        startT = time.time()
        if (i + windowSize) > n_row:
            X = np_basic.normalize(np_pre.baseline_drift(testData[i:, :]))
            y = testLabel[i:]
        else:
            X = np_basic.normalize(np_pre.baseline_drift(testData[i:i + windowSize, :]))
            y = testLabel[i:i + windowSize]
        predicted_label = svmModel.predict(X)
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
    class_accuracy = np_analysis.class_accuracy(testLabel, predicted_labels, [0,1])
    last_line = ["Model's performance: for class#1 (rest):", str(class_accuracy[0] * 100), "%; for class#2 (squeeze): ", str(class_accuracy[1] * 100), "%"]

    f.write(''.join(last_line))
    f.close()