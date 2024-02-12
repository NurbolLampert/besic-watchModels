# Machine learning classification
import os
# For data manipulation
import pandas as pd
import accelReading as ar
import classifierTraining as ct
from sklearn.preprocessing import StandardScaler
import time

#Call full data processing pipeline on collected data
#TODO: Refactor later to run as command line script, change filenames to be taken as args
print("Processing Data")
start = time.time()

filePath = os.getcwd()
organizedData1 = ar.fullDataOrganizingPipeline(filePath+"/besi-c-default-rtdb-accelerometer-export.json")
# organizedData2 = ar.fullDataOrganizingPipeline(filePath+"/annotatedAccelerationdata2.json")
# organizedData3 = ar.fullDataOrganizingPipeline(filePath+"/annotatedAccelerationdata3.json")

#Get dataframes of accelerometer data and concatenate them and convert to dataFrame to create training data input
dFrame1 = organizedData1[2]
# dFrame2 = organizedData2[2]
# dFrame3 = organizedData3[2]

concatLists = pd.concat([dFrame1],axis=0)
asObjects = organizedData1[0]
masterTrainingData = concatLists
end = time.time()-start
print("Data processing time: "+ str(end))
print("Avg per file: " + str(float(end)/3))
print("Data processing complete.")

#Train classifer
startTime = time.time()
ct.trainClassifierSVC(masterTrainingData)
endTime = time.time() - startTime
print("Classifier processing time: "+str(endTime))
#targetVariable = np.where(masterTrainingData)