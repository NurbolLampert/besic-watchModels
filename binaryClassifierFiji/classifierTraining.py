import sklearn as sk
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

import pandas as pd

#Use this command for ML classifier, as it is currently the best classifer
def trainClassifierSVC(dataFrame):
    print("Running Linear SVC Classifier")
    #Selecting rows of data from test data and placing into pandas dataframe format
    selectRows = int(.7 * len(dataFrame))
    traindataFrame = pd.DataFrame(dataFrame)
    trainingX = traindataFrame.iloc[:selectRows, 0:3]
    trainingY = traindataFrame.iloc[:selectRows, -1]

    testingX = traindataFrame.iloc[selectRows:, 0:3]
    testingY = traindataFrame.iloc[selectRows:, -1]
    #Normalizing data to prevent issues with convergence
    print("Normalizing Data")
    startNorm = time.time()
    dataScaling = StandardScaler()
    fitTrainingX= dataScaling.fit(trainingX)
    print("X Y Z mean: " + str(dataScaling.mean_))
    normalizedTrainingDataX = dataScaling.transform(trainingX)
    normalizedTestingDataX = dataScaling.transform(testingX)
    #print(normalizedTestingDataX[0,0:3])
    endNorm = time.time() - startNorm
    print("Normalization time: " + str(endNorm))
    #Beginning classifier training code
    print("Starting training :")
    trainStart = time.time()
    #Using 100000 iterations in order to ensure that the program converges and does not throw an error
    #line below is initializing the classifer
    runClassifier = svm.LinearSVC(max_iter=100000)
    #Line below is training/fitting the data in order to create the classifer
    runClassifier.fit(normalizedTrainingDataX,trainingY)
    trainEnd = time.time() - trainStart
    print("Data training Complete: " + str(trainEnd))

    #After classifer is trained, using the classifier on training data for prediction (this should be higher accuracy, as
    #the classifer is running on the data it was trained on, do not use this as a metric for determining overall accuracy)
    afterRunningTraining = runClassifier.predict(normalizedTrainingDataX)
    #Obtain the accuracy score of the data
    trainingAccuracy = accuracy_score(trainingY, afterRunningTraining)



    print("Starting prediction on testing data: ")
    testStart = time.time()
    #Using trained classifer to predict worn/unworn state of data with labels that are hidden from the code
    afterRunningTest = runClassifier.predict(normalizedTestingDataX)
    testEnd = time.time() - testStart
    #Determine how accurate the classifer is on the blind test data
    testingAccuracy = accuracy_score(testingY, afterRunningTest)
    #Get confusion matrix and classification report
    confusion = confusion_matrix(testingY, afterRunningTest)
    reportClass = classification_report(testingY,afterRunningTest)
    print("Data testing Complete: " + str(testEnd))

    #Printing results
    print("Linear SVC Training accuracy {: .2f}%".format(trainingAccuracy*100))
    print("Linear SVC Testing accuracy {: .2f}%".format(testingAccuracy*100))
    print("Confusion Matrix results \n")
    print(confusion)
    print("Classification Report \n")
    print(reportClass)


#def trainClassifierOneVsAllSVC(dataFrame):


# def trainClassifierSVM():
#
#
# def trainClassifierNN(dataFrame):
#     print("Running Linear SVC Classifier")
#     selectRows = int(.7 * len(dataFrame))
#     traindataFrame = pd.DataFrame(dataFrame)
#     trainingX = traindataFrame.iloc[:selectRows, 0:3]
#     trainingY = traindataFrame.iloc[:selectRows, -1]
#
#     testingX = traindataFrame.iloc[selectRows:, 0:3]
#     testingY = traindataFrame.iloc[selectRows:, -1]
#
#     print("Normalizing Data")
#     dataScaling = StandardScaler()
#     fitTrainingX = dataScaling.fit(trainingX)
#     print(dataScaling.mean_)
#     normalizedTrainingDataX = dataScaling.transform(trainingX)
#     normalizedTestingDataX = dataScaling.transform(testingX)
#
#     runClassifier = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='sigmoid',solver='adam')
#
#     print("Starting training :")
#     trainStart = time.time()
#     runClassifier.fit(normalizedTrainingDataX,trainingY)
#     trainEnd = time.time() - trainStart
#     print("Data training Complete: " + str(trainEnd))