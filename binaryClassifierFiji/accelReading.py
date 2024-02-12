import json
import pandas as pd
#import firebase-admin
#Uncomment line when firebase functionality is added to pull from server
#TODO: Change to take filenames as args and find abspath for file
class accelerometerReading:

    def __init__(self, x_accel, y_accel, z_accel, wearStatus):
        """

        :param x_accel:
        :param y_accel:
        :param z_accel:
        :param wearStatus:
        """
        self.x_accel = x_accel
        self.y_accel = y_accel
        self.z_accel = z_accel
        self.wearStatus = wearStatus



def jsonToReading(filename):
    """
    This function takes in a file of accelerometer readings, converts them to the accelerometerReading object/class
    format and then returns a list of accelerometerReadings for data processing.
    :param filename: The path to the file containing accelerometer data
    :return: List of accelerometerReading objects
    """
    trainingData = []
    importedData = open(filename)
    datapool = json.load(importedData)
    for i in range(0,len(datapool['accel_readings'])):
        val = accelerometerReading(datapool['accel_readings'][i]['xvalue'],
        datapool['accel_readings'][i]['yvalue'],
        datapool['accel_readings'][i]['zvalue'],datapool['accel_readings'][i]['WearStatus'])
        trainingData.append(val)

    return trainingData

def jsonToListList(filename):
    """
    This function takes in a file of accelerometer readings, converts them to the accelerometerReading object/class
    format and then returns a list of accelerometerReadings for data processing.

    :param filename: The path to the file containing accelerometer data
    :return: List of lists of accelerometer reading data.
    Structure is as follows:
    [
    ['xaccel','yaccel','zaccel','wearstatus'],
    ['xaccel','yaccel','zaccel','wearstatus'], etc..
    ]

    """
    trainingData = []

    importedData = open(filename)

    datapool = json.load(importedData)

    for i in range(0,len(datapool['accel_readings'])):
        readingList = \
        [
        datapool['accel_readings'][i]['xvalue'],
        datapool['accel_readings'][i]['yvalue'],
        datapool['accel_readings'][i]['zvalue'],
        datapool['accel_readings'][i]['WearStatus']
        ]
        trainingData.append(readingList)
    return trainingData

def readingToPandas(listListOfData):
    """
    :param listListOfData: A list of lists of data to be converted to a pandas DataFrame.
    Structure should be as follows
    [
    ['xaccel','yaccel','zaccel','wearstatus'],
    ['xaccel','yaccel','zaccel','wearstatus'], etc..
    ]
    :return: A pandas dataframe of accelerometerReadings
    """
    df = pd.DataFrame(listListOfData,columns=['x-acceleration','y-acceleration','z-acceleration','wear-status'])
    return df


def fullDataOrganizingPipeline(filename):
    """
    This function takes in a file of accelerometer readings, converts them to the accelerometerReading object/class and
    list of lists
    format and then returns all three formats.

    :param filename: The path to the file containing accelerometer data

    :return: A tuple of three data formats.
    First is a list of accelerometerReading objects
    Second is the list of lists format of data
    Last is the data as a pandas DataFrame
    """
    accelerometerReadingObjectList = jsonToReading(filename)
    readingsAsList = jsonToListList(filename)
    dataFrameForm = readingToPandas(readingsAsList)

    return accelerometerReadingObjectList, readingsAsList, dataFrameForm


def firebasePull():
    """
    This function takes in a file of accelerometer readings, converts them to the accelerometerReading object/class and
    list of lists
    format and then returns all three formats.

    :param filename: The path to the file containing accelerometer data

    :return: A tuple of three data formats.
    First is a list of accelerometerReading objects
    Second is the list of lists format of data
    Last is the data as a pandas DataFrame
    """
    return 0