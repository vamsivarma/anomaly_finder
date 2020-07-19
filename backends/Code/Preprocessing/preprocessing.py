'''
Created on 05/05/2020

@author: Francesco Pugliese, Vamsi Gunturi
'''

from Misc.mongo_wrapper import MongoWrapper as mdb 
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pandas as pd

# import keras 
from keras.datasets import mnist

# import os
import os
from os.path import isfile, isdir, join

#other imports
import numpy as np

# True for local MongoDB
# False for remote Atlas MongoDB
mongoFlag = False

class Preprocessing(object):
    def __init__(self, params):
        self.__params = params
    
    # Private method for normalization
    def __normalize_features(self, train_x, test_x):
        train_x = train_x.copy()
        test_x = test_x.copy()
        
        for feature_i in range(train_x.shape[1]):
            train_values = train_x[:,feature_i]
            test_values = test_x[:,feature_i]
            scaler = MinMaxScaler(feature_range=[0,1])
            train_values = scaler.fit_transform(train_values.reshape(-1, 1))
            test_values = scaler.transform(test_values.reshape(-1,1))
            
            train_x[:, feature_i] = train_values.flatten()
            test_x[:, feature_i] = test_values.flatten()
        
        return train_x, test_x    
    
    # Private method for encoding and preparing data
    def __data_preparation(self, dataset, classify):
        # integer encode direction
        dataset = dataset.drop(['duration','protocol_type','service','flag'], axis=1)
        if classify == False: 
            dataset_x = dataset.drop('class', axis=1).astype(float).values        
            encoder = LabelEncoder()
            labels = 1 - encoder.fit_transform(dataset['class'].values)  # 0: normal, 1: anomaly
            classes = len(np.unique(labels))
        else: 
            dataset_x = dataset.astype(float).values
            
        print('\nIntrusion Detection Dataset Numeber of Rows: ', (dataset_x.shape[0]))
        print('Intrusion Detection Dataset Number of attributes per row: ', (dataset_x.shape[1]))

        # Normalize the input between 0 - 1
        if self.__params.normalize_x == True:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            dataset_x = x_scaler.fit_transform(dataset_x)            
        
        if classify == False: 
            print('Number of Classes: ', (classes))
            sss = StratifiedShuffleSplit(n_splits = 1, test_size=self.__params.test_set_perc, random_state=self.__params.random_seed)
            for train_idx, test_idx in sss.split(dataset_x, labels):
                train_x = dataset_x[train_idx]
                train_y = labels[train_idx]
                test_x = dataset_x[test_idx]
                test_y = labels[test_idx]
                
        #train_x, test_x = self.__normalize_features(train_x, test_x)

        if classify == False: 
            return [train_x, train_y, test_x, test_y, classes]
        else: 
            return dataset_x
        
    def load_intrusion_detection_train_data_from_mongodb(self, __params):
        print ("\nLoading data from Mongo DB...")
        
        # True for Local Mongo DB
        # False for remote Atlas instance
        db = mdb(self.__params, mongoFlag) 
        dataset, columns = db.get_dataset('Id_Trainset', False)                     # We read all the configuration of the backend from MongoDB       
        dataset = pd.DataFrame(data = dataset, columns = columns)

        train_x, train_y, test_x, test_y, classes = self.__data_preparation(dataset, False)
        
        return [train_x, train_y, test_x, test_y, classes]
    
    def load_intrusion_detection_train_data_from_csv(self, __params):
        print ("\nLoading data from Csv...")
        dataset_string = os.path.join(self.__params.intrusion_detection_path, self.__params.intrusion_detection_train_filename)
        if os.path.isfile(dataset_string):
            # Reads CSV, extracts interesting columns and recode them as input for the model 
            dataset = pd.read_csv(dataset_string, sep=self.__params.intrusion_detection_field_separator, header=0, nrows = self.__params.limit)             # select only the interested columns
        
        train_x, train_y, test_x, test_y, classes = self.__data_preparation(dataset, False)
        
        return [train_x, train_y, test_x, test_y, classes]
        
    def load_intrusion_detection_classification_data_from_mongodb(self, __params):
        print ("\nLoading data from Mongo DB...")
        
        # True for Local MongoDB
        # Flase for remote Atlas instance
        db = mdb(self.__params, mongoFlag) 
        
        dataset, columns = db.get_dataset('Id_Testset', True)                     # We read all the configuration of the backend from MongoDB       
        dataset = pd.DataFrame(data = dataset, columns = columns)

        classification_x = self.__data_preparation(dataset, True)
        
        return classification_x, dataset
    
    def load_intrusion_detection_classification_data_from_csv(self, __params):
        print ("\nLoading data from Csv...")
        dataset_string = os.path.join(self.__params.intrusion_detection_path, self.__params.intrusion_detection_classification_filename)
        if os.path.isfile(dataset_string):
            # Reads CSV, extracts interesting columns and recode them as input for the model 
            dataset = pd.read_csv(dataset_string, sep=self.__params.intrusion_detection_field_separator, header=0, nrows = self.__params.limit)             # select only the interested columns

        classification_x = self.__data_preparation(dataset, True)
        
        return classification_x, dataset

    def load_mnist_benchmark_data(self, params):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        # Normalize the input between 0 - 1
        if self.__params.normalize_x == True:
            if train_x.max()>test_x.max():                                          # normalize X with respect to overall the data
                dataXMax = train_x.max()   
            else: 
                dataXMax = test_x.max() 

            train_x = train_x / dataXMax                               
            test_x = test_x / dataXMax  

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])        
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])        

        classes = len(np.unique(train_y))

        return [train_x, train_y, test_x, test_y, classes]
