'''
Created on 01/01/2020
Modified on 10/02/2020

@author: Francesco Pugliese, Vamsi Gunturi
'''

# Keras imports
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical

from keras import backend as K
from keras.models import model_from_json

# Program imports
from Preprocessing.preprocessing import Preprocessing
from Settings.settings import SetParameters
from Models.lenet5 import Lenet5
from Models.simple_mlp import SimpleMlp
from Models.complex_mlp import ComplexMlp

# Other imports
import numpy as np
import timeit
import pdb

import pandas as pd


# Classification metrics - Start

# Accuracy score
from sklearn.metrics import accuracy_score
# Example usage: accuracy_score(y_test, y_pred)

# Classification report
# from sklearn.metrics import classification_report
# Example usage: print(classification_report(y_test, y_pred))

# Confusion matrix
# from sklearn.metrics import confusion_matrix
# Example usage: print(confusion_matrix(y_test, y_pred))

# Precision and Recall
from sklearn.metrics import precision_score, recall_score
# Example usage: print("Precision:", precision_score(Y_train, Y_pred))
# Example usage: print("Recall:",recall_score(Y_train, Y_pred))

# F1-score
from sklearn.metrics import f1_score
# Example usage: f1_score(Y_train, Y_pred)

# ROC curve
from sklearn.metrics import roc_auc_score
# Example usage: roc_auc_score(Y_train, Y_pred)

class AnomalyDetectionTraining(object):
    def __init__(self, params, mongodb):
        self.__params = params
        self.__mongodb = mongodb

    # Flattens the list of lists and formats each item to a float
    def get_formatted_floats(self, l):
         f_l = [float('{:f}'.format(item)) for sublist in l for item in sublist]
         return f_l

    # Here t is threshold
    # This is only for testing the metrics
    # Need to remove after unit testing
    def format_sigmoid_result(self, l, t):
        f_l = []
        for i in l:
            if i < t:
                f_l.append(0)
            else:
                f_l.append(1)
        
        return f_l
       
    # To calculate metrics for predictions
    def get_metrics_df(self, pred_map):
        
        train_y = pred_map['train_y']
        pred_train_y = pred_map['pred_train_y']
        test_y = pred_map['test_y']
        pred_test_y = pred_map['pred_test_y']

        metric_values_list = []

        # accuracy
        cur_value_list = ['Accuracy']
        acc_model_train = accuracy_score(train_y, pred_train_y)
        acc_model_test = accuracy_score(test_y, pred_test_y)
        cur_value_list.append(acc_model_train)
        cur_value_list.append(acc_model_train)

        metric_values_list.append(cur_value_list)

        # precision
        cur_value_list = ['Precision']
        p_score_train = precision_score(train_y, pred_train_y, average='weighted')
        p_score_test = precision_score(test_y, pred_test_y, average='weighted')
        
        cur_value_list.append(p_score_train)
        cur_value_list.append(p_score_test)

        metric_values_list.append(cur_value_list)

        # recall
        cur_value_list = ['Recall']
        r_score_train = recall_score(train_y, pred_train_y, average='weighted')
        r_score_test = recall_score(test_y, pred_test_y, average='weighted')
        
        cur_value_list.append(r_score_train)
        cur_value_list.append(r_score_test)

        metric_values_list.append(cur_value_list)

        # F1 score
        cur_value_list = ['F1 score']
        f1m_score_train = f1_score(train_y, pred_train_y, average='weighted')
        f1m_score_test = f1_score(test_y, pred_test_y, average='weighted')
        
        cur_value_list.append(f1m_score_train)
        cur_value_list.append(f1m_score_test)

        metric_values_list.append(cur_value_list)

        # ROC score
        cur_value_list = ['ROC score']
        roc_score_train = roc_auc_score(train_y, pred_train_y)
        roc_score_test = roc_auc_score(test_y, pred_test_y)
               
        cur_value_list.append(roc_score_train)
        cur_value_list.append(roc_score_test)

        metric_values_list.append(cur_value_list)

        metric_columns = ['Metric Name', 'Train set', 'Validation set']

        # Convert the metric results in to dataframe sorted by accuracy value
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        metric_df = pd.DataFrame(metric_values_list, columns = metric_columns)
        
        print(metric_df)
    
        return metric_df

    def train(self):
        default_callbacks = []
        preproc = Preprocessing(self.__params)    
        if self.__params.mnist_benchmark == False: 
            if self.__mongodb == True: 
                train_x, train_y, test_x, test_y, classes = preproc.load_intrusion_detection_train_data_from_mongodb(self.__params)
            else: 
                train_x, train_y, test_x, test_y, classes = preproc.load_intrusion_detection_train_data_from_csv(self.__params)
        else: 
            train_x, train_y, test_x, test_y, classes = preproc.load_mnist_benchmark_data(self.__params)
        
        n_train_batches = len(train_x) // self.__params.train_batch_size                                                                                             
        n_test_batches = len(test_x) // self.__params.test_batch_size                                                                                              
        print('\nTrain set size: ', (train_x.shape[0], train_x.shape[1]))
        print('Test set size: ', (test_x.shape[0], test_x.shape[1]))
        print('Train set number of batches: ', n_train_batches)
        print('Test set number of batches: ', n_test_batches)
        print('\n')
        
        #solver = SGD(lr = self.__params.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        solver = Adam(lr = self.__params.learning_rate)
        #solver = RMSprop(lr = self.__params.learning_rate)

        deepnetwork = SimpleMlp.build(train_x.shape[1], classes, True)
        #deepnetwork = ComplexMlp.build(train_x.shape[1], classes, True)
        
        #train_x = np.reshape(train_x, (7, 7))
        
        #deepnetwork = Lenet5.build(train_x.shape[1], classes, True)
        
        if classes > 2: 
            loss = "categorical_crossentropy"
            train_y = to_categorical(train_y, num_classes=classes)
            test_y = to_categorical(test_y, num_classes=classes)
        else: 
            loss = "binary_crossentropy"
        
        deepnetwork.compile(optimizer = solver, loss = loss, metrics = ["accuracy"])
        
        if self.__params.save_best_model == True:
            if self.__params.OS == "Linux":
                monitor_variable = 'val_acc'
            elif self.__params.OS == "Windows": 
                monitor_variable = 'val_accuracy'
            else:
                monitor_variable = 'val_acc'
           
            #checkPoint=ModelCheckpoint(self.__params.models_path+'/'+model_prefix+parameters.model_file, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            #check_point=ModelCheckpoint(self.__params.intrusion_detection_models_path+'/'+self.__params.intrusion_detection_model_file, save_weights_only=True, monitor=monitor_variable, verbose=1, save_best_only=True, mode='max')
            check_point=ModelCheckpoint(self.__params.intrusion_detection_model_file, save_weights_only=True, monitor=monitor_variable, verbose=1, save_best_only=True, mode='max')
            default_callbacks = default_callbacks + [check_point]
        
        if self.__params.early_stopping == True:
            earlyStopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 10, verbose = 0, mode = 'min') 
            default_callbacks = default_callbacks + [earlyStopping]

        #pdb.set_trace()
        if self.__params.csv_logger == True:
            #csv_logger = CSVLogger(self.__params.log_path +'/' + self.__params.log_file)
            csv_logger = CSVLogger(self.__params.log_file)
            default_callbacks = default_callbacks + [csv_logger]

        print ('\nTraining the neural network...\n')
        print(self.__params.valid_set_perc)
        
        history = deepnetwork.fit(train_x, train_y, validation_split=self.__params.valid_set_perc, epochs = self.__params.epochs_number, batch_size = self.__params.train_batch_size, shuffle=True, callbacks = default_callbacks)

        #pdb.set_trace()

        #print(self.__params.save_best_model)

        # Save the Model after the last epoch, not saving the best model 
        if self.__params.save_best_model == False:
            #deepnetwork.save_weights(self.__params.intrusion_detection_models_path+'/'+self.__params.intrusion_detection_model_file)
            deepnetwork.save_weights(self.__params.intrusion_detection_model_file)

        # serialize model to JSON
        model_json = deepnetwork.to_json()
        with open("id_model.json", "w") as json_file:
            json_file.write(model_json)
            
        print("Saved model to disk")

        '''
        # Need to move this test python file
        # Load the saved model and Evaluate
        # load json and create model
        json_file = open('id_model.json', 'r') 
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.__params.intrusion_detection_model_file)
        print("Loaded model from disk")
        # evaluate loaded model on test data

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
        score = loaded_model.evaluate(test_x, test_y)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        '''

        pred_map = {
            'train_y': train_y,
            'pred_train_y': deepnetwork.predict_classes(train_x).flatten(),
            'test_y': test_y,
            'pred_test_y': deepnetwork.predict_classes(test_x).flatten()
        }
        
        '''
        pred_map['pred_train_y'] = self.get_formatted_floats(pred_map['pred_train_y'])
        pred_map['pred_train_y'] = self.format_sigmoid_result(pred_map['pred_train_y'], 0.3)

        pred_map['pred_test_y'] = self.get_formatted_floats(pred_map['pred_test_y'])
        pred_map['pred_test_y'] = self.format_sigmoid_result(pred_map['pred_test_y'], 0.3)
        '''
        
        metrics_df = self.get_metrics_df(pred_map)

        K.clear_session()

        history_csv = pd.read_csv("history.log")

        return [history_csv, metrics_df]