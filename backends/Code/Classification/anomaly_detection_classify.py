from Preprocessing.preprocessing import Preprocessing
from Models.simple_mlp import SimpleMlp
from Models.complex_mlp import ComplexMlp
import numpy as np
import pandas as pd
import pdb

from keras import backend as K

class AnomalyDetectionClassify(object):
    def __init__(self, params, mongodb):
        self.__params = params
        self.__mongodb = mongodb

    # Decode the output variable predictions in to a readable format
    # 0 - Normal
    # 1 - Anomaly
    def decode_output(self, pred_list):
        decode_list = []

        for i in pred_list:
            if int(i[0]) == 0:
                decode_list.append(['Normal'])
            else:
                decode_list.append(['Anomaly'])

        return decode_list

    def classify(self):
        preproc = Preprocessing(self.__params)    
        if self.__mongodb == True: 
            classification_x, dataset = preproc.load_intrusion_detection_classification_data_from_mongodb(self.__params)
        else: 
            classification_x, dataset = preproc.load_intrusion_detection_classification_data_from_csv(self.__params)
        
        n_classify_batches = len(classification_x) // self.__params.test_batch_size                                                                                              
        print('\nClassification set size: ', (classification_x.shape[0], classification_x.shape[1]))
        print('Classification set number of batches: ', n_classify_batches)
        print('\n')
        
        deepnetwork = SimpleMlp.build(classification_x.shape[1], 1, False)
        deepnetwork.load_weights(self.__params.intrusion_detection_model_file)

        pred_classification_y = deepnetwork.predict_classes(classification_x, batch_size = self.__params.test_batch_size)
        
        K.clear_session()
        #pdb.set_trace()
        pred_classification_y = self.decode_output(pred_classification_y)

        # Get only first 6 columns of the list
        dataset = dataset[list(dataset.columns)[0:6]]
        output = np.hstack((dataset, pred_classification_y))

        pd_output = pd.DataFrame(output, columns = np.append(np.asarray(dataset.columns), 'prediction'))

        #pdb.set_trace()

        return pd_output