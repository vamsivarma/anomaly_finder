'''
Created on 12/05/2020

@author: Francesco Pugliese, Vamsi Gunturi
'''

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

class SimpleMlp:
    @staticmethod
    def build(input_dim, classes, summary, weightsPath=None):
        model = Sequential()
        
        model.add(Dense(32, activation='relu', input_dim=input_dim))

        if classes > 2:
            model.add(Dense(classes, activation='softmax'))
        else: 
            model.add(Dense(1, activation='sigmoid'))
        
        if summary==True:
            model.summary()
        
		#if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model