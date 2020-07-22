from Misc.utils import Utilities                                                # Utilities Class
from Initialization.init import Init                                            # Initialization Class
from Settings.settings import SetParameters                                     # Settings 
from Training.anomaly_detection_train import AnomalyDetectionTraining           # Training
from Classification.anomaly_detection_classify import AnomalyDetectionClassify  # Classification Class

import platform

# Globals
times = []
# Default Configuration File
config_file = "anomaly_detection.ini"
gpu_desc = ""

# Operating System
# returns 'Windows', 'Linux', etc
OS = platform.system()                                                               
ad_mode = None

set_parameters = SetParameters(None, None, OS) 
params = set_parameters.read_config_mongodb()
mdb = True

if params.gpu==True:
    # GPUs
    gpu_desc = Utilities.get_available_gpus()                                            # Returns a list of available gpus

# Set the CPU or GPU
if gpu_desc == "" or gpu_desc == None:                                        # if gpu not available use CPU automatically
    params.gpu = False
Utilities.set_cpu_or_gpu(params)

class AD_Run:

    def start(self, trainFlag):
        results = ''
        metric_df = ''

        # Read the Configuration from MongoDB every time
        # to read the latest settings
        set_parameters = SetParameters(None, None, OS) 
        params = set_parameters.read_config_mongodb()

        # Select the action based on the chosen land cover mode 
        if trainFlag:                                   
            # Training
            ad_training = AnomalyDetectionTraining(params, mdb) 
            results,metric_df = ad_training.train()
            # prints - ['epoch', 'acc', 'loss', 'val_acc', 'val_loss']    
            # print(results.columns)
        else:
            # Predict
            ad_testing = AnomalyDetectionClassify(params, mdb)
            results = ad_testing.classify() 

            # Print the prediction variable 
            #print(results['class'])
        
        return [results, metric_df]

'''
# This is for unit testing
ad_run_obj = AD_Run()
ad_run_obj.start(True) # For training
#ad_run_obj.start(False) # For prediction
'''