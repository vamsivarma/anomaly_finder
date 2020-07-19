'''
Created on 01/01/2020
Modified on 10/02/2020

@author: Francesco Pugliese, Vamsi Gunturi
'''
import os

from Misc.utils import Utilities                                                # Utilities Class
from Initialization.init import Init                                            # Initialization Class
from Settings.settings import SetParameters                                     # Settings 
from Training.anomaly_detection_train import AnomalyDetectionTraining           # Training Class
from Classification.anomaly_detection_classify import AnomalyDetectionClassify  # Classification Class                            

# Other Imports
import platform

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# Globals
times = []
default_config_file = "anomaly_detection.ini"                                                # Default Configuration File
gpu_desc = ""

# Operating System
OS = platform.system()                                                               # returns 'Windows', 'Linux', etc

#if OS == "Linux": 
## Configuration
# Command Line Parser

parser = Init.parser()                                                               # Define arguments parser

(arg1) = parser.parse_args()
config_file = arg1.conf

if arg1.mode is None: 
    ad_mode = None
else: 
    ad_mode = int(arg1.mode)

if arg1.mdb is False: 
    if config_file is None: 
        config_file = default_config_file                                                # Default Configuration File
        
    # Read the Configuration from File
    set_parameters = SetParameters("../Conf", config_file, OS) 
    params = set_parameters.read_config_file()
else: 
    # Read the Configuration from MongoDB
    set_parameters = SetParameters(None, None, OS) 
    params = set_parameters.read_config_mongodb()
    
# Overwrite configuration file
if ad_mode is not None: 
    params.ad_mode = ad_mode
    
if params.gpu==True:
    # GPUs
    gpu_desc = Utilities.get_available_gpus()                                            # Returns a list of available gpus

# Overwrite configuration file
if ad_mode is not None: 
    params.ad_mode = ad_mode

# Set the CPU or GPU
if gpu_desc == "" or gpu_desc == None:                                        # if gpu not available use CPU automatically
    params.gpu = False
Utilities.set_cpu_or_gpu(params)

'''
    times_prefix = "training_"+parameters.system_description.lower()+'_'+'_'.join(list(gpu_desc.lower().split()))+"_"+OS.lower()+"_"
    times_title_description = "Execution Times of Training \n"+"   System: "+parameters.system_description+"   GPU: "+gpu_desc+"   O.S.: "+OS
    model_prefix = '_'.join(list(gpu_desc.lower().split()))+"_"+OS.lower()+"_"
'''


# Select the action based on the chosen land cover mode 
if params.ad_mode == 0:                                   # Training
    ad_training = AnomalyDetectionTraining(params, arg1.mdb) 
    ad_training.train()
elif params.ad_mode == 1:                                 # Classification
    print(None)
    ad_testing = AnomalyDetectionClassify(params, arg1.mdb)
    ad_testing.classify()