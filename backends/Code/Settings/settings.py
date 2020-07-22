'''
Created on 01/01/2020
Modified on 10/02/2020

@author: Francesco Pugliese
'''

import configparser
from Misc.mongo_wrapper import MongoWrapper as mdb 

import pdb

config = {}

mongoFlag = False

class SetParameters:
    
    def __init__(self, conf_file_path, conf_file_name, OS):
        
        self.random_seed = 77
        
        # Class Initialization (constructor) 
        self.conf_file_path = conf_file_path
        self.conf_file_name = conf_file_name
        
        # System
        self.ad_mode = 0                                                     # 0: training, 1: testing
        self.gpu = False
        self.gpu_id = '0'
        self.system_desc = 'Laptop'
        self.gpu_desc = None
        
        # MongoDB
        self.connection_string = "mongodb://adadmin:adadmin@ad2020-shard-00-00-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-01-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-02-zmesm.gcp.mongodb.net:27017/test?ssl=true&replicaSet=ad2020-shard-0&authSource=admin&retryWrites=true&w=majority"
        self.mdbname = "ad2020"       

        # Dataset
        self.geox_data = False
        self.intrusion_detection_data = False
        self.geox_dataset_path = ""
        self.intrusion_detection_path = ""
        self.geox_filename = ""
        self.intrusion_detection_train_filename = ""
        self.intrusion_detection_classification_filename = ""
        self.geox_field_separator = ";"
        self.intrusion_detection_field_separator = ","
        self.mnist_benchmark = False
        
        # Preprocessing
        self.valid_set_perc = 3                                          # Validation set percentage with respect to the Training Set
        self.test_set_perc = 3                                           # Test set percentage with respect to the entire Data Set
        self.normalize_x = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        self.normalize_y = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        self.limit = None
        
        # Model
        self.neural_model = 'mlp'
        self.intrusion_detection_models_path = ''
        self.intrusion_detection_model_file = ''
        self.summary = True
  
        # Training
        self.train_tag = ""
        self.epochs_number = 2
        self.learning_rate = 0.0001                                      # best learning rate = 0.0015 at moment, 0.001 on mcover
        self.train_batch_size = 32
        self.training_algorithm = 'sgd'
        self.early_stopping = True
        self.save_best_model = True
        self.csv_logger = True
        self.log_path = ""
        self.log_file = ""
        
        # Testing
        self.model_testing_file = ''
        self.test_batch_size = 64

        # Output
        self.output_path = '../Output'
        self.log_path = '../Log'
        self.save_tiles = False
        self.charts_path = '../Output/Charts' 
        self.csv_path = '../Output/Csv'
        self.pause_time = 5
        
        # Others
        self.OS = OS
        
        # Global Constants
        # Alert Sound
        self.sound_freq = 1000                                           # Set Frequency in Hertz
        self.sound_dur = 3000                                            # Set Duration in ms, 1000 ms == 1 second
        self.times_header = ["Preprocessing Time", "Postprocessing Time", "Gpu Time", "Overall Time without Gpu", "Overall Time"]
        self.model_prefix = ''
        if conf_file_name is None:
            global config 
            
            # True for Local Mongo DB
            # False for remote instance
            db = mdb(self, mongoFlag) 
            config = db.get_config('ad_settings', self, mongoFlag)     # We read all the configuration of the backend from MongoDB       
        
    def read_config_file(self):
        
        # Read the Configuration File
        config = configparser.ConfigParser()
        config.read(self.conf_file_path+'/'+self.conf_file_name)
        config.sections()

        # System
        self.ad_mode = config.getint('System', 'ad_mode')                                                     # 0: classify, 1: preprocess, 2: training, 3: testing
        self.gpu = config.getboolean('System', 'gpu')
        self.gpu_id = config.get('System','gpu_id')
        self.system_desc = config.get('System', 'system_desc')
        self.gpu_desc = config.get('System','gpu_desc')
        if self.gpu_desc == 'None': 
            self.gpu_desc = None

        # MongoDB
        self.connection_string = config.get('MongoDB', 'connection_string')
        self.db = config.get('MongoDB', 'mdbname')

        # Dataset
        self.geox_data = config.getboolean('Dataset', 'geox_data')
        self.intrusion_detection_data = config.getboolean('Dataset', 'intrusion_detection_data')

        if self.OS == "Linux":
            self.geox_dataset_path = config.get('Dataset', 'geox_dataset_path_linux')
            self.intrusion_detection_path = config.get('Dataset', 'intrusion_detection_path_linux')
        elif self.OS == "Windows": 
            self.geox_dataset_path = config.get('Dataset', 'geox_dataset_path_win')
            self.intrusion_detection_path = config.get('Dataset', 'intrusion_detection_path_win')
        
        self.geox_filename = config.get('Dataset', 'geox_filename')
        self.intrusion_detection_train_filename = config.get('Dataset', 'intrusion_detection_train_filename')
        self.intrusion_detection_classification_filename = config.get('Dataset', 'intrusion_detection_classification_filename')
        self.geox_field_separator = config.get('Dataset', 'geox_field_separator')
        self.intrusion_detection_field_separator = config.get('Dataset', 'intrusion_detection_field_separator')
        self.mnist_benchmark = config.getboolean('Dataset', 'mnist_benchmark')
              
        # Preprocessing
        self.valid_set_perc = config.getint('Preprocessing', 'valid_set_perc') / 100                                     
        self.test_set_perc = config.getint('Preprocessing', 'test_set_perc') / 100
        self.normalize_x = config.getboolean('Preprocessing', 'normalize_x')
        self.normalize_y = config.getboolean('Preprocessing', 'normalize_y')
        self.limit = config.get('Preprocessing', 'limit')
        try: 
            self.limit = int(self.limit)
        except ValueError: 
            self.limit = None

        # Model
        self.neural_model = config.get('Model', 'neural_model')
        if self.OS == "Linux":
            self.intrusion_detection_models_path = config.get('Model', 'intrusion_detection_models_path_linux')
        elif self.OS == "Windows": 
            self.intrusion_detection_models_path = config.get('Model', 'intrusion_detection_models_path_win')
        self.intrusion_detection_model_file = config.get('Model', 'intrusion_detection_model_file')
        self.summary = config.getboolean('Model', 'summary')

        # Training
        self.epochs_number = config.getint('Training', 'epochs_number')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.train_batch_size = config.getint('Training', 'train_batch_size')
        self.training_algorithm = config.get('Training', 'training_algorithm')
        self.early_stopping = config.getboolean('Training', 'early_stopping')
        self.save_best_model = config.getboolean('Training', 'save_best_model')
        self.csv_logger = config.getboolean('Training', 'csv_logger')
        self.log_path = config.get('Training', 'log_path')
        self.log_file = config.get('Training', 'log_file')

        # Testing
        self.model_testing_file = config.get('Testing', 'model_testing_file')
        self.test_batch_size = config.getint('Testing', 'test_batch_size')

        # Output
        self.output_path = config.get('Output', 'output_path')
        self.log_path = config.get('Output', 'log_path')
        self.save_tiles = config.getboolean('Output', 'save_tiles')
        self.charts_path = config.get('Output', 'charts_path') 
        self.csv_path = config.get('Output', 'csv_path')
        self.pause_time = config.getint('Output', 'pause_time')
		
        # Global Constants
        self.model_prefix = self.gpu_desc.lower()+'_'+self.OS.lower()+'_'+str(self.epochs_number)+'_epochs_model_'+self.neural_model.lower()+'_'
        self.train_output_header = ' '.join([x.capitalize() for x in self.model_prefix.split('_')])
        self.test_output_header = ' '.join([x.capitalize() for x in self.model_testing_file.split('.')[0].replace('linux', self.OS).replace('windows', self.OS).split('_')])
        return self		

    def read_config_mongodb(self):
        global config

        self.random_seed = int(config['random_seed'])
        self.ad_mode = config['ad_mode']

        # Dataset
        geox_data = config['geox_data']
        intrusion_detection_data = config['intrusion_detection_data']
        mnist_benchmark = config['mnist_benchmark']
        
        # Preprocessing
        self.valid_set_perc = config['valid_set_perc'] / 100                                        # Validation set percentage with respect to the Training Set
        self.test_set_perc = config['test_set_perc'] / 100                                          # Test set percentage with respect to the entire Data Set
        self.normalize_x = config['normalize_x']
        self.normalize_y = config['normalize_y']
        self.limit = config['limit']
        try: 
            self.limit = int(self.limit)
        except ValueError: 
            self.limit = None

        # Model 
        self.neural_model = config['neural_model']
        if self.OS == "Linux":
            self.intrusion_detection_models_path = config['intrusion_detection_models_path_linux']
        elif self.OS == "Windows": 
            self.intrusion_detection_models_path = config['intrusion_detection_models_path_win']
        self.intrusion_detection_model_file = config['intrusion_detection_model_file']
        self.summary = config['summary']

        # Training        
        self.epochs_number = config['epochs_number']
        self.learning_rate = config['learning_rate']
        self.train_batch_size = config['train_batch_size']
        self.training_algorithm = config['training_algorithm']
        early_stopping = config['early_stopping']
        self.save_best_model = config['save_best_model']
        self.csv_logger = config['csv_logger']
        self.log_path = config['log_path']
        self.log_file = config['log_file']
       
        # Testing
        self.model_testing_file = config['model_testing_file']
        self.test_batch_size = config['test_batch_size']  
            
        return self