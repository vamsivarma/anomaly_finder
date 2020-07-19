'''
Created on 01/01/2020
Modified on 10/02/2020

@author: Francesco Pugliese, Vamsi Gunturi
'''
import os
from tensorflow.python.client import device_lib

# Utilities Class
class Utilities:

    @staticmethod
    # Returns a list of available gpus
    def get_available_gpus():
        try:
            local_device_protos = device_lib.list_local_devices()
            return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU'][0].split(':')[2].split(',')[0].strip()
        except(IndexError):
            return None

    @staticmethod
    def set_cpu_or_gpu(parameters):
        # Set CPU or GPU type
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        if parameters.gpu == False: 
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else: 
            os.environ["CUDA_VISIBLE_DEVICES"] = parameters.gpu_id            
