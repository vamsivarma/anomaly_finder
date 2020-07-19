from pymongo import MongoClient as mc
import numpy as np
import pdb

class MongoWrapper:

    db = ""
        
    def get_db_collections(self):
        return self.db.collection_names(include_system_collections=False)

    def get_config(self, config_name, params, localFlag):
        # Use Atlas only for getting configuration file
        # This is a tweak for demo
        # @TODO: Need to fix this
        connection = mc(params.connection_string)
        db = connection[params.mdbname]

        cObj = db[config_name]
        config = {}
        config = cObj.find_one()
        return config
          
    def get_dataset(self, dataset_name, subsetFlag):
        dObj = self.db[dataset_name]
        dataset = {}
        if subsetFlag == True:
            # Only for prediction, we use a subset of dataset
            dataset = dObj.find().limit(1000) #- Add this for unit testing
        else:
            dataset = dObj.find()

        columns =  list(dataset[0].keys())[1:]
        dataset_result = [list(d.values())[1:] for d in dataset]
 
        return [np.asarray(dataset_result), columns]

    def __init__(self, params, localFlag):
        if localFlag:
            # For unit testing
            connection = mc('localhost',27017)
        else:
            connection = mc(params.connection_string)
        self.db = connection[params.mdbname]