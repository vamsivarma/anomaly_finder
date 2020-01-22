from pymongo import MongoClient
import csv
import pandas as pd
import pymongo
import json
import os
from hurry.filesize import size

class Mongo_Wrapper:

    db = ""
        
    def save(self, cName, data): 
        cObj = self.db[cName]
        
        #@TODO: If there are more records then insert 1000 records at a time
        cObj.insert_many(data)  
    
    def save_one(self, cName, data): 
        cObj = self.db[cName]
        
        cObj.insert_one(data)
    
    def get(self, cName):
        
        data = list(self.db[cName].find({}))
        result = []
        
        for d in data:
            result.append(d)
        
        return result
    
    def get_db_collections(self):
        return self.db.collection_names(include_system_collections=False)


    def file_upload(self, uploaded_file, c_name):

        #collection_name = 'collection_name'  
        # Replace mongo db collection name

        # Create collection beforehand
        #self.db.createCollection("sample_collection")

        db_sc = self.db[c_name]

        cdir = os.path.dirname(__file__)
        file_res = os.path.join(cdir, uploaded_file)

        filename, file_extension = os.path.splitext(uploaded_file)
        if(file_extension == ".csv"):
            data = pd.read_csv(file_res) 
        else:
            data = pd.read_excel(file_res)

        data_json = json.loads(data.to_json(orient='records'))
        db_sc.remove()
        db_sc.insert(data_json)

    def get_fields(self, c_name):

        collection_obj = {
            'row_count': 0,
            'columns': [],
            'col_count': 0,
            'size': 0,
            'name': c_name
        }

        colstats = self.db.command("collstats", c_name)
        collection_obj['columns'] = list(self.db[c_name].find_one().keys())
        # Since we dont need to count _id column
        collection_obj['col_count'] = len(collection_obj['columns']) - 1
        collection_obj['size'] = size(colstats['size'])
        collection_obj['row_count'] = colstats['count']

        return collection_obj
          
    def __init__(self, dbName):
        #connection = MongoClient('localhost',27017)
        connection = pymongo.MongoClient("mongodb://adadmin:adadmin@adcluster-shard-00-00-zmesm.mongodb.net:27017,adcluster-shard-00-01-zmesm.mongodb.net:27017,adcluster-shard-00-02-zmesm.mongodb.net:27017/test?ssl=true&replicaSet=adcluster-shard-0&authSource=admin&retryWrites=true&w=majority")


        #connection = pymongo.MongoClient("mongodb+srv://adadmin:adadmin@adcluster-zmesm.mongodb.net/test?retryWrites=true&w=majority")

        self.db = connection[dbName]