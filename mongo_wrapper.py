from pymongo import MongoClient as mc
import csv
import pandas as pd
#import pymongo
import json
import os

from pandas import json_normalize

class Mongo_Wrapper:

    db = ""
    dbName = ""

    # Need to maintain a common map for this
    # This is also present in chart_wrapper
    d_type_map = {
        "int64": "Integer",
        "object": "Categorical",
        "float64": "Float",
        "bool": "Boolean"
    }
        
    def save(self, cName, data): 
        cObj = self.db[cName]
        
        #@TODO: If there are more records then insert 1000 records at a time
        cObj.insert_many(data)  
    
    def save_one(self, cName, qObj, sObj, localFlag): 

        cObj = {}

        # Only for configurations we use remote DB
        # Only added for demo
        # @TODO: need to remove this
        if localFlag:
            db = self.get_remote_db()
            cObj = db[cName]
        else:
            cObj = self.db[cName]
        
        newvalues = { "$set": sObj }

        # Update the new settings
        cObj.update_one(qObj, newvalues)

        # Once update get the new settings from db
        config_obj = self.get_one(cName, qObj, localFlag)

        return config_obj
        

    def get_one(self, cName, qObj, localFlag):

        # Only for configurations we use remote DB
        # Only added for demo
        # @TODO: need to remove this
        # db = self.get_remote_db()

        data = {}

        if localFlag:
            db = self.get_remote_db()
            data = db[cName].find(qObj)
        else:
           data = self.db[cName].find(qObj)

        result = []
        
        doc_dict = {}

        for d in data:
            doc_dict = {}
            for k in d:
                # Removing the object type keys to eliminate the issue with JSON encoding
                if k not in ["_id"]:
                     doc_dict[k] = d[k]

            result.append(doc_dict)
        
        return result
    
    def get(self, cName):
        
        # This(10000) is only for initial testing
        # @TODO: Need to remove this after integration of charts
        data = list(self.db[cName].find({}).limit(50000)) #.limit(1000))

        #print(len(data))
        result = []
        
        for d in data:
            result.append(d)
        
        df = json_normalize(result)

        # Since we dont need key column for our analysis 
        # We drop it
        df.drop(columns = ['_id'], inplace = True)

        return df
    
    def get_db_collections(self):
        return self.db.collection_names(include_system_collections=False)


    def file_upload(self, uploaded_file, fname, fext):

        #collection_name = 'collection_name'  
        # Replace mongo db collection name

        # Create collection beforehand
        #self.db.createCollection("sample_collection")

        db_sc = self.db[fname]

        #cdir = os.path.dirname(__file__)
        #file_res = os.path.join(cdir, uploaded_file)

        #filename, file_extension = os.path.splitext(uploaded_file)
        if(fext == "csv"):
            data = pd.read_csv(uploaded_file) 
        else:
            data = pd.read_excel(uploaded_file)

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
        single_row = self.db[c_name].find_one()
        col_list = list(single_row.keys())

        # Remove _id columnn from column list
        col_list = col_list[1:]

        col_some_df = self.get_m_df(c_name)
        col_type_map = self.get_df_meta(col_some_df)

        collection_obj['col_type_map'] = col_type_map

        collection_obj['columns'] = col_list

        #print(collection_obj['columns'])

        collection_obj['col_count'] = len(collection_obj['columns'])
        collection_obj['size'] = colstats['size']
        collection_obj['row_count'] = colstats['count']

        return collection_obj

    def get_df_meta(self, df):
        
        col_type_map = {}
    
        # Since we dont need key column for our analysis 
        # We drop it
        df.drop(columns = ['_id'], inplace = True)

        for i in range(len(df.columns)):
            field_key = df.columns[i]
            field_type = self.d_type_map[str(df[df.columns[i]].dtype)]
            col_type_map[field_key] = field_type

        return col_type_map


    def get_m_df(self, cName):

        # Need to properly check if this is accurate
        data = list(self.db[cName].find({}).limit(100))
        
        result = []
        
        for d in data:
            result.append(d)

        df = json_normalize(result)

        return df

    # Only for demo
    # Need to remove this later
    def get_remote_db(self):
        connection = mc("mongodb://adadmin:adadmin@ad2020-shard-00-00-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-01-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-02-zmesm.gcp.mongodb.net:27017/test?ssl=true&replicaSet=ad2020-shard-0&authSource=admin&retryWrites=true&w=majority")

        db = connection[self.dbName]

        return db
          
    def __init__(self, dbName, localFlag):

        if localFlag:
            connection = mc('localhost',27017)
        else:
            connection = mc("mongodb://adadmin:adadmin@ad2020-shard-00-00-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-01-zmesm.gcp.mongodb.net:27017,ad2020-shard-00-02-zmesm.gcp.mongodb.net:27017/test?ssl=true&replicaSet=ad2020-shard-0&authSource=admin&retryWrites=true&w=majority")

        self.dbName = dbName

        self.db = connection[dbName]