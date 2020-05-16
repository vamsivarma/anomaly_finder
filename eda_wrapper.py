import pandas as pd
import numpy as np

'''
# For unit testing of module
import mongo_wrapper as mdb
#db_name = "ad2020"
db_name = "anomaly_detection"
db = mdb.Mongo_Wrapper(db_name)
'''

class EDA_Wrapper():

    d_type_map = {
            "int32": "Integer",
            "int64": "Integer",
            "object": "Categorical",
            "float32": "Float",
            "float64": "Float",
            "bool": "Boolean"
    }

    def get_ds_summary(self, ds_df):

        ds_summary = {
            "numeric": "",
            "categorical": "",
            "properties": ""
        }
        
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        num_desc_df = ds_df.describe(include = [np.number])
        num_desc_df = num_desc_df.T
        cat_desc_df = ds_df.describe(include = ['O'])
        cat_desc_df = cat_desc_df.T

        prop_df = self.get_properties(ds_df)

        ds_summary['numeric'] = num_desc_df.to_html()
        ds_summary['categorical'] = cat_desc_df.to_html()
        ds_summary['properties'] = prop_df.to_html(index=False)

        return ds_summary

    def get_properties(self, ds_df):

        field_list = []
        field_type_list = []
        value_list = []

        for i in range(len(ds_df.columns)):

            field_key = ds_df.columns[i]
            field_type = self.d_type_map[str(ds_df[ds_df.columns[i]].dtype)]
            field_na_value = ds_df[ds_df.columns[i]].isna().sum()
            
            field_list.append(field_key)
            field_type_list.append(field_type)
            value_list.append(field_na_value)
        
        data = {
            'Name': field_list,
            'Type': field_type_list,
            'NA count': value_list
        }

        prop_df = pd.DataFrame( data, columns = ['Name','Type','NA count'] )

        return prop_df



# For unit testing of the module
def get_dataset():
    c_name = "ID_train"
    c_data = db.get(c_name)
    return c_data

'''
eda_w = EDA_Wrapper()
#print(eda_w.get_ds_summary(get_dataset()))
print(eda_w.get_properties(get_dataset()))

#df = get_dataset()

#print(type(df.info()))
'''