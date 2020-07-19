import mongo_wrapper as mdb

import json
import pandas
from pandas import json_normalize


db_name = "ad2020"
db = mdb.Mongo_Wrapper(db_name)


def get_dataset():
    c_name = "GRUPPI_FRIGO_00044"
    c_data = db.get(c_name)
    return c_data


def df_heatmap(df):
    
    corr_df = df.corr()
    corr_df = list(corr_df.values)

    for i in range(0, len(corr_df)):
        corr_df[i] = list(corr_df[i])

    print(corr_df)
    print(list(df.columns))

def df_missing_values(df):
    for i in range(len(df.columns)):
        print(df.columns[i] + ": " + str(df[df.columns[i]].isna().sum()))

def get_col_types(cName):

    d_type_map = {
        "int64": "Integer",
        "object": "Categorical",
        "float64": "Float",
        "bool": "Boolean"
    }

    field_list = []
    field_type_list = []

    col_type_map = {}
    
    result = db.m_find_some(cName)

    df = json_normalize(result)

    # Since we dont need key column for our analysis 
    # We drop it
    df.drop(columns = ['_id'], inplace = True)

    for i in range(len(df.columns)):
        field_key = df.columns[i]
        field_type = d_type_map[str(df[df.columns[i]].dtype)]
        col_type_map[field_key] = field_type

    return col_type_map


#df = get_dataset()

#df_missing_values(df)

#df_heatmap(df)

#print(db.get_fields("titanic"))

#print(get_col_types("titanic"))

col_some_df = db.get_m_df("titanic")
col_type_map = db.get_df_meta(col_some_df)

print(col_type_map)