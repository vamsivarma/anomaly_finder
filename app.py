import os
from datetime import datetime
#from generate_data import word_to_array
from flask import Flask, jsonify, request, render_template, redirect, url_for, make_response
from flask import flash, session, abort


#from keras.models import load_model
#import tensorflow as tf

import mongo_wrapper as mdb
import charts_wrapper as cw
import eda_wrapper as edaw

import ml_wrapper as mlw

import traceback

db = {}
collection_list = [] 

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'tmp'

ALLOWED_EXTENSIONS = ['csv', 'xls']

# Wrapper for all the code realted to MongoDB
db_name = "ad2020" # For atlas
#db_name = "anomaly_detection" # For local machine
db = mdb.Mongo_Wrapper(db_name)

# Wrapper for all the charts
charts_obj = cw.Charts_Wrapper()

# Wrapper which contains code for doing Exploratory data analysis
eda_obj = edaw.EDA_Wrapper()

# ML wrapper for Experiments section
ml_obj = mlw.ML_Wrapper()


'''
models = {}
def load():
    
    models['CNN'] = load_model('models/modelCNN.hdf5')
    models['RNN'] = load_model('models/modelRNN.hdf5')
    models['FF'] = load_model('models/modelFF.hdf5')

    global graph
    graph = tf.get_default_graph()
'''



@app.route('/')
def landing():
    return render_template('login.html')
    #return render_template('index.html')

# Loading HTML templates for each section like Datasets, Analytics etc..,

@app.route('/nav')
def nav():
    return render_template('nav.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/datasets')
def datasets():
    return render_template('datasets.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/experiments')
def experiments():
    return render_template('experiments.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


# Routes for pages, Index and Landing page (Login form)
@app.route('/index')
def index():
    '''
    if not session.get('logged_in'):
        return landing()
    else:  
        return render_template('index.html')
    '''
    return render_template('index.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return landing()


@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        try:
            input_params = request.get_json(force=True)
            print(input_params['username'])
            print(input_params['password'])

            session['logged_in'] = True

            # Return the index file for now...
            # return redirect('/index')
            # return landing()
            return make_response(jsonify({"message": "File uploaded"}), 200)

        except Exception as e:
                print("type error: " + str(e))
                print(traceback.format_exc())

@app.route('/dsupload', methods=['POST'])
def dsupload():

    # print('Inside Upload API with request type: ' + request.method)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            print("File is valid")
            now = datetime.now()
            #filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f"), file.filename.rsplit('.', 1)[1]))
            #file.save(filename)
            
            try:
                # Connect to Mongo DB 
                #db_connect()
                db = mdb.Mongo_Wrapper(db_name)

                fname = file.filename.rsplit('.', 1)[0]
                print(fname)
                fext = file.filename.rsplit('.', 1)[1] 
                print(fext)
                
                # Save the file contents to Mongo collection based on the imported file name
                db.file_upload(file, fname, fext)

            except Exception as e:
                print("type error: " + str(e))
                print(traceback.format_exc())

            return make_response(jsonify({"message": "File uploaded"}), 200)


# Get the datasets list
# Used in Datasets, Explore and Analytics sections
@app.route('/datasets', methods=['POST'])
def get_datasets():

    if request.method == 'POST':

        try:    
            response_obj = {
                "total": 0,
                "rows": []
            }

            collection_list = db.get_db_collections()
            response_obj['total'] = len(collection_list)

            for c_name in collection_list:

                # Need to do this more efficiently
                # Removing the configuration related collections from the response
                if c_name not in ["configuration", "ad_settings"]:
                    #print(c_name)
                    response_obj['rows'].append(db.get_fields(c_name))
            
            #print(response_obj)
            return make_response(jsonify(response_obj), 200)

        except Exception as e:
                    print("type error: " + str(e))
                    print(traceback.format_exc())

# ML model prediction in Experiments page
# Commenting this for now
'''
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        print("Predict called")
        try:
            response_obj = {
                "fr": 0,
                "en": [],
                "es": 0
            }

            input_params = request.get_json(force=True)
            word = input_params['word']
            model_name = input_params['model']

            model = models[model_name]

            if model_name == 'FF':
                arr = word_to_array(word).reshape(1,26*12)
            else:
                arr = word_to_array(word).reshape(1,26,12)
            with graph.as_default():
                prediction = model.predict(arr)

            response_obj["fr"] = round(100*prediction[0, 0],2)
            response_obj["en"] = round(100*prediction[0, 1],2)
            response_obj["es"] = round(100*prediction[0, 2],2)
        
            return make_response(jsonify(response_obj), 200)

        except Exception as e:
                    print("type error: " + str(e))
                    print(traceback.format_exc())
'''

# Routes for settings page
# From Mongo DB
@app.route('/get_settings', methods=['POST'])
def get_settings():
    if request.method == 'POST':
        
        print("Get Settings called")
        try:
            response_obj = {
                "sconfig": {},
                "user_name": "",
                "user_id": 0
            }

            c_dict = {
                "user_name": "",
                "user_id": 0
            }

            input_params = request.get_json(force=True)
            c_dict["user_name"] = input_params['user_name']
            c_dict["user_id"] = input_params['user_id']

            sconfig = get_user_config(c_dict)

            if(len(sconfig)):
                response_obj['sconfig'] = sconfig[0]
                response_obj["user_name"] = input_params['user_name']
                response_obj["user_id"] = input_params['user_id']

            return make_response(jsonify(response_obj), 200)

        except Exception as e:
                    print("type error: " + str(e))
                    print(traceback.format_exc())

def get_user_config(c_dict):
    c_name = "ad_settings"
    config_obj = db.get_one(c_name, c_dict)
    return config_obj


# To Mongo DB
@app.route('/save_settings', methods=['POST'])
def save_settings():
    if request.method == 'POST':
        
        print("Save Settings called")
        try:
            response_obj = {
                "user_name": "",
                "user_id": 0
            }

            c_dict = {
                "user_name": "",
                "user_id": 0
            }

            input_params = request.get_json(force=True)
            c_dict["user_name"] = input_params['user_name']
            c_dict["user_id"] = input_params['user_id']
            sconfig = input_params['s_config']

            sconfig = save_user_config(c_dict, sconfig)

            if(len(sconfig)):
                response_obj["user_name"] = input_params['user_name']
                response_obj["user_id"] = input_params['user_id']

            return make_response(jsonify(response_obj), 200)

        except Exception as e:
                    print("type error: " + str(e))
                    print(traceback.format_exc())

def save_user_config(c_dict, sconfig):
    # Collection where setting are saved
    c_name = "ad_settings"
    config_obj = db.save_one(c_name, c_dict, sconfig)
    return config_obj

# Routes for Analytics page
@app.route('/get_analytics', methods=['POST'])
def get_analytics():
    input_params = request.get_json(force=True)

    # Charts to be rendered
    charts_list = input_params['charts_list']
    
    # Selected dataset
    c_name = input_params['d_name']

    # Selected target column
    label_col = input_params['label_col']

    # This variable holds the current dataset structure
    col_type_map = {}

    # Need to check if this is required 
    if(label_col == ''):
        # Label column is not set so get the dataset meta
        col_some_df = db.get_m_df(c_name)
        col_type_map = db.get_df_meta(col_some_df)

    #c_name = "regression_analysis" # Need to get this from interface
    c_df = db.get(c_name)

    # Get all the plots based on user selection
    graphsJSON = charts_obj.create_plot(charts_list, c_df, col_type_map, label_col)

    return make_response(jsonify(graphsJSON), 200)

# Routes for Explore page
@app.route('/explore_dataset', methods=['POST'])
def explore_dataset():
    input_params = request.get_json(force=True)

    # Selected dataset
    c_name = input_params['d_name']

    c_df = db.get(c_name)
    ds_summary = eda_obj.get_ds_summary(c_df)

    return make_response(jsonify(ds_summary), 200)

# Routes for Experiments page
@app.route('/launch_experiment', methods=['POST'])
def launch_experiment():
    input_params = request.get_json(force=True)

    # Experiments to be rendered
    e_list = input_params['e_list']
    
    # Selected dataset
    c_name = input_params['d_name']

    # Selected target column
    label_col = input_params['label_col']

    # Data split
    d_split = input_params['d_split']

    #c_name = "regression_analysis" # Need to get this from interface
    c_df = db.get(c_name)

    # Get all the experiments based on user selection (all or individual)
    expJSON = ml_obj.get_experiments(e_list, c_df, label_col, d_split)

    return make_response(jsonify(expJSON), 200)


if __name__ == '__main__':

    app.secret_key = 'thisisasupersecret007'

    #print('Loading model...')
    #load()

    #get_user_config()

    app.run(debug=False)
