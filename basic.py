
import os
from datetime import datetime
from flask import Flask, jsonify, request, render_template, redirect, url_for, make_response
from flask import flash, session, abort

import mongo_wrapper as mdb
import traceback

db = {}
collection_list = [] 

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['csv', 'xls', ]

db_name = "anomaly_detection"
db = mdb.Mongo_Wrapper(db_name)


@app.route('/')
def landing():
    return render_template('login.html')
    #return render_template('index.html')

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


'''
@app.route('/dsupload', methods=['POST'])
def dsupload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f"), file.filename.rsplit('.', 1)[1]))
            file.save(filename)
            return jsonify({"success":True})
'''

@app.route('/dsupload', methods=['POST'])
def dsupload():

    # print('Inside Upload API with request type: ' + request.method)
    if request.method == 'POST':
        file = request.files['file']
        if file:
            print("File is valid")

            now = datetime.now()
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f"), file.filename.rsplit('.', 1)[1]))
            file.save(filename)
            
            try:
                # Connect to Mongo DB 
                #db_connect()
                db = mdb.Mongo_Wrapper(db_name)

                print(filename)
                print(file.filename.rsplit('.', 1)[0])
                
                # Save the file contents to Mongo collection based on the imported file name
                db.file_upload(filename, file.filename.rsplit('.', 1)[0])

            except Exception as e:
                print("type error: " + str(e))
                print(traceback.format_exc())

            return make_response(jsonify({"message": "File uploaded"}), 200)

'''
@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return "Hello Boss!  <a href="/logout">Logout</a>"

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    
    return home()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()
'''

'''
@app.route('/mupload', methods=['POST'])
def mupload():
    # print('Inside Upload API with request type: ' + request.method)
    if request.method == 'POST':
        input_params = request.get_json(force=True)
        file_name = input_params['file_name']
        input_name = input_params['input_name']

        if file:
            #print("File is valid")
            db_connect()
            db.file_upload(file_name, input_name)
            
            return make_response(jsonify({"message": "Data imported to Mongo"}), 200)
'''

@app.route('/datasets', methods=['GET'])
def get_datasets():

    response_obj = {
        "total": 0,
        "rows": []
    }

    collection_list = db.get_db_collections()
    response_obj['total'] = len(collection_list)

    for c_name in collection_list:
        #print(c_name)
        response_obj['rows'].append(db.get_fields(c_name))
    
    #print(response_obj)
    return make_response(jsonify(response_obj), 200)


if __name__ == '__main__':

    app.secret_key = 'thisisasupersecret007'

    #get_datasets()
    app.run(debug=False)
