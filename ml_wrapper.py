
import pandas as pd
import numpy as np

import sklearn
from numpy.core.umath_tests import inner1d

# SCALING NUMERICAL ATTRIBUTES
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#from sklearn.model_selection import StratifiedShuffleSplit

# Label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# Combining multiple ML algorithms
from sklearn.pipeline import make_pipeline

# For accuracy score
from sklearn.metrics import accuracy_score

# For splitting input data in to train and test dataset
from sklearn.model_selection import train_test_split

# For importing methods related to preprocessinng
from sklearn import preprocessing

# Loading Machine Learning Algorithms
from sklearn import linear_model

# GB
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100,max_depth=5)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr_std_pca = make_pipeline(scaler, pca, lr)

# SVM
#from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVC
svc = SVC(kernel='linear')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

from sklearn.linear_model import Perceptron

# SGD
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()

# DT
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

'''
# XGBoost
import xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
'''

# Classification metrics - Start

# Accuracy score
from sklearn.metrics import accuracy_score
# Example usage: accuracy_score(y_test, y_pred)

# Classification report
from sklearn.metrics import classification_report
# Example usage: print(classification_report(y_test, y_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
# Example usage: print(confusion_matrix(y_test, y_pred))

# Precision and Recall
from sklearn.metrics import precision_score, recall_score
# Example usage: print("Precision:", precision_score(Y_train, Y_pred))
# Example usage: print("Recall:",recall_score(Y_train, Y_pred))

# F1-score
from sklearn.metrics import f1_score
# Example usage: f1_score(Y_train, Y_pred)

# ROC curve
from sklearn.metrics import roc_auc_score
# Example usage: roc_auc_score(Y_train, Y_pred)

def one_hot_encode_feature_df(df, cat_vars=None, num_vars=None):
    '''performs one-hot encoding on all categorical variables and combines result with continous variables'''
    cat_df = pd.get_dummies(df[cat_vars])
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)#,ignore_index=False)


def train_model(model, feature_df, target_df, num_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feature_df, target_df, cv=2, n_jobs=num_procs, scoring='neg_mean_squared_error')
    mean_mse[model] = -1.0*np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)

def print_summary(model, mean_mse, cv_std):
    print('\nModel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during CV:\n', cv_std[model])

def save_results(model, mean_mse, predictions, feature_importances):
    '''saves model, model summary, feature importances, and predictions'''
    with open('model.txt', 'w') as file:
        file.write(str(model))
    feature_importances.to_csv('feature_importances.csv') 
    np.savetxt('predictions.csv', predictions, delimiter=',')

def train_test_split_func():
    X = titanic.drop('survived', axis=1)
    y = titanic['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train)

def data_preprocessing_common():
    # Standardization
    # Normalization
    # Binarization
    # Encoding Categorical Features
    # Imputing missing values

    pass

def get_feature_importance(X_train, X_test, y_train, y_test):
        
        # FEATURE SELECTION
        # By default, fit random forest classifier on the training set
        # need to configure this from outside

        # For train set
        rfc.fit(X_train, y_train)
        # extract important features
        train_score = np.round(rfc.feature_importances_,3)

        #rfc = RandomForestClassifier()

        # For test set
        rfc.fit(X_test, y_test)
        # extract important features
        test_score = np.round(rfc.feature_importances_,3)

        importances = pd.DataFrame({'Feature':X_train.columns,'Trainset score': train_score, 'Testset score': test_score})
        importances = importances.sort_values('Testset score',ascending=False) #.set_index('feature')

        return importances.to_html(index=False)

# @TODO: Need to remove this if not used
def cr_todf(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = ' '.join(line.split())
        row_data = row_data.split(' ')
        #row_data = line.split('      ')
        
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    report_df = pd.DataFrame.from_dict(report_data)
    #report_df = report_df(index = False)
    #dataframe.to_csv('classification_report.csv', index = False)
    
    return report_df

def get_metrics(X_train, X_test, y_train, y_test):

    #initialize model list and dicts
    models = []
    mean_mse = {}
    cv_std = {}
    res = {}

    metrics_map = {}

    #define number of processes to run in parallel
    num_procs = 2

    #shared model paramaters
    verbose_lvl = 5

    #create models -- hyperparameter tuning already done by hand for each model
    # @TODO: Add Pipeline of PCA and LR
    models.extend([lr, rfc, sgd, dt, knn, gnb, gbc])
    alg_list = ["Logistic regression", "Random Forest", "SGD", "Decision tree", "KNN", 
                "Gaussian Naive Bayes", "Gradient Boosting"]

    t_cols_list = ['Algoritm', 'Train Accuracy', 'Test Accuracy', 
                    'Train Precision', 'Test Precision', 
                    'Train Recall', 'Test Recall',
                    'Train F1 Score', 'Test F1 Score',
                    'Train ROC', 'Test ROC'
    ]

    metric_values = []

    index = 0
    # Find accuracy for every model
    for model in models:

        # For train set
        # Fitting
        # X_train, X_test, y_train, y_test
        # X_train - df_x
        # y_train - df_y

        # df_pred - y_train_pred

        model.fit(X_train, y_train)
        # Prediction
        y_train_pred = model.predict(X_train)

        # Encode categorical variable
        y_train_np = pd.get_dummies(y_train)
        y_train_pred_np = pd.get_dummies(y_train_pred)
        
        y_train_np = np.array(y_train_np)
        y_train_pred_np = np.array(y_train_pred_np)


        # For test set
        # Fitting
        # X_train, X_test, y_train, y_test
        # X_train - df_x
        # y_train - df_y

        # df_pred - y_train_pred

        model.fit(X_test, y_test)
        # Prediction
        y_test_pred = model.predict(X_test)

        # Encode categorical variable
        y_test_np = pd.get_dummies(y_test)
        y_test_pred_np = pd.get_dummies(y_test_pred)
        
        y_test_np = np.array(y_test_np)
        y_test_pred_np = np.array(y_test_pred_np)

        '''
        metrics_dict = {
            'accuracy': '',
            'precision': '',
            'recall': '',
            'f1 score': '',
            'roc': ''
        }
        '''

        cur_metric_values = [alg_list[index]]

        # X_train - df_x
        # y_train - df_y

        # df_pred - y_train_pred

        # accuracy
        acc_model_train = accuracy_score(y_train,y_train_pred)
        acc_model_test = accuracy_score(y_test,y_test_pred)
        #metrics_dict['accuracy'] = str(acc_model)
        cur_metric_values.append(acc_model_train)
        cur_metric_values.append(acc_model_test)

        # precision
        p_score_train = precision_score(y_train_np, y_train_pred_np, average='weighted')
        p_score_test = precision_score(y_test_np, y_test_pred_np, average='weighted')
        #metrics_dict['precision'] = str(p_score)
        cur_metric_values.append(p_score_train)
        cur_metric_values.append(p_score_test)

        # recall
        r_score_train = recall_score(y_train_np, y_train_pred_np, average='weighted')
        r_score_test = recall_score(y_test_np, y_test_pred_np, average='weighted')
        #metrics_dict['recall'] = str(r_score)
        cur_metric_values.append(r_score_train)
        cur_metric_values.append(r_score_test)

        # F1 score
        f1m_score_train = f1_score(y_train_np, y_train_pred_np, average='weighted')
        f1m_score_test = f1_score(y_test_np, y_test_pred_np, average='weighted')
        #metrics_dict['f1 score'] = str(f1m_score)
        cur_metric_values.append(f1m_score_train)
        cur_metric_values.append(f1m_score_test)

        # ROC score
        roc_score_train = roc_auc_score(y_train_np, y_train_pred_np)
        roc_score_test = roc_auc_score(y_test_np, y_test_pred_np)
        #metrics_dict['roc'] = str(roc_score)
        cur_metric_values.append(roc_score_train)
        cur_metric_values.append(roc_score_test)

        metric_values.append(cur_metric_values)

        '''
        if(metric_type == "accuracy"):
            # Accuracy Score on test dataset
            acc_model = accuracy_score(df_y,df_pred)
            # print("Accuracy of " + alg_list[index] + " is " + str(acc_model) )

            metrics_map[alg_list[index]] = str(acc_model)

        elif(metric_type == "precision"):

            p_score = precision_score(df_y_np, df_pred_np, average='weighted')
            metrics_map[alg_list[index]] = str(p_score)
            # print("Precision of " + alg_list[index] + " is " + str(p_score))
        
        elif(metric_type == "recall"):
            
            r_score = recall_score(df_y_np, df_pred_np, average='weighted')
            metrics_map[alg_list[index]] = str(r_score)
            # print("Recall of " + alg_list[index] + " is " + str(r_score))
        
        elif(metric_type == "f1_score"):

            f1m_score = f1_score(df_y_np, df_pred_np, average='weighted')
            metrics_map[alg_list[index]] = str(f1m_score)
            #print("F1 score of " + alg_list[index] + " is " + str(f1m_score))
        
        elif(metric_type == "roc"):

            roc_score = roc_auc_score(df_y_np, df_pred_np)
            metrics_map[alg_list[index]] = str(roc_score)
            #print("ROC score of " + alg_list[index] + " is " + str(roc_score) )
        
        elif(metric_type == "creport"):
            print("Classification report of " + alg_list[index] + " is: ")
            report = classification_report(df_y, df_pred)
            report_df = cr_todf(report)
            print(report_df)
        else:
            pass
        '''

        index += 1

    # Convert the metric results in to dataframe sorted by accuracy value
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    metric_df = pd.DataFrame(metric_values, columns = t_cols_list)
    metric_df = metric_df.sort_values("Train Accuracy",ascending=False) #.set_index('feature')
    
    return metric_df.to_html(index=False)


class ML_Wrapper: 

    # SVM - Support vector machines
    # PCA - Principle component analysis
    # DT - Decision trees
    # RF - Random forest
    # NB - Naive Bayes
    # SGD - Stocastic gradient descent
    # XB - XGBoost
    # LR - Logistic regression
    # KNN - K nearest neighbours
    ml_algoritms_list = ['SVM', 'PCA', 'DT', 'RF', 'NB', 'SGD', 'XB', 'LR', 'KNN']

    # cm - confusion matrix
    # caccuracy - cost sensitive accuracy
    # auc - area under ROC curve
    metrics_list = ['cm', 'accuracy', 'caccuracy','precision', 'recall', 'auc']

    # Hyperparameter tuning methods
    # cv - cross validation
    # gs - grid search
    # rs - random search
    # bt - bayesian techniques
    h_tuning_list = ['cv', 'gs', 'rs', 'bt']


    def prepare_data(self, df, output_col, d_split):

        df = df.reset_index()

        # extract numerical attributes and scale it to have zero mean and unit variance  
        cols = df.select_dtypes(include=['float64','int64']).columns
        sc_df = scaler.fit_transform(df.select_dtypes(include=['float64','int64']))

        # turn the result back to a dataframe
        sc_df = pd.DataFrame(sc_df, columns = cols)

        # ENCODING CATEGORICAL ATTRIBUTES

        # extract categorical attributes from input dataframe
        catdf = df.select_dtypes(include=['object']).copy()

        # encode the categorical attributes
        dfcat = catdf.apply(encoder.fit_transform)

        # separate target column from encoded data 
        encdf = dfcat.drop([output_col], axis=1)
        cat_Ydf = dfcat[[output_col]].copy()

        df_x = pd.concat([sc_df,encdf],axis=1)
        df_y = df[output_col]

        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=d_split)

        return X_train, X_test, y_train, y_test

    def get_experiments(self, e_list, c_df, output_col, d_split):

        e_response = {}

        X_train, X_test, y_train, y_test = self.prepare_data(c_df, output_col, d_split)

        for e in e_list:
            if e in self.e_dict:
                e_response[e] = self.e_dict[e](X_train, X_test, y_train, y_test)

        return e_response

    
    e_dict = {
        'importance': get_feature_importance,
        'metrics': get_metrics 
    }


'''
train = pd.read_csv("data/ID_train.csv")
ml_obj = ML_Wrapper()
e_res = ml_obj.get_experiments(['metrics'], train, 'class', 0.2) # class, flag, protocol_type, service

print(e_res)

#print(e_res)
#print(f_df['feature'])
#print(f_df['importance'])
#print(f_df)
'''