
#import paramiko
# import pdb
import json

import charts_wrapper as cw

import pandas as pd

from plotly import utils as pu

#import sys
#sys.path.append("..")

import sys, os
sys.path.append(os.path.join(sys.path[0],'backends', 'Code'))

#pdb.set_trace()

# Adding deep learning related packages
from backends.Code import ad_run as ar
#import ad_run as ar
adObj = ar.AD_Run()

plot_map = {
    "acc": "Train Accuracy", 
    "val_acc": "Validation Accuracy",
    "loss": "Train Loss", 
    "val_loss": "Validation Loss"
}

class DL_Wrapper: 

    def launch_predict(self, pred_list, dtrain_name, dtest_name):
        
        results_df, metric_df = adObj.start(False)

        p_response = {
            "prediction": results_df.to_html(index=False),
            "prediction summary": ''
        }

        pred_pie = cw.get_pie(results_df, "prediction")

        pred_pie_obj = {}
        pred_pie_obj['data'] = json.dumps(pred_pie, cls = pu.PlotlyJSONEncoder)
        
        p_response['prediction summary'] = pred_pie_obj

        return p_response

        

    def run_experiment(self, e_list, c_df, output_col, d_split):
        
        # Invoke deep learning model
        # True for train
        # False for predict
        results_df, metric_df = adObj.start(True)
        # For unit testing
        #results_df = pd.read_csv("history.log")

        e_response = {
            "history": results_df.to_html(index=False),
            "accuracy": "",
            "loss": "",
            "dl_metrics": metric_df.to_html(index=False)
        }

        #['epoch', 'acc', 'loss', 'val_acc', 'val_loss']

        accuracy_plot_values = cw.plot_results(results_df, "epoch", "acc", "val_acc", plot_map)
        loss_plot_values = cw.plot_results(results_df, "epoch", "loss", "val_loss", plot_map)

        aGraphObj = {}
        aGraphObj['data'] = json.dumps(accuracy_plot_values, cls=pu.PlotlyJSONEncoder)

        lGraphObj = {}
        lGraphObj['data'] = json.dumps(loss_plot_values, cls=pu.PlotlyJSONEncoder)
        
        e_response["accuracy"] = aGraphObj
        e_response["loss"] = lGraphObj

        
        '''
        e_response = {
            "ssh_stdout": [],
            "ssh_stderr": []
        }

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys()
        ssh.connect("52.232.124.137", 22, username="pugliese", password="anomaly$2020")
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("/home/pugliese/dl_pugliese/anomaly_detection/ad_run")

        for line in ssh_stdout:
            e_response["ssh_stdout"].append(line)

        for line in ssh_stderr:
            e_response["ssh_stderr"].append(line)

        '''

        return e_response


'''
# For unit testing
dlw = DL_Wrapper()
print(dlw.launch_predict('', '', ''))
#print(dlw.run_experiment('', '', '', ''))
'''