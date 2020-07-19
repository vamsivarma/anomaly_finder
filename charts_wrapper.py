
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

import ml_wrapper as mlw

ml_obj = mlw.ML_Wrapper()


def plot_results(df, xlabel, ylabel1, ylabel2, plot_map):

    data = [go.Scatter(x=df[xlabel], y=df[ylabel1], name=plot_map[ylabel1]), 
            go.Scatter(x=df[xlabel], y=df[ylabel2], name=plot_map[ylabel2])]

    return data

def get_box(charts_df, label_col):
    label_values = list(charts_df[label_col])

    data = [go.Box(y=label_values,
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8 # relative position of points wrt box
              )]

    return data


def get_line(charts_df, label_col):
    x = np.arange(10)

    data = go.Scatter(x=x, y=x**2)

    return data

def get_lines(charts_df, label_col):
    x = np.arange(10)

    data = [go.Scatter(x=x, y=x**2, name="Square"), go.Scatter(x=x, y=x**3, name="Cube")]

    return data

def get_bar(charts_df, label_col):
    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    return data

def get_scatter(charts_df, label_col):
    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)

    # Create a trace
    data = [go.Scatter(
        x = random_x,
        y = random_y,
        mode = 'markers'
    )]

    return data

def get_histogram(charts_df, label_col):

    # Get the distribution of label column
    label_vc = charts_df[label_col].value_counts()

    # Convert it in to dictionary
    label_vc_dict = pd.DataFrame(label_vc).to_dict('dict')

    # Get the label field categories
    vc_dict_keys = list(label_vc_dict[label_col].keys())

    # values for that categories
    vc_dict_values = []

    for k in vc_dict_keys:
        vc_dict_values.append(label_vc_dict[label_col][k])

    labels = vc_dict_keys
    values = vc_dict_values
    
    x = vc_dict_values
    data = [go.Histogram(x=x)]

    return data


def get_heatmap(charts_df, label_col):

    corr_df = charts_df.corr()
    corr_df = list(corr_df.values)

    for i in range(0, len(corr_df)):
        corr_df[i] = list(corr_df[i])

    

    d=corr_df
    data=[go.Heatmap(z=d,
                    x = list(charts_df.columns),
                   y = list(charts_df.columns),
                   hoverongaps = False)
        ]
    return data

def get_correlation(charts_df, label_col):
    return True

def get_confusion_matrix(charts_df, label_col):
    return True

def get_area(charts_df, label_col):
    return True

def get_pie(charts_df, label_col):
    
    # Get the distribution of label column
    label_vc = charts_df[label_col].value_counts()

    # Convert it in to dictionary
    label_vc_dict = pd.DataFrame(label_vc).to_dict('dict')

    # Get the label field categories
    vc_dict_keys = list(label_vc_dict[label_col].keys())

    # values for that categories
    vc_dict_values = []

    for k in vc_dict_keys:
        vc_dict_values.append(label_vc_dict[label_col][k])

    labels = vc_dict_keys
    values = vc_dict_values

    data = [go.Pie(labels=labels, values=values)]

    return data

def get_stacked(charts_df, label_col):
    return True

def get_hortizontal_bar(charts_df, label_col):
    data = [go.Bar(
            x=[20, 14, 23],
            y=['giraffes', 'orangutans', 'monkeys'],
            orientation='h')]

    return data

def get_bubble(charts_df, label_col):

    data = [go.Scatter(
        x=[1, 2, 3, 4], y=[10, 11, 12, 13],
        mode='markers',
        marker=dict(
            color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
               'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
            opacity=[1, 0.8, 0.6, 0.4],
            size=[40, 60, 80, 100],
        )
    )]

    return data

def get_polar(charts_df, label_col):
    data = [ go.Scatterpolar(
        r = [0.5,1,2,2.5,3,4],
        theta = [35,70,120,155,205,240],
        mode = 'markers') 
    ]

    return data

def get_normal(charts_df, label_col):
    return True


def get_table(charts_df, label_col):
    d_type_map = {
        "int64": "Integer",
        "object": "Categorical",
        "float64": "Float",
        "bool": "Boolean"
    }

    field_list = []
    field_type_list = []
    value_list = []

    for i in range(len(charts_df.columns)):
        field_key = charts_df.columns[i]
        field_type = d_type_map[str(charts_df[charts_df.columns[i]].dtype)]
        field_na_value = charts_df[charts_df.columns[i]].isna().sum()
        
        field_list.append(field_key)
        field_type_list.append(field_type)
        value_list.append(field_na_value)

    table_values = [field_list, field_type_list, value_list]

    # Create table
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    data = [go.Table(header=dict(
            values=['<b>Name</b>','<b>Type</b>', '<b>NA count</b>'],
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left','center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=table_values,
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color = [[rowOddColor,rowEvenColor] * len(charts_df.columns)],
            align = ['left', 'center'],
            font = dict(color = 'darkslategray', size = 11)
        ))
    ]

    return data

def get_features(charts_df, label_col):
    f_df = ml_obj.get_feature_importance(charts_df, label_col)

    data = [
        go.Bar(
            x = f_df['feature'], 
            y = f_df['importance']
        )
    ]

    return data

    

class Charts_Wrapper: 

    # Mapper for dataset field type with Chart type
    chart_col_map = {
        'pie': ['Categorical', 'Boolean'],
        'histogram': ['Integer', 'Float'],
        'box': ['Integer', 'Float'],
        'heatmap': ['Integer', 'Float'],
        'field': ['Categorical', 'Boolean', 'Integer', 'Float'],
        'feature': ['Categorical', 'Boolean', 'Integer', 'Float'],
        'lines': ['Categorical', 'Boolean', 'Integer', 'Float']
    }

    def create_plot(self, charts_list, charts_df, col_meta, label_col):

        charts_data = {}
        chart_data = []
        
        for chart_type in charts_list:
            if chart_type in self.charts_dict:

                cur_chart_types = self.chart_col_map[chart_type]

                if(label_col == ''):
                    # Initializing the label column to the first column by default
                    label_col = charts_df.columns[1]

                    for col in col_meta:
                        if col_meta[col] in cur_chart_types:
                            label_col = col
                            # Break on first match
                            break

                chart_data = self.charts_dict[chart_type](charts_df, label_col)   
                graphJSON = json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)
                
                charts_data[chart_type] = {}
                charts_data[chart_type]['data'] = graphJSON
                charts_data[chart_type]['label_col'] = label_col
                charts_data[chart_type]['chart_types'] = cur_chart_types
          
        return charts_data

    charts_dict = {
        'line': get_line,
        'lines': get_lines,
        'bar': get_bar,
        'scatter': get_scatter,
        'histogram': get_histogram,
        'heatmap': get_heatmap,
        'correlation': get_correlation,
        'confusion_matrix': get_confusion_matrix,
        'area': get_area,
        'pie': get_pie,
        'stacked': get_stacked,
        'horizontal_bar': get_hortizontal_bar,
        'bubble': get_bubble,
        'polar': get_polar,
        'normal': get_normal,
        'field': get_table,
        'box': get_box,
        'feature': get_features 
    }
