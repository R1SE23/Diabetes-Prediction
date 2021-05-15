import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.subplots as tls
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
import lightgbm as lgbm
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

# Stats
import scipy.stats as ss
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform



#------------COUNT-----------------------
# outcome distribution
@st.cache
def target_count(data):
    trace = go.Bar(x = data['Outcome'].value_counts().values.tolist(), 
                    y = ['healthy','diabetic' ], 
                    orientation = 'h', 
                    text=data['Outcome'].value_counts().values.tolist(), 
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.8,marker=dict(
                    color=['lightskyblue', 'gold'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Count of Outcome variable')

    fig = dict(data = [trace], layout=layout)
    return fig

#------------PERCENTAGE-------------------
@st.cache
def target_percent(data):
    trace = go.Pie(labels = ['healthy','diabetic'], values = data['Outcome'].value_counts(), 
                   textfont=dict(size=15), opacity = 0.8,
                   marker=dict(colors=['lightskyblue', 'gold'], 
                               line=dict(color='#000000', width=1.5)))


    layout = dict(title =  'Distribution of Outcome variable')

    fig = dict(data = [trace], layout=layout)
    return fig

# plot columns with missing values
@st.cache
def missing_value(data):
    percent_null = data.isnull().mean().round(4) * 100
    trace = go.Bar(x = percent_null.index, y = percent_null.values ,opacity = 0.8, text = percent_null.values,  textposition = 'auto',marker=dict(color = '#7EC0EE',
        line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values (count & %)")

    fig = dict(data = [trace], layout=layout)
    return fig

@st.cache
# plot distribution between two outcome
def plot_distribution(data_select, size_bin):  
    dat = pd.read_csv('diabetes.csv')
    D = dat[(dat['Outcome'] != 0)]
    H = dat[(dat['Outcome'] == 0)]
    # 2 datasets
    tmp1 = D[data_select]
    tmp2 = H[data_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['diabetic', 'healthy']
    colors = ['#ff71ce', '#85b6f7']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = data_select)

    return fig


def plot_all_feature(data):
    features = [i for i in data.columns]
    for feat in features[:-1]:
        f = plot_distribution(feat, 0)
        st.plotly_chart(f)
    
# plot feature correlation between the two
@st.cache
def feat_cor(data, data_x, data_y):
    fig = px.scatter(data, x=data_x, y=data_y,
              color='Outcome',size_max=18,color_continuous_scale=["#3339FF", "#ff8811"])
    fig.update_layout({"template":"plotly_dark"})
    return fig

# plot all feature correlation
def feat_cor_all(data):
    features = [x for x in data.columns]
    for i in range(len(features)):
        for z in range(i+1, len(features)):
            a = feat_cor(data, features[i], features[z])
            st.plotly_chart(a)

    

