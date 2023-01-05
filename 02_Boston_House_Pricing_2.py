import streamlit as st
import pandas as pd
#import shap
import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from statistics import mean
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

st.write("""
# Regression Model
This app predicts using **Random Forest**!
""")
st.write('---')

# Loads the Boston House Price Dataset
st.write('---')
#Load the Boston House Price Dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
header= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
X = pd.DataFrame(data,columns=header)
Y = pd.DataFrame(target,columns=["MEDV"])
st.write(X.head(2))
st.write(Y.head(2))

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
CRIM = st.sidebar.slider('CRIM',min(X.CRIM),max(X.CRIM),mean(X.CRIM))
ZN = st.sidebar.slider('ZN',min(X.ZN),max(X.ZN),mean(X.ZN))
INDUS = st.sidebar.slider('INDUS',min(X.INDUS),max(X.INDUS),mean(X.INDUS))
CHAS = st.sidebar.slider('CHAS',min(X.CHAS),max(X.CHAS),mean(X.CHAS))
NOX = st.sidebar.slider('NOX',min(X.NOX),max(X.NOX),mean(X.NOX))
RM = st.sidebar.slider('RM',min(X.RM),max(X.RM),mean(X.RM))
AGE = st.sidebar.slider('AGE',min(X.AGE),max(X.AGE),mean(X.AGE))
DIS = st.sidebar.slider('DIS',min(X.DIS),max(X.DIS),mean(X.DIS))
RAD = st.sidebar.slider('RAD',min(X.RAD),max(X.RAD),mean(X.RAD))
TAX = st.sidebar.slider('TAX',min(X.TAX),max(X.TAX),mean(X.TAX))
PTRATIO = st.sidebar.slider('PTRATIO',min(X.PTRATIO),max(X.PTRATIO),mean(X.PTRATIO))
B = st.sidebar.slider('B',min(X.B),max(X.B),mean(X.B))
LSTAT = st.sidebar.slider('LSTAT',min(X.LSTAT),max(X.LSTAT),mean(X.LSTAT))
data = {'CRIM': CRIM,'ZN': ZN,'INDUS': INDUS,'CHAS': CHAS,'NOX': NOX,'RM': RM,'AGE': AGE,'DIS': DIS,'RAD': RAD,'TAX': TAX,'PTRATIO': PTRATIO,'B': B,'LSTAT': LSTAT}
#st.write(data)
df = pd.DataFrame(data, index=[0])

# Main Panel

# # Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# # Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

## Features
st.header('Important Features')
d = pd.DataFrame(model.feature_importances_,columns=["Features"])
d.index = header 
d.sort_values(by='Features', ascending=False)
st.dataframe(d, width=500, height=800)
st.write(plt.barh(boston.feature_names, rf.feature_importances_))

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
