import streamlit as st 
import pandas as pd 
#import matplotlib.pyplot as plt 
#import shap
import numpy as np
#from sklearn import datasets
#from sklearn.datasets import fetch_california_housing
#from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App

This predicts the  **Boston House Prices**!
""")
st.write('---')
#Load the Boston House Price Dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
header= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
X = pd.DataFrame(data,columns=header)
X = X.astype(int)
Y = pd.DataFrame(target,columns=["MEDV"])
Y = Y.astype(int)
st.write(X.head(2))
st.write(Y.head(2))

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

# def user_input_features():
    CRIM = st.sidebar.slider('CRIM', min(X.CRIM), max(X.CRIM), np.mean(X.CRIM))
    ZN = st.sidebar.slider('ZN', min(X.ZN), max(X.ZN), np.mean(X.ZN))
    INDUS = st.sidebar.slider('INDUS', min(X.INDUS), max(X.INDUS), np.mean(X.INDUS))
    CHAS = st.sidebar.slider('CHAS', min(X.CHAS), max(X.CHAS), np.mean(X.CHAS))
    NOX = st.sidebar.slider('NOX', min(X.NOX), max(X.NOX), np.mean(X.NOX))
    RM = st.sidebar.slider('RM', min(X.RM), max(X.RM), np.mean(X.RM))
    AGE = st.sidebar.slider('AGE', min(X.AGE), max(X.AGE), np.mean(X.AGE))
    DIS = st.sidebar.slider('DIS', min(X.DIS), max(X.DIS), np.mean(X.DIS))
    RAD = st.sidebar.slider('RAD', min(X.RAD), max(X.RAD), np.mean(X.RAD))
    TAX = st.sidebar.slider('TAX', min(X.TAX), max(X.TAX), np.mean(X.TAX))
    PTRATIO = st.sidebar.slider('PTRATIO', min(X.PTRATIO), max(X.PTRATIO), np.mean(X.PTRATIO))
    B = st.sidebar.slider('B', min(X.B), max(X.B), np.mean(X.B))
    LSTAT = st.sidebar.slider('LSTAT', (X.LSTAT), max(X.LSTAT), np.mean(X.LSTAT))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

# df = user_input_features()

# Main Panel

#Print Specified input Parameters
# st.header('Specified Input Parameters')
# #st.write(df)
# st.write('---')

# # Build Regression Model
# model =RandomForestRegressor()
# model.fit(X,Y)
# #Apply model to make prediction
# prediction = model.predict(df)

# st.header('Prediction of MEDV')
# st.write(prediction)
# st.write('---')


# # Explain the model's predictability using SHAP values

# explainer = shap.TreeExplainer(model)
# shap_values =explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature Importance based on SHAP Values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches = 'tight')
# st.write('---')


# plt.title('Feature importance based on SHAP values (BAR)')
# shap.summary_plot(shap_values, X, plot_type = "bar")
# st.pyplot(bbox_inches= 'tight')
