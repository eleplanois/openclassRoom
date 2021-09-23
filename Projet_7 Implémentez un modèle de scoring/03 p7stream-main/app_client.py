import plotly.graph_objects as go
import pandas as pd
import shap
import requests
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('df_feats_sample.csv', index_col=0)

import pickle
with open('feats_sample_explainer_shap.p', 'rb') as f3:
    explainer_shap = pickle.load(f3)

df_feats = pd.read_csv('feats_sample_shap_values_lgb.csv', index_col=0)

df_TARGET = df.TARGET.copy()
df.drop(columns='TARGET', inplace=True)
index1500 = [i for i in range(0,1500)]
df.index = index1500


response = requests.get("https://p7api.herokuapp.com/predict/")
if response:
    list_client_id = response.json()['list_client_id']
    list_client_id = sorted(list_client_id)
else:
    print("erreur web : ", response)

st.write("""
# STREMLIT
## premier essai stream
alors ok
""")

def update_sk(sk_id):
    predict_proba_1=0.5
    if sk_id in list_client_id:
        url_pred = "https://p7api.herokuapp.com/predict/" + sk_id
        response = requests.get(url_pred)
        if response:
            print(sk_id,response.json())
            predict_proba_0 = float(response.json()['predict_proba_0'])
            predict_proba_1 = float(response.json()['predict_proba_1'])
            retour_prediction = response.json()['retour_prediction']
        else:
            print("erreur web : ", response)

    gauge_predict = go.Figure(go.Indicator( mode = "gauge+number",
                                            value = predict_proba_1,
                                            domain = {'x': [0, 1], 'y': [0, 1]},
                                            gauge = {
                                                'axis': {'range': [0, 1], 'tickwidth': 0.2, 'tickcolor': "darkblue"},
                                                'steps': [
                                                    {'range': [0, 0.5], 'color': 'lightgreen'},
                                                    {'range': [0.5, 1], 'color': 'lightcoral'}],
                                                'threshold': {
                                                    'line': {'color': "red", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': 1}},
                                            title = {'text': f"client {sk_id}"}))

    return gauge_predict


option_sk = st.selectbox('Selectionner un numero de client',list_client_id)

fig = update_sk(option_sk)
st.plotly_chart(fig)

class ShapObject:

    def __init__(self, base_values, data, values, feature_names):
        self.base_values = base_values # Single value
        self.data = data # Raw feature values for 1 row of data
        self.values = values # SHAP values for the same row of data
        self.feature_names = feature_names # Column names


def update_shap(sk_id, fig):
    ind=df[df.SK_ID_CURR==int(sk_id)].index.values[0]
    shap_object = ShapObject(base_values = explainer_shap.expected_value[1],
                             values = df_feats.loc[ind].values,
                             feature_names = df.columns,
                             data = df.iloc[ind,:])

    return shap.waterfall_plot(shap_object, max_display=20)
#    return shap.force_plot(explainer_shap.expected_value[1], df_feats.loc[ind].values, df.iloc[ind,:], matplotlib=True)

fig, ax = plt.subplots(nrows=1, ncols=1)
fig = update_shap(option_sk, fig)
st.pyplot(fig)