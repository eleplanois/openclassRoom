import pandas as pd
import shap
import requests
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def client(df, df_features):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    explainer_shap = -0.5813475526725127

    df_feats = pd.read_csv('feats_sample_shap_values_lgb.csv', index_col=0)

    list_client_id = ["321233","326865","362153","399215","273004","386852","174425","203834","263961","103255","393937","124290","393147","274552","306176","253537","167565","432855","284022","445189","187478","135480","414327","448969","307757","376673","273602","368642"]
    list_client_id = sorted(list_client_id)

    st.write("""
    # STREMLIT
    ## premier essai stream
    alors ok
    """)

    def update_sk(sk_id):
        sk_id = 362153
        predict_proba_1 = 0.6215692372249696

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
        shap_object = ShapObject(base_values = explainer_shap,
                                 values = df_feats.loc[ind].values,
                                 feature_names = df.columns,
                                 data = df.iloc[ind,:])
        df_shap = pd.DataFrame(np.abs(df_feats.loc[ind].values), df.columns, columns=['abs_shap'])
        list_shap_feats = list(df_shap.sort_values(by='abs_shap', ascending=False).head(20).index)
#        list_shap_feats = sorted(list_shap_feats)
        df_features[df_features.TAG_FEAT.isin(list_shap_feats)]

        return shap.waterfall_plot(shap_object, max_display=20), df_features[df_features.TAG_FEAT.isin(list_shap_feats)]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig, df_top_feats = update_shap(option_sk, fig)
    st.pyplot(fig)

    st.table(df_top_feats)