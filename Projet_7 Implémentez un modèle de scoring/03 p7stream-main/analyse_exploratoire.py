import streamlit as st
import plotly.graph_objs as go

def run(df, df_features):
    st.write("""
    ## ANALYSE EXPLORATOIRE
    """)
    st.write("""
    ### CLIENTS
    """)
    list_client_id = sorted(list(df.SK_ID_CURR))
    list_client_selec = st.multiselect("selection Multiple clients", list_client_id)

    st.write("""
    ### VARIABLES A AFFICHER DANS LE TABLEAU
    """)
    colonne = list(df.columns)
    colonne = sorted(colonne)
    list_colonne =  st.multiselect("selection Multiple variables", colonne,
                                   default=['SK_ID_CURR','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','INSTAL_DPD_MEAN',
                                            'AMT_CREDIT', 'PAYMENT_RATE','CODE_GENDER','DAYS_BIRTH',
                                            'AMT_ANNUITY'])

    df_isin = df[df.SK_ID_CURR.isin(list_client_selec)]
    df_notin = df[~df.SK_ID_CURR.isin(list_client_selec)]

    st.dataframe(df_isin[list_colonne])

    st.write("""
    ## GRAPHIQUE INTERACTIF POUR RECHERCHE CLIENTS VOISINS
    """)
    st.write("""
    ### ABSCISSE GRAPH
    """)
    graph_colonne_X = st.selectbox(label='abscisse X : ', options=colonne, index=66)
    text_X = df_features[df_features.TAG_FEAT==graph_colonne_X]['Meaning'].values[0]
    st.text(text_X)
    st.write("""
    ### ORDONNEE GRAPH
    """)
    graph_colonne_Y = st.selectbox(label='ordonnee Y : ', options=colonne, index=65)
    text_Y = df_features[df_features.TAG_FEAT==graph_colonne_Y]['Meaning'].values[0]
    st.text(text_Y)

    data1 = go.Scatter(
        x=df_isin[graph_colonne_X],
        y=df_isin[graph_colonne_Y],
        mode='markers',
        marker=dict(
            size=6,
            color='rgb(255,87,51)',
            symbol='square'),
        text=df_isin['SK_ID_CURR'],
        name='Selection'
    )

    data2 = go.Scatter(
        x=df_notin[graph_colonne_X],
        y=df_notin[graph_colonne_Y],
        mode='markers',
        marker=dict(
            size=4,
            color='rgb(45,180,250)',
            symbol='circle'),
        text=df_notin['SK_ID_CURR'],
        name='Hors selection'
    )

    data = [data1, data2]

    layout = go.Layout(
        title=graph_colonne_Y + " en fonction de " + graph_colonne_X,
        xaxis=dict(title=text_X),
        yaxis=dict(title=text_Y),
        hovermode='closest'
    )
    plotly_fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(plotly_fig, use_container_width=True)
