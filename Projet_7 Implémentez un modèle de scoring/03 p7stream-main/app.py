import streamlit as st
import client, home, general
import analyse_exploratoire
import pandas as pd

st.set_page_config(layout="wide", page_title='dashboard P7')

@st.cache
def load_data():
    df = pd.read_csv('df_feats_sample.csv', index_col=0)
    df.drop(columns='TARGET', inplace=True)
    index1500 = [i for i in range(0,1500)]
    df.index = index1500
    df_appli = pd.read_csv('df_application_sample.csv')
    df_features = pd.read_csv('features174 meanings.csv', sep=';', header=None)
    df_features.columns = ['TAG_FEAT', 'Meaning']
    df_analyse = pd.read_csv('df_analyse.csv', index_col=0)
    df_feats = pd.read_csv('feats_sample_shap_values_lgb.csv', index_col=0)
    return df, df_appli, df_features, df_analyse, df_feats

df, df_appli, df_features, df_analyse, df_feats = load_data()



st.title("Pret a DEPENSER")


def main():
    st.sidebar.write('# Pret a Depenser')
    st.sidebar.write('## utilisateur :', st.session_state.user)
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select a page:',
                               ['Home', 'General Information', 'Client Information', 'Clients Analysis'])

    if options == 'Home':
        home.home()
    elif options == 'General Information':
        general.general(df_analyse)
    elif options == 'Client Information':
        client.client(df, df_appli, df_features, df_feats)
    elif options == 'Clients Analysis':
        analyse_exploratoire.run(df, df_features)

# Initialization
if 'user' not in st.session_state:
    st.session_state.user = ''
if 'password' not in st.session_state:
    st.session_state.password = ''
if 'loginOK' not in st.session_state:
    st.session_state.loginOK = False

if ~st.session_state.loginOK:
#if st.session_state.password != 'pwd123':
    user_placeholder = st.empty()
    pwd_placeholder = st.empty()
    user = user_placeholder.text_input("User:", value="")
    st.session_state.user = user
    pwd = pwd_placeholder.text_input("Password:", value="", type="password")
    st.session_state.password = pwd
    if st.session_state.password == 'pwd123':
        st.session_state.loginOK = True
        user_placeholder.empty()
        pwd_placeholder.empty()
        main()
    else:
        st.error("the password you entered is incorrect")
else:
    main()