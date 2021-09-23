import streamlit as st

def home():
    st.image('PretADepenser.jpg', width=300)
    st.write("""
    Ce site est composé de trois parties :
    
      
    **1. GENERAL INFORMATION**
    
        description du jeu de données et rappel interpretablite du modèle
            
    **2. CLIENT INFORMATION**
    
        Information sur un client , prediction crédit et interprétation résultat
    
    **3. CLIENT ANALYSIS**
    
        Recherche dans le jeu de données sur plusieurs en fonction de critères au choix
    
    """)