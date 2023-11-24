# General imports
from operator import index
import streamlit as st
import plotly.express as px
import os 
import pandas as pd

# Profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.regression import setup, compare_models, pull, save_model, load_model



if os.path.exists('./data/dataset.csv'): 
    df = pd.read_csv('data/dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Simon's App")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('data/dataset.csv', index=None)
        st.dataframe(df)


if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    df.dropna(subset=[chosen_target], inplace=True)
    df[chosen_target] = pd.to_numeric(df[chosen_target], errors='coerce')
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')


if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")