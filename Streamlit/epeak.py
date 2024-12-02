import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EPEAK"
)


# Read our dataset, load it and prepare it as we need 
df = pd.read_csv("processed_data.csv")

# Design the Web App: principal page
st.image("epeak_logo.png", width=400)
st.title("A Mobility Prediction Application")
st.sidebar.write("Group 102.G, with Telefonica")

st.write("We are presenting **EPEAK**, a predictive model of mobility peaks used to regularize public\
          transportation between the provinces with higher population.")
st.write("We want to identify the causes of traffic peaks and improve the management of public transportation\
          fleets according to demand.")
st.write("We will develop a predictive model with datasets like:")
st.write("*   Telefónica’s mobility data")
st.write("*  Meteorological information from AEMET")
st.write("*  National and provincial holiday calendars")
st.write("*  Relevant events such as congresses or festivals taking place in such provinces")

st.subheader("Our objective?")
st.write("That public transport companies can use our results to increase their fleet size during predicted peaks\
          of demand, and the other way round during low-demand periods, so traffic is reduced and \
         therefore contamination too. ")

st.subheader("Final Dataset:")
st.write("Here you can see the final dataset, where all information from all datasets is\
          processed together as a unique csv:")
st.write(df)
