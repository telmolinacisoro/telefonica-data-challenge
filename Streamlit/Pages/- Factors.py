import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read and preprocess the dataset once
df = pd.read_csv("processed_data.csv")
st.sidebar.write("Group 102.G, with Telefonica")

# --- 1. Correlation Between All Factors (Heatmap) ---
# Columns selection
st.subheader("Correlation Between All Factors")

factors = ["weekday", 'viajeros', 'viajes',"is_holiday_origen","temp_origen",'precip_origen','icon_origen_clear-day',
           'icon_origen_cloudy','icon_origen_partly-cloudy-day','icon_origen_rain','icon_origen_wind',
           'is_event_origen','is_holiday_destino', 'temp_destino', 'precip_destino', 'icon_destino_clear-day',
           'icon_destino_cloudy','icon_destino_partly-cloudy-day','icon_destino_rain','icon_destino_wind', 
           'is_event_destino']

figure = plt.figure(figsize=(12, 10))
st.markdown('**Province Origin Factors**')
sns.heatmap(df[factors].corr(), cmap="YlGnBu", linewidths=0.1)
st.pyplot(figure)
