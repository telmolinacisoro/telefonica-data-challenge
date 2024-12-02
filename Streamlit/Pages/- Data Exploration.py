import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read and preprocess the dataset once
df = pd.read_csv("processed_data.csv")
st.sidebar.write("Group 102.G, with Telefonica")

# --- Design the titles ---
st.title("Data Exploration🚘")

# --- 1. Total Travellers per Province (Origin) - Horizontal Bar Plot for better readability  ---
st.subheader("Total Travellers per Province (Origin)")

travelers_by_origin = df.groupby('provincia_origen_name')['viajeros'].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
travelers_by_origin.plot(kind='barh', color='skyblue')
plt.title('Total Travelers per Province (Origin)')
plt.xlabel('Total Travelers')
plt.ylabel('Province (Origin)')
plt.gca().invert_yaxis()  # To have the highest value at the top
st.pyplot(plt)




# --- 2. Daily Trips Over Time - Check and fix the date format if needed ---
st.subheader("Daily Trips Over Time")
daily_trips = df.groupby('day')['viajes'].sum()
plt.figure(figsize=(12, 6))
daily_trips.plot(color='skyblue')
plt.title('Daily Trips Over Time')
plt.xlabel('Date')
plt.ylabel('Total Trips')
plt.xticks(rotation=45)
st.pyplot(plt)



# --- 3. Top Travel Connections Between Provinces (Heatmap) ---
st.subheader("Top Travel Connections Between Provinces")
top_connections = df.groupby(['provincia_origen_name', 'provincia_destino_name'])['viajes'].sum().unstack().fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(top_connections, cmap="YlGnBu", linewidths=0.1)
plt.title('Top Travel Connections Between Provinces')
plt.xlabel('Destination Province')
plt.ylabel('Origin Province')
st.pyplot(plt)




# --- 4. Average peak detection (Interactive) ---
# The user needs to fill some information to be able to predict the salary
st.subheader("Average peak detection")
st.write("Please, fill the following information to visualize the weekly and monthly peaks:")
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", 
     "October", "November", "December"]
input_month = st.selectbox('Month:', months)
input_origin = st.selectbox('Origin:', df["provincia_origen_name"].unique())
input_dest = st.selectbox('Destination:', df["provincia_destino_name"].unique())

# Convert 'day' column to datetime if not already in datetime format
df['day'] = pd.to_datetime(df['day'])
df['month'] = df['day'].dt.month_name()

# Aggregate by day of the week and province origin and destination to see weekly patterns
df['day_of_week'] = df['day'].dt.day_name()
origin = df['provincia_origen_name'] == input_origin
dest = df['provincia_destino_name'] == input_dest
month = df["month"] == input_month
selection = df[origin & dest & month]
weekly_trips = selection.groupby('day_of_week')['viajes'].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

# Aggregate by month to see seasonal trends
selection = df[origin & dest]
monthly_trips = selection.groupby('month')['viajes'].mean().reindex(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", 
     "October", "November", "December"]
)

# Plotting weekly data with peaks and troughs
plt.figure(figsize=(10, 6))
plt.plot(weekly_trips.index, weekly_trips.values, color='skyblue', marker='o', label='Weekly Average Trips')
weekly_peaks, _ = find_peaks(weekly_trips.values, prominence=0.1)  # Adjust prominence as needed
weekly_troughs, _ = find_peaks(-weekly_trips.values, prominence=0.1)
plt.plot(weekly_trips.index[weekly_peaks], weekly_trips.values[weekly_peaks], 'r^', markersize=10, label='Weekly Peaks')  # Increased size
plt.plot(weekly_trips.index[weekly_troughs], weekly_trips.values[weekly_troughs], 'bv', markersize=10, label='Weekly Drops')  # Increased size
plt.title('Average Weekly Trips with Peaks and Troughs')
plt.xlabel('Day of the Week')
plt.ylabel('Average Trips')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Plotting monthly data with peaks and troughs
st.markdown(f"And here you can see the **average monthly trips** with the peaks and troughs for the trip between\
            {input_origin}-{input_dest}:")
plt.figure(figsize=(12, 6))
plt.plot(monthly_trips.index, monthly_trips.values, color='skyblue', marker='o', label='Monthly Average Trips')
monthly_peaks, _ = find_peaks(monthly_trips.values, prominence=0.1)
monthly_troughs, _ = find_peaks(-monthly_trips.values, prominence=0.1)
plt.plot(monthly_trips.index[monthly_peaks], monthly_trips.values[monthly_peaks], 'r^', markersize=10, label='Monthly Peaks')  # Increased size
plt.plot(monthly_trips.index[monthly_troughs], monthly_trips.values[monthly_troughs], 'bv', markersize=10, label='Monthly Drops')  # Increased size
plt.title('Average Monthly Trips with Peaks and Troughs')
plt.xlabel('Month')
plt.ylabel('Average Trips')
plt.legend()
plt.grid(True)
st.pyplot(plt)
