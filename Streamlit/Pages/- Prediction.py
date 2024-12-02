import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np
from prophet import Prophet



# ----------------------------------------------- FUNCTIONS TO USE --------------------------------------------
def predict_and_apriori_knowledge(jour, dataFrame, prioriDict, prophetPredictionsDf):

  # journey_key ex: Barcelona_to_Madrid
  # jour: day to predict in datetime format
  # dataFrame: the dataframe info with the known info (prophet trained daily with the updated dataframes)
  # priori_dict: the dict with the info to plug to correct prophet, and at the busy level
  # prophetPredictions: predictions obtained with the previously trained prophet model
  raw_prediction = prophetPredictionsDf['predictions'][prophetPredictionsDf['predictions']['ds'] == jour]['yhat'].values[0]
  

  ####---> Statistical properties extraction (De-trending of prediction day)
  X = dataFrame['days_since_start'].values.reshape(-1, 1)
  y = dataFrame['viajes'].values

  model = LinearRegression()
  model.fit(X, y)

  day_of_prediction = (jour - dataFrame['day'].iloc[0]).days
  predicted_trend = model.predict(np.array(day_of_prediction).reshape(-1, 1))[0]
  adjusted_prediction = raw_prediction - (predicted_trend - dataFrame['predicted_trend'].iloc[0])

  ###---> Data adjustments, information provision and preparation for time series analysis prediction correction
  contrastmedian = dataFrame['detrended_value'].quantile(0.5)

  if dataFrame['day'].tail(1).values[0] >= (jour - pd.Timedelta(days=7)):
    value1w = dataFrame[dataFrame['day'] == (jour - pd.Timedelta(days=7))]['viajes'].values[0]
    deviation = adjusted_prediction/contrastmedian
  elif dataFrame['day'].tail(1).values[0] >= (jour - pd.Timedelta(days=14)):
    value2w = dataFrame[dataFrame['day'] == (jour - pd.Timedelta(days=14))]['viajes'].values[0]
    deviation = adjusted_prediction/contrastmedian
    
  else:
    value1wp =  prophetPredictionsDf['predictions'][prophetPredictionsDf['predictions']['ds'] == (jour - pd.Timedelta(days=7))]['yhat'].values[0]
    contrastmedian = predicted_trend
    deviation = adjusted_prediction/contrastmedian

  ###---> data study and adjustment for proposed correction measures
  corrections = {}

  #--> Week day adjustment
  wprior, wjour, wpercentile = prioriDict['weekday']
  if wprior:
    corrections['weekday'] = dataFrame[dataFrame['weekday'] == wjour]['difference'].quantile(wpercentile/100)
    
  #--> events adjustment
  edprior, edlabel, edpercentile = prioriDict['is_event_destino']
  if edprior:
    corrections['is_event_destino'] = dataFrame[dataFrame['is_event_destino'] == edlabel]['difference'].quantile(edpercentile/100)
    
  eoprior, eolabel, eopercentile = prioriDict['is_event_origen']
  if eoprior:
    corrections['is_event_origen'] = dataFrame[dataFrame['is_event_origen'] == eolabel]['difference'].quantile(eopercentile/100)
    
  #--> Holiday adjustment
  hdprior, hdlabel, hdpercentile = prioriDict['is_holiday_destino']
  if hdprior:
    corrections['is_holiday_destino'] = dataFrame[dataFrame['is_holiday_destino'] == hdlabel]['difference'].quantile(hdpercentile/100)
    
  hoprior, holabel, hopercentile = prioriDict['is_holiday_origen']
  if hoprior:
    corrections['is_holiday_origen'] = dataFrame[dataFrame['is_holiday_origen'] == holabel]['difference'].quantile(hopercentile/100)
    
  combinedCorrections = 0
  if corrections.keys():
    for correction_to_apply in corrections.keys():
      combinedCorrections += corrections[correction_to_apply]
    combinedCorrections = combinedCorrections/len(corrections.keys())
  else:
    combinedCorrections = 1

  finalCorrection = (combinedCorrections + deviation)/2
  finalPred = (contrastmedian + (predicted_trend - dataFrame['predicted_trend'].iloc[0])) * (finalCorrection)
  
  return finalPred, combinedCorrections, finalCorrection



# ----------------------------------------------- STREAMLIT CODE ------------------------------------------------

# Read and preprocess the dataset once
df = pd.read_csv("processed_data.csv")
st.sidebar.write("Group 102.G, with Telefonica")

with open('epeak.pkl', 'rb') as file:
    data = pickle.load(file)

modDict = data["modDict"]
dataframes = data["database"]

st.title('Peak Traffic Prediction🔮')
st.markdown('We have trained a regression model with a Linear Regression  with \
           Prophet, and with that we can predict the amount of travels and hence the peaks of\
            traffic, but for that, we need some information: desired Date,\
            if it will rain in the desired destination, if there are important events in the\
             destination and if it is a holiday.')
st.markdown("**Please, fill this information, and then press Predict Traffic Peaks:**")


# The user needs to fill some information to be able to predict the salary
input_date = st.date_input("Date:", value="default_value_today",format="YYYY-MM-DD")
input_origin = st.selectbox('Origin:', df["provincia_origen_name"].unique())
input_dest = st.selectbox('Destination:', df["provincia_destino_name"].unique())


checkbox = st.checkbox("I want to take into account holidays and important events.")
if checkbox:
  st.write(f"Please, check the boxes that are True for the date {input_date}:")

  input_holiday_origin = st.checkbox(f"It is holiday in {input_origin}")
  if input_holiday_origin:   
    input_holiday_origin_percent = st.slider(f"Relevance of being holiday in {input_origin}?", 0, 100, step=25)
  else:
    input_holiday_origin_percent = 50

  input_event_origin = st.checkbox(f"There is an event in {input_origin}")
  if input_event_origin:
    input_event_origin_percent = st.slider(f"Relevance of this event in {input_origin}?", 0, 100, step=25)
  else:
    input_event_origin_percent = 50

  input_holiday_dest = st.checkbox(f"It is holiday in {input_dest}")
  if input_holiday_dest:
    input_holiday_dest_percent = st.slider(f"Relevance of being holiday in {input_dest}?", 0, 100, step=25)
  else:
    input_holiday_dest_percent = 50

  input_event_dest = st.checkbox(f"There is an event {input_dest}")
  if input_event_dest:
    input_event_dest_percent = st.slider(f"Relevance of this event in {input_dest}?", 0, 100, step=25)
  else:
    input_event_dest_percent = 50

else:
  input_holiday_origin = 0
  input_holiday_origin_percent = 50
  input_event_origin = 0
  input_event_origin_percent = 50
  input_holiday_dest = 0
  input_holiday_dest_percent = 50
  input_event_dest = 0
  input_event_dest_percent = 50

d = input_date.weekday()+1

user_inputs = {'weekday':(True, d, 75), 
 'is_holiday_origen':(checkbox, input_holiday_origin, input_holiday_origin_percent),
 'is_event_origen':(checkbox, input_event_origin, input_event_origin_percent), 
 'is_holiday_destino':(checkbox, input_holiday_dest, input_holiday_dest_percent),
 'is_event_destino':(checkbox, input_event_dest, input_event_dest_percent)
  }

_, _, _, jdict = modDict[f"{input_origin}_to_{input_dest}"]

button = st.button("Predict Traffic Peaks")
if button:
  finalPred, _, _ = predict_and_apriori_knowledge(pd.to_datetime(input_date), dataframes[f"{input_origin}_to_{input_dest}"], user_inputs, jdict)
  finalPred = str(int(finalPred))
  st.markdown(f'#### PREDICTION of TRAVELS {input_origin}-{input_dest} on {input_date}: {finalPred}')