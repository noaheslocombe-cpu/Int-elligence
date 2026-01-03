'''
Notes:
python -m streamlit run "Programming Projects\app\appSetup.py"         in cmd to run streamlit
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import urllib
import streamlit as st
import altair as alt
import openpyxl

import prophet # Import prophet for Prophet models
import xgboost as xgb # Import xgboost for DMatrix
import tensorflow as tf # Import tensorflow for LSTM models

# Functions
# Load data
def Load(file_path):
    # Loads data and removes rows with empty values
    data = pd.read_excel("Data/"+file_path)
    data=data.dropna()
    return data

def LSTM_Predict(model, historical_data, prediction_length, n_steps, last_historical_date):
    """
    Predicts future outcomes using a trained LSTM model.

    Args:
        model: The trained Keras Sequential model (LSTM).
        historical_data (np.array): A numpy array of the last n_steps historical data points.
        prediction_length (int): The number of future steps to predict.
        n_steps (int): The look-back window used by the model.
        last_historical_date (pd.Timestamp): The last date of the historical data.

    Returns:
        pd.DataFrame: A DataFrame containing 'Date' and 'Predicted Value' for the future.
    """
    temp_input = list(historical_data.flatten()) # Ensure it's a flat list for appending
    lst_output = []

    for i in range(prediction_length):
        # Reshape for LSTM input: (samples, timesteps, features)
        x_input = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
        prediction = model.predict(x_input, verbose=0)[0][0] # Assuming single output

        lst_output.append(prediction)
        temp_input.append(prediction) # Add prediction to input for next step

    # Prepare dates for plotting, starting from the day after the last historical date
    prediction_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                                     periods=prediction_length, freq='D')

    predicted_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Value': lst_output
    }).set_index('Date')

    return predicted_df


def XGBoost_Predict(model, historical_data, prediction_length, n_steps, last_historical_date):
    """
    Predicts future outcomes using a trained XGBoost model.

    Args:
        model: The trained XGBoost model (Booster type).
        historical_data (np.array): A numpy array of the last n_steps historical data points.
        prediction_length (int): The number of future steps to predict.
        n_steps (int): The look-back window used by the model.
        last_historical_date (pd.Timestamp): The last date of the historical data.

    Returns:
        pd.DataFrame: A DataFrame containing 'Date' and 'Predicted Value' for the future.
    """
    temp_input = list(historical_data.flatten()) # Ensure it's a flat list for appending
    lst_output = []

    for i in range(prediction_length):
        x_input = np.array(temp_input[-n_steps:]).reshape(1, -1)
        dmatrix_input = xgb.DMatrix(x_input)
        prediction = model.predict(dmatrix_input)[0]

        lst_output.append(prediction)
        temp_input.append(prediction)

    # Prepare dates for plotting, starting from the day after the last historical date
    prediction_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                                     periods=prediction_length, freq='D')

    predicted_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Value': lst_output
    }).set_index('Date')

    return predicted_df

def Prophet_Predict(model, historical_data_df, prediction_length):
    """
    Predicts future outcomes using a trained Prophet model.

    Args:
        model: The trained Prophet model.
        historical_data_df (pd.DataFrame): DataFrame containing historical data with 'ds' and 'y' columns.
        prediction_length (int): The number of future steps to predict.

    Returns:
        pd.DataFrame: The full forecast DataFrame from Prophet.
    """
    # Prepare historical data for Prophet
    prophet_df = historical_data_df.reset_index()
    prophet_df.columns = ['ds', 'y']

    # Make future dataframe
    future = model.make_future_dataframe(periods=prediction_length)
    forecast = model.predict(future)

    return forecast

# Import currency models data
# Load Data
df = Load('Foreign_Exchange_Rates.xlsx')
# Remove excess index column
df = df.drop('Unnamed: 0', axis=1)
# Remove rows with incomplete data
df = df.replace('ND', np.nan)
df = df.dropna()

# Convert first column to time series
df['Time Series'] = pd.to_datetime(df['Time Series'], format='%Y.%m.%d')
df.set_index('Time Series', inplace=True)
# Replace colomn names for ease of use
df = df.rename(columns = {'AUSTRALIA - AUSTRALIAN DOLLAR/US$': 'AUSTRALIAN DOLLAR',
                            'EURO AREA - EURO/US$': 'EURO',
                            'NEW ZEALAND - NEW ZEALAND DOLLAR/US$': 'NEW ZEALAND DOLLAR',
                            'UNITED KINGDOM - UNITED KINGDOM POUND/US$': 'BRITISH POUND',
                            'BRAZIL - REAL/US$': 'BRAZILIAN REAL',
                            'CANADA - CANADIAN DOLLAR/US$': 'CANADIAN DOLLAR',
                            'CHINA - YUAN/US$': 'CHINESE YUAN',
                            'HONG KONG - HONG KONG DOLLAR/US$': 'HONG KONG DOLLAR',
                            'INDIA - INDIAN RUPEE/US$': 'INDIAN RUPEE',
                            'KOREA - WON/US$': 'KOREAN WON',
                            'MEXICO - MEXICAN PESO/US$': 'MEXICAN PESOS',
                            'SOUTH AFRICA - RAND/US$': 'SOUTH AFRICAN RAND',
                            'SINGAPORE - SINGAPORE DOLLAR/US$': 'SINGAPORE DOLLAR',
                            'DENMARK - DANISH KRONE/US$': 'DANISH KRONE',
                            'JAPAN - YEN/US$': 'JAPANESE YEN',
                            'MALAYSIA - RINGGIT/US$': 'MALASIAN RINGGIT',
                            'NORWAY - NORWEGIAN KRONE/US$': 'NORWEGIAN KRONE',
                            'SWEDEN - KRONA/US$': 'SWEDISH KRONA',
                            'SRI LANKA - SRI LANKAN RUPEE/US$': 'SRI LANKAN RUPEE',
                            'SWITZERLAND - FRANC/US$': 'SWISS FRANC',
                            'TAIWAN - NEW TAIWAN DOLLAR/US$': 'TAIWANESE DOLLAR',
                            'THAILAND - BAHT/US$': 'THAI BAHT'})

# Load prediction models for each currency
currencies = df.columns.to_list()
models = {}
testcount = 1
for currency in currencies:
    print(testcount)
    testcount += 1
    f = open("Models/"+currency+'.pkl', 'rb')
    models[currency] = pickle.load(f)

# Allow the user to select the desired currency
selected_currency = st.selectbox("Select currency", currencies)
selected_model = models[selected_currency]
# Allow the user to select the desired prediction length
prediction_length = st.number_input("Value prediction (days)", 1, 365, value=50, step=1)

# Assuming a look-back window (n_steps) is used for models like XGBoost and LSTM
# If the models were trained with a specific window, this should match it.
n_steps = 10 # Placeholder: This should be consistent with model training

# Streamlit UI elements
st.title(f"{selected_currency} Exchange Rate Prediction")
model_type = str(type(selected_model)).split('.')[-1].replace("'>","") # Get simplified model type
st.write(f"Using {model_type} model")

# Get last 100 days of historical data for plotting
historical_for_plot = df[selected_currency].tail(100).rename('Historical Value')

if model_type == "Booster": # XGBoost model
    # Get the last n_steps historical data points
    input_data = df[selected_currency].tail(n_steps).values.astype(float)
    last_date = df.index[-2] # Get the last date from the DataFrame
    # Call the XGBoost_Predict function
    predictions_for_plot = XGBoost_Predict(selected_model, input_data, prediction_length, n_steps, last_date)
    # Combine historical data for plotting
    combined_df = pd.concat([historical_for_plot, predictions_for_plot])

    # Present on Streamlit
    st.subheader("XGBoost Model Prediction")
    st.line_chart(combined_df)
    st.write("Predicted values:")
    st.write(predictions_for_plot)

elif model_type == "Prophet": # Prophet model
    # Call the Prophet_Predict function
    forecast = Prophet_Predict(selected_model, df[[selected_currency]], prediction_length)
    # Filter for only the predictions after the historical data
    last_historical_date = df.index[-1]
    filtered_forecast = forecast[forecast['ds'] >= last_historical_date]
    predictions_for_plot = filtered_forecast.set_index('ds')[['yhat']]
    # Combine historical and predictive data
    combined_df = pd.concat([historical_for_plot, predictions_for_plot])
    
    # Present on Streamlit
    st.subheader("Prophet Model Prediction")
    st.line_chart(combined_df)
    st.write("Predicted values:")
    st.write(predictions_for_plot)

elif model_type == "Sequential": # LSTM model
    # Get the last n_steps historical data points
    input_data = df[selected_currency].tail(n_steps).values.astype(float)
    last_date = df.index[-1] # Get the last date from the DataFrame

    # Call the LSTM_Predict function
    predictions_for_plot = LSTM_Predict(selected_model, input_data, prediction_length, n_steps, last_date)
    # Combine historical data for plotting
    combined_df = pd.concat([historical_for_plot, predictions_for_plot])

    # Present on Streamlit
    st.subheader("LSTM Model Prediction")
    st.line_chart(combined_df)
    st.write("Predicted values:")
    st.write(predictions_for_plot)

else:
    st.write("Unsupported model type for prediction.")