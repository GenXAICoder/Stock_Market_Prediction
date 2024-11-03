import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit App setup
st.title('Stock Trend Predictor')

# Date range for the stock data
start = '2012-01-01'
end = '2024-11-01'

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'GOOG')

# Fetching the data with yf.download
df = yf.download(user_input, start=start, end=end)

# Displaying basic data description
st.subheader('Data from 2012 to 2024')
st.write(df.describe())

# Plot Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

# Moving average 100
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()  # Added parentheses for .mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b', label='Closing Price')
plt.plot(ma100, 'r', label='100-day MA')
plt.legend()
st.pyplot(fig)

# Moving average 200
st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma200 = df['Close'].rolling(200).mean()  # Added parentheses for .mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'b', label='Closing Price')
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(ma200, 'g', label='200-day MA')
plt.legend()
st.pyplot(fig)

# Splitting data into training and test sets
data_train = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_train)

# Load the pre-trained model
model = load_model('stock_predictions.h5')

# Preparing test data
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)  # Use transform() instead of fit_transform()

# Creating test sequences
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting the values
y_predicted = model.predict(x_test)

# Inversing the scale for predictions and actual values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting the results
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
