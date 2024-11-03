Stock Market Trend Predictor
This project is a stock trend prediction application that uses historical stock data to predict future trends. The application is built with Streamlit, a popular framework for creating web applications with Python, and uses a trained neural network model for predictions.

Table of Contents
Features
Setup and Installation
Usage
Files
Model Training
Dependencies
License
Features
Fetches historical stock data from Yahoo Finance.
Displays interactive visualizations of closing prices and moving averages.
Provides stock price predictions based on a pre-trained LSTM model.
Allows user input for any stock ticker symbol to analyze stock trends.
Setup and Installation
Clone the Repository
Clone this repository to your local machine.

bash
Copy code
git clone https://github.com/GenXAICoder/stock-market-trend-predictor.git
Navigate to the Project Directory

bash
Copy code
cd stock-market-trend-predictor
Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

bash
Copy code
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
Install Dependencies
Install the necessary packages from the requirements.txt file.

bash
Copy code
pip install -r requirements.txt
Download the Model File
Place the pre-trained stock_predictions.h5 model file in the project directory.

Usage
Run the Streamlit App
Start the Streamlit app by running:

bash
Copy code
streamlit run app.py
Enter Stock Ticker Symbol
In the app, enter the stock ticker symbol (e.g., GOOG for Google) to fetch and analyze stock data.

View Predictions
The app will display data descriptions, charts, and the model's predictions for stock trends.

Files
app.py: Main application file that sets up the Streamlit app, loads data, and makes predictions.
requirements.txt: Contains the required Python libraries to run the app.
stock_predictions.h5: Pre-trained model file for stock prediction.
Stock_market_Prediction.ipynb: Jupyter notebook used to train the model for stock market prediction.
Model Training
The LSTM model was trained on historical stock market data in the Stock_market_Prediction.ipynb notebook. The model file (stock_predictions.h5) is saved and used in the Streamlit app to predict stock prices based on a 100-day sequence of past closing prices.

Dependencies
The required dependencies for this project are listed in requirements.txt:

numpy
pandas
matplotlib
pandas_datareader
keras
streamlit
tensorflow
sklearn
License
This project is licensed under the MIT License.
