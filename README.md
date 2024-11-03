# Stock Trend Predictor

## Overview
The Stock Trend Predictor is a web application built with Streamlit that allows users to predict stock prices based on historical data. The application fetches stock data, calculates moving averages, and uses a pre-trained machine learning model to forecast future stock prices.

## Features
- Fetches stock data from Yahoo Finance.
- Displays basic data description and visualizes closing prices over time.
- Computes and visualizes 100-day and 200-day moving averages.
- Utilizes a pre-trained Keras model to predict stock prices.
- Visualizes predicted prices against actual closing prices.

## Technologies Used
- Python
- Streamlit
- Keras (with TensorFlow)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Getting Started

### Prerequisites
Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
   
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   
3. Install the required packages:
    ```bash
   pip install -r requirements.txt

### Running the Application
1. Run the Streamlit app:
   ```bash
   streamlit run app.py

2. Open your web browser and navigate to http://localhost:8501 to view the application.

### Model
The application uses a pre-trained Keras model saved in the stock_predictions.h5 file for making predictions. Ensure that this file is located in the same directory as the app.py.

### Jupyter Notebook
The Jupyter Notebook stock_market_prediction.ipynb contains the code for the stock market prediction model training. You can modify and run it to train the model with different parameters.
