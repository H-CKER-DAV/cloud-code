from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Indian stock symbol mapping
INDIAN_STOCKS = {
    "HINDALCO": "HINDALCO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFC.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "LT": "LT.NS",
    # Add more Indian stock symbols as needed
}

# Function to fetch stock data
enddt = datetime.now().strftime('%Y-%m-%d')
def get_stock_data(symbol, start='2010-01-01', end=enddt):
    logging.info(f"Fetching stock data for symbol: {symbol}")
    stock_data = yf.download(symbol, start=start, end=end)
    
    if stock_data.empty:
        raise ValueError("No data found for the provided stock symbol.")
    
    stock_data = stock_data[['Close']]  # Only keep the 'Close' column
    stock_data = stock_data.ffill()  # Fill missing data (forward fill)
    logging.info(f"Stock data fetched successfully. Data size: {len(stock_data)} rows")
    return stock_data

# Function to train the model and predict Buy/Sell/Hold
def train_model(stock_data, prediction_days=15, model_type='linear'):
    logging.info(f"Training model with prediction days: {prediction_days} using {model_type} model")
    
    # Create the 'Prediction' column (next day's close price)
    stock_data['Prediction'] = stock_data[['Close']].shift(-prediction_days)
    
    # Define features (X) and labels (y)
    X = stock_data.drop(['Prediction'], axis=1).values[:-prediction_days]  # Closing price as features
    y = stock_data['Prediction'].values[:-prediction_days]  # Next day's closing price as labels

    # If there isn't enough data to make a prediction
    if len(X) < prediction_days:
        raise ValueError("Not enough data to train the model.")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Select the model type (Linear Regression or Random Forest)
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100)
    
    # Train the model
    model.fit(X_train, y_train)

    # Now we predict the next day's stock price using the last available close price (1 feature)
    last_close_price = stock_data['Close'].values[-1].reshape(-1, 1)  # Reshape for single feature
    future_price = model.predict(last_close_price)[0]  # Get future price as a single value
    current_price = last_close_price[0][0]  # Current closing price

    logging.info(f"Prediction completed. Future price: {future_price}")

    # Determine recommendation based on price change
    price_diff = future_price - current_price
    threshold = 0.01 * current_price  # 1% threshold to avoid small fluctuations
    if price_diff > threshold:
        recommendation = "Buy"
    elif price_diff < -threshold:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return recommendation, future_price, current_price

# Function to perform sentiment analysis on stock-related news
def sentiment_analysis(news_data):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(article) for article in news_data]
    average_sentiment = np.mean([sentiment['compound'] for sentiment in sentiments])
    return average_sentiment

# Route for the homepage
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'home.html')

# API route to predict stock prices with Buy/Sell/Hold recommendation
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received a request for prediction")
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        model_type = data.get('model_type', 'linear').lower()

        if not symbol:
            logging.warning("No stock symbol provided")
            return jsonify({'error': 'Please provide a valid stock symbol.'}), 400

        # Fetch stock data and make the prediction
        stock_data = get_stock_data(symbol)
        recommendation, future_price, current_price = train_model(stock_data, model_type=model_type)
        logging.info(f"Recommendation for {symbol}: {recommendation}")

        # Determine currency based on symbol
        if symbol.endswith('.NS'):
            currency = '₹'  # Indian Rupee for NSE stocks
        else:
            currency = '$'  # US Dollar for other stocks

        response = jsonify({
            'prediction': f"The recommendation for {symbol} is to '{recommendation}'.",
            'details': f"Predicted future price: {currency}{future_price:.2f}, Current price: {currency}{current_price:.2f}"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')  # Add CORS headers manually
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

        return response

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
