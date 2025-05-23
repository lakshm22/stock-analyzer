import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from forex_python.converter import CurrencyRates

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data
def preprocess_data(data):
    data['Date'] = data.index
    data = data[['Date', 'Close']]
    
    # Check if 'High' and 'Low' columns exist, if not add them as NaN or use 'Close' for simplicity
    if 'High' not in data.columns:
        data['High'] = data['Close']
    if 'Low' not in data.columns:
        data['Low'] = data['Close']
    
    # Min-max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    return data, scaler

# Function to create dataset for training
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to build and train model
def build_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def test_sample():
    assert 1 + 1 == 2

def convert_currency(amount, from_currency="USD", to_currency="INR"):
    c = CurrencyRates()
    try:
        converted_amount = c.convert(from_currency, to_currency, amount)
        return converted_amount
    except Exception as e:
        st.warning(f"Currency conversion failed: {e}")
        return amount  # Fallback to original
    
# Streamlit app to visualize and predict stock price
def run_app():
    
    st.title("📈 Stock Price Analyzer")
    
    # User inputs
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
    start_date = st.date_input("Select Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("Select End Date", pd.to_datetime('2022-01-01'))
    currency = st.selectbox("Select Currency", ["USD", "INR"])  # Allow user to select currency
    time_unit = st.selectbox("Select Time Unit", ["DAYS", "MONTHS"])  # Select time unit for X-axis
    
    # Fetch and preprocess stock data
    data = fetch_stock_data(ticker, start_date, end_date)
    data, scaler = preprocess_data(data)
    
    # Add daily return
    data['Daily Return (%)'] = data['Close'].pct_change() * 100

    # Add volatility based on 'High' and 'Low'
    data['Volatility'] = data['High'] - data['Low']
    
    # Create dataset for training
    X, y = create_dataset(data['Close'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = build_model(X_train, y_train)
    
    # Model prediction
    predicted_price = model.predict(X_test)

    # Ensure 2D shape for inverse transform
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
   
    # === Financial Metrics (with corrected 2D reshape) ===
    initial_scaled = data['Close'].iloc[0]
    final_scaled = data['Close'].iloc[-1]

    initial_price = scaler.inverse_transform(np.array(initial_scaled).reshape(1, -1))[0][0]
    final_price = scaler.inverse_transform(np.array(final_scaled).reshape(1, -1))[0][0]

    profit_margin = abs((final_price - initial_price) / initial_price) * 100


    close_prices = scaler.inverse_transform(data['Close'].values.reshape(-1, 1)).flatten()
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    risk_percentage = np.std(daily_returns) * 100

    # Convert profit and prices if INR is selected
    if currency == "INR":
        initial_price = convert_currency(initial_price, "USD", "INR")
        final_price = convert_currency(final_price, "USD", "INR")
        close_prices = [convert_currency(p, "USD", "INR") for p in close_prices]
        predicted_price = [convert_currency(p[0], "USD", "INR") for p in predicted_price]
        y_test = [convert_currency(p[0], "USD", "INR") for p in y_test]
        currency_symbol = "₹"
else:
    currency_symbol = "$"


    st.markdown("### 📊 Financial Metrics")
    st.write(f"*Profit Margin:* {profit_margin:.2f}%")
    st.write(f"*Risk (Volatility):* {risk_percentage:.2f}%")


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the actual and predicted prices
    ax.plot(y_test, color='blue', label="Actual Price")
    ax.plot(predicted_price, color='red', label="Predicted Price")

    # Highlight the first and last points for better visualization
    ax.scatter([0, len(y_test)-1], [y_test[0], predicted_price[-1]], color='green', zorder=5)
    ax.text(0, y_test[0], f'{y_test[0][0]:.2f} {currency}', color='green', fontsize=10, verticalalignment='bottom')
    ax.text(len(y_test)-1, predicted_price[-1][0], f'{predicted_price[-1][0]:.2f} {currency}', color='green', fontsize=10, verticalalignment='top')

    # Add title and labels
    ax.set_title(f'{ticker} Stock Price Prediction', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'Stock Price ({currency_symbol})', fontsize=12)

    # Add gridlines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend
    ax.legend(loc='upper left', fontsize=12)

    # Customize x-axis ticks (time in DAYS or MONTHS)
    if time_unit == "DAYS":
        ax.set_xticks(np.arange(0, len(y_test), step=len(y_test)//5))
        ax.set_xticklabels([str(i) for i in np.arange(0, len(y_test), step=len(y_test)//5)])
    elif time_unit == "MONTHS":
        # Assuming monthly data, we use the Date index to show months
        data['Month'] = data['Date'].dt.strftime('%Y-%m')
        ax.set_xticks(np.arange(0, len(data), step=len(data)//5))
        ax.set_xticklabels(data['Month'][::len(data)//5])

    # Display the plot
    st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    run_app()
