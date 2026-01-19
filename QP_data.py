import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def compute_RSI(prices, period=7):
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    delta = prices.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

    rs = gain / (loss + 1e-10)  # Adding small epsilon to avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Function: Convert Data to Preferred Format
def preferred_format():
    data = st.session_state.data.copy()

    # Remove commas from 'Price' and convert to numeric
    data['Price'] = data['Price'].replace({',': ''}, regex=True).astype(float)
    data['Change %'] = data['Change %'].replace({'%': ''}, regex=True).astype(float) / 100

    # Calculate MACD and Signal Line
    data['EMA_12'] = data['Price'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Price'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Add RSI Smooth Calculation
    data['RSI'] = compute_RSI(data['Price'])  # Assuming you have an RSI computation function
    data['rsi_smooth'] = data['RSI'].rolling(window=5, min_periods=1).mean().fillna(method="bfill")   # Adjust window if needed

    # Scale 'Change %' using MinMaxScaler
    scaler = MinMaxScaler()
    data['change_scaled'] = scaler.fit_transform(data[['Change %']])

    # Drop unnecessary columns
    data = data.drop(columns=['EMA_12', 'EMA_26'], errors='ignore')

    # Create 'ds' and 'y' columns for Prophet
    data['ds'] = data['Year'] + '-' + data['Month'] + '-' + data['Day']
    data['y'] = data['Price']

    # Drop unnecessary columns
    data = data.drop(columns=['Year', 'Month', 'Day', 'Date', 'Price'], errors='ignore')

    # Validate the processed data
    if len(data) < 2:
        raise ValueError("Dataframe has less than 2 non-NaN rows after processing. Please check the input data.")

    st.session_state.processed_data = data