import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st

# Cache to avoid retraining
model_cache = {}

# ----------------------- Data Fetching -----------------------
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    return data[['Close']]

# ----------------------- Feature Engineering -----------------------
def preprocess_data(df):
    # Technical indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)

    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df

# ----------------------- Normalize Features -----------------------
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# ----------------------- Prepare Data for LSTM -----------------------
def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)

# ----------------------- Build LSTM Model -----------------------
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ----------------------- Streamlit Interface -----------------------
st.title('📈 Stock Price Prediction with LSTM')

stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
selected_stock = st.selectbox('Select a Stock:', stock_list)

st.write(f"Fetching data for {selected_stock}...")
data = fetch_data(selected_stock)
latest_price = data['Close'].iloc[-1]
st.write(f'Latest Closing Price: ${latest_price:.2f}')

if st.button("Train & Predict"):
    st.write("⏳ Training model, please wait...")

    try:
        if selected_stock in model_cache:
            model, scaler, scaled_data = model_cache[selected_stock]
        else:
            data = preprocess_data(data)
            scaled_data, scaler = normalize_data(data)

            X, y = prepare_data(scaled_data)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = build_model((x_train.shape[1], x_train.shape[2]))
            model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

            # Save to cache
            model_cache[selected_stock] = (model, scaler, scaled_data)

            # Evaluate
            loss = model.evaluate(x_test, y_test, verbose=0)
            st.write(f"Model Evaluation - MSE: {loss:.4f}")

            # Plot predicted vs actual on test set
            y_pred_test = model.predict(x_test)
            y_pred_test_inv = scaler.inverse_transform(
                np.hstack([y_pred_test, np.zeros((y_pred_test.shape[0], scaled_data.shape[1]-1))])
            )[:,0]
            y_test_inv = scaler.inverse_transform(
                np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))])
            )[:,0]

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=np.arange(len(y_test_inv)), y=y_test_inv, mode='lines', name='Actual'))
            fig_test.add_trace(go.Scatter(x=np.arange(len(y_test_inv)), y=y_pred_test_inv, mode='lines', name='Predicted'))
            fig_test.update_layout(title='Actual vs Predicted on Test Set', xaxis_title='Index', yaxis_title='Price ($)')
            st.plotly_chart(fig_test)

        # ----------------------- Future Prediction -----------------------
        st.write("Predicting next 10 days prices...")
        predictions = []
        input_sequence = scaled_data[-60:]  # last 60 days

        for day in range(10):
            input_seq_reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])
            predicted_price = model.predict(input_seq_reshaped)[0][0]
            predictions.append(predicted_price)

            # Append predicted price and remove oldest row
            next_row = input_sequence[-1].copy()
            next_row[0] = predicted_price  # update Close price
            input_sequence = np.vstack([input_sequence[1:], next_row])

        # Convert predictions back to original scale
        predictions_inv = scaler.inverse_transform(
            np.hstack([np.array(predictions).reshape(-1,1), np.zeros((10, scaled_data.shape[1]-1))])
        )[:,0]

        # Display predictions
        days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime('%Y-%m-%d').tolist()
        prediction_df = pd.DataFrame({'Date': days, 'Predicted Price': predictions_inv})
        st.write("📊 Predicted Prices for Next 10 Days:")
        st.table(prediction_df)

        # Plot predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=prediction_df['Date'],
            y=prediction_df['Predicted Price'],
            mode='lines+markers',
            name='Predicted Prices'
        ))
        fig_future.update_layout(title=f"10-Day Price Prediction for {selected_stock}",
                                 xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark")
        st.plotly_chart(fig_future)

    except Exception as e:
        st.error(f"Error: {e}")
