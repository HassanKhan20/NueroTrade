import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

def prepare_data(df, days=7):
    df = df[['Close']].copy()
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(len(df) - 60 - days):
        X.append(df['Close'].iloc[i:i+60].values)
        y.append(df['Close'].iloc[i+60:i+60+days].values)

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def build_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_stock(df, days=7):
    X, y, scaler = prepare_data(df, days)
    model = build_model((60, 1), days)
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    last_60 = df['Close'].iloc[-60:].values.reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60).reshape(1, 60, 1)
    forecast = model.predict(last_60_scaled)[0]

    forecast_prices = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1], periods=days + 1, freq='B')[1:]

    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': forecast_prices})
    rmse = mean_squared_error(y[-1], model.predict(X[-1].reshape(1, 60, 1))[0], squared=False)
    return prediction_df, rmse
