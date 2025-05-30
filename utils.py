import pandas as pd
import plotly.graph_objs as go
import requests
import os

API_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")

def fetch_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={API_KEY}&datatype=csv"
    try:
        df = pd.read_csv(url)
        df = df.rename(columns={'timestamp': 'Date', 'close': 'Close'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        return df[['Close']]
    except Exception as e:
        return pd.DataFrame()

def plot_predictions(original_df, prediction_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_df.index, y=original_df['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Close'], mode='lines+markers', name='Predicted'))
    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    return fig
