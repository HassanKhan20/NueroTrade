import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from model import predict_stock
from utils import fetch_data, plot_predictions

st.set_page_config(page_title="InsightTrader - AI Stock Forecast", layout="wide")
st.title("📈 NueroTrade: AI-Powered Stock Forecast Dashboard")

st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose Data Source:", ("Upload CSV", "Fetch from Ticker"))

if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with 'Date' and 'Close' columns")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
    if ticker:
        df = fetch_data(ticker)

if 'df' in locals() and df is not None and not df.empty:
    st.subheader("📊 Stock Price Data Preview")
    st.dataframe(df.tail())

    st.subheader("📉 Model Prediction")
    days = st.slider("Days to Predict", 3, 30, 7)
    try:
        prediction_df, rmse = predict_stock(df, days)
        st.plotly_chart(plot_predictions(df, prediction_df))
        st.markdown(f"### ✅ RMSE: {rmse:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a CSV or enter a valid stock ticker to begin.")
