# InsightTrader: AI-Powered Stock Forecasting Dashboard

InsightTrader is a machine learning-based stock price prediction tool built with Python, TensorFlow, and Streamlit. It allows users to upload historical stock data or fetch it using a ticker symbol, visualize trends, and forecast future prices using an LSTM model.

##  Features
- Upload custom stock data (CSV with 'Close' prices)
- Fetch real-time data using Alpha Vantage API
- Predict next 3–30 business days of stock prices
- Interactive visualizations with Plotly
- Evaluation metric: RMSE (Root Mean Squared Error)

##  Technologies Used
- **Frontend:** Streamlit
- **ML Model:** LSTM (TensorFlow/Keras)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Plotly
- **APIs:** Alpha Vantage (optional)

##  How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
