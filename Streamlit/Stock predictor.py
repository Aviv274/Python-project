import streamlit as st
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import logging
from datetime import datetime
from Livepredictor import LivePredictor

def configure_global_logging(file_path):
    """Sets up logging for the main process using config."""
    logging.basicConfig(
        filename=file_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Ensure logging level is applied
    return logger

# Streamlit App
logger = configure_global_logging("app.log")
st.title("Automated Daily Trading System")

# Sidebar for stock selection
st.sidebar.header("Stock Selection")

tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]  # Example list of stocks
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker", tickers)

selected_date = st.sidebar.date_input("Select a Date")

if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        try:
            # Get the last open market date
            market_date = selected_date.strftime("%Y-%m-%d")
            #st.write(type(market_date))
            #market_date = get_last_open_market_date(logger,market_date)
            
            # Initialize the LivePredictor
            predictor = LivePredictor(
                api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b",
                model_path=f"ml/models/{selected_ticker}_logistic_regression_model.pkl",
                scaler_path=f"ml/scalers/{selected_ticker}_logistic_scaler.pkl",
                feature_path=f"ml/features/{selected_ticker}_selected_features.json",
                logger=logger,
            )
            
            # Make Prediction
            st.subheader("Prediction for Selected Date")
            prediction = predictor.predict_next_day(selected_ticker, market_date)
            st.write(f"Predicted movement on {market_date}: {prediction}")
            
            if prediction == "Increase":
                st.success("🔼 BUY Signal")
            elif prediction == "Decrease":
                st.error("🔽 SELL Signal")
            else:
                st.info("➡️ HOLD")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Run with: `streamlit run your_script.py`
