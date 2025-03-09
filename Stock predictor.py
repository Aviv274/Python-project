import streamlit as st
import logging
from livepredictor.Livepredictor import LivePredictor

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

start_date = st.sidebar.date_input("Select Start Date")
end_date = st.sidebar.date_input("Select End Date")

if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        try:
            # Initialize the LivePredictor
            predictor = LivePredictor(
                api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b",
                logger=logger,
            )
                        
            prediction = predictor.predict_next_day(selected_ticker, start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))
            if prediction is not None:
                # Basic Prediction (last day)
                st.subheader("Basic Prediction (Last Day)")
                last_prediction = "Increase" if prediction[-1] == 1 else "Decrease"
                st.write(f"Predicted movement on the next market day of {end_date}: {last_prediction}")
            
                if last_prediction == "Increase":
                    st.success("🔼 BUY Signal")
                else:
                    st.error("🔽 SELL Signal")
            else:
                st.warning("No data available for the selected date range.")
        
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")

