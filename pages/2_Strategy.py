import streamlit as st
import logging
from livepredictor.Livepredictor import LivePredictor
from livepredictor.Wrapper import PySimFin

def buy_and_sell(df, predictions, initial_cash):
    """Backtest Strategy: Buy on positive prediction, sell on negative."""
    cash = initial_cash
    holdings = 0  
    df['predictions'] = predictions

    for i in range(len(df)):
        if df["predictions"].iloc[i] == 1:  # Buy signal
            shares_to_buy = cash // df["Close"].iloc[i]  # Buy as many shares as possible
            holdings += shares_to_buy
            cash -= shares_to_buy * df["Close"].iloc[i]
        elif df["predictions"].iloc[i] == 0 and holdings > 0:  # Sell signal
            cash += holdings * df["Close"].iloc[i]
            holdings = 0

    final_value = cash + (holdings * df["Close"].iloc[-1])
    st.write(f"Initial Cash: ${initial_cash:,.2f}")
    st.write(f"Final Portfolio Value: ${final_value:,.2f}")
    st.write(f"Profit: ${final_value - initial_cash:,.2f}")


def buy_and_hold_strategy(df, initial_cash, profit_target):
    """Strategy 1: Buy-and-Hold with Profit Target"""
    cash = initial_cash
    holdings = 0  
    purchase_price = None

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:  # Buy when price rises
            if cash >= df["Close"].iloc[i]:
                holdings += 1
                cash -= df["Close"].iloc[i]
                if purchase_price is None:
                    purchase_price = df["Close"].iloc[i]
    
    final_value = cash + (holdings * df["Close"].iloc[-1])
    profit = final_value - initial_cash

    st.write(f"Initial Cash: ${initial_cash:,.2f}")
    st.write(f"Final Portfolio Value: ${final_value:,.2f}")
    st.write(f"Profit: ${profit:,.2f}")
    
    if profit >= profit_target:
        st.success(f"Congratulations! You have exceeded your profit target of ${profit_target:,.2f}.")
    else:
        st.warning(f"You did not reach your profit target. Your profit was ${profit:,.2f}, which is ${profit_target - profit:,.2f} below the goal.")


def configure_global_logging(file_path):
    logging.basicConfig(
        filename=file_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

logger = configure_global_logging("app.log")

st.sidebar.header("Stock Selection")
tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker", tickers)
start_date = st.sidebar.date_input("Select Start Date")
end_date = st.sidebar.date_input("Select End Date")
initial_cash = st.sidebar.number_input("Initial Cash ($)", min_value=1000, value=10000, step=1000)

strategy = st.sidebar.radio("Select Strategy", ["Buy-and-Sell", "Buy-and-Hold"])

if strategy == "Buy-and-Hold":
    profit_target = st.sidebar.number_input("Profit Target ($)", min_value=10, value=100, step=10)

if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        predictor = LivePredictor(api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b", logger=logger)
        stock_data = predictor.fetch_data(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        prediction = predictor.predict_next_day(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if prediction is not None:
            if strategy == "Buy-and-Sell":
                buy_and_sell(stock_data, prediction, initial_cash)
            else:
                buy_and_hold_strategy(stock_data, initial_cash, profit_target)
        else:
            st.error("Failed to generate predictions. Please try again.")