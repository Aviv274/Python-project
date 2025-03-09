import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import requests
from livepredictor.Wrapper import PySimFin

# Function to configure logging
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

# Function to fetch stock news
def fetch_stock_news(ticker):
    """Fetches real-time news for the provided stock ticker using the NewsAPI."""
    api_key = "5c6a706aeb344e39b88c0e6f43eda95e"  # Replace with your NewsAPI key
    #ticker = "tesla"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == 'ok':
            articles = data['articles']
            if articles:
                return articles
            else:
                return None
        else:
            st.error("Failed to fetch news.")
            return None
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None

# Function to display stock data and news
def stock_data_page():
    tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]  # Example list of stocks
    st.title("Stock Market Data Viewer")
    
    st.sidebar.header("Stock Data Selection")
    stock_ticker = st.sidebar.selectbox("Choose a Stock Ticker", tickers)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    
    if st.sidebar.button("Load Stock Data"):
        with st.spinner("Fetching stock data..."):
            try:
                py = PySimFin("339da715-7249-4c7b-9e0e-a30eef1fdf6b", configure_global_logging("temp.log"))
                stock_data = py.get_share_prices(stock_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                
                if not stock_data.empty:
                    st.subheader(f"Historical Stock Data for {stock_ticker}")
                    st.write(stock_data.tail(10))
                    
                    # Convert 'Date' to datetime format
                    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

                    # Interactive Plotly chart
                    fig = px.line(stock_data, x="Date", y="Last Closing Price", 
                                  title=f"{stock_ticker} Stock Price Trend",
                                  labels={"Date": "Date", "Last Closing Price": "Price (USD)"},
                                  template="plotly_white")
                    
                    fig.update_xaxes(rangeslider_visible=True)  # Add range slider for better interaction
                    fig.update_layout(xaxis=dict(rangeslider=dict(
                    bgcolor='lightgray',
                    borderwidth=2)))  # Border width of the range slider
                    fig.update_layout(xaxis=dict(title="Date"), yaxis=dict(title="Price"), hovermode="x unified")

                    st.plotly_chart(fig)  # Display Plotly chart
                    
                    # Fetch and display stock news
                    st.subheader(f"Real-time News for {stock_ticker}")
                    news_articles = fetch_stock_news(stock_ticker)
                    if news_articles:
                        for article in news_articles[:5]:  # Display the top 5 news articles
                            st.markdown(f"**{article['title']}**")
                            st.write(article['description'])
                            st.write(f"[Read more]({article['url']})")
                            st.write("")  # Add a blank line between articles
                    else:
                        st.warning("No news available for this ticker.")
                    
                else:
                    st.warning("No data found for this ticker in the selected range.")
                    
            except Exception as e:
                st.error(f"Error fetching stock data: {e}")

# Navigation for Multi-Page App
stock_data_page()
