#Home Page

import streamlit as st
import pandas as pd
import numpy as np
import logging
from livepredictor.Livepredictor import LivePredictor
import requests
from livepredictor.Wrapper import PySimFin
import plotly.express as px
import plotly.graph_objects as go

#Logging Configuring

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
    logger.setLevel(logging.INFO)
    return logger

logger = configure_global_logging("app.log")


#Function to extract news on the stock

def fetch_stock_news(ticker):
    """Fetches real-time news for the provided stock ticker using the NewsAPI."""
    api_key = "5c6a706aeb344e39b88c0e6f43eda95e"  
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


#Strategy Functions

#Buy and sell function

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
    #st.write(f"Initial Cash: ${initial_cash:,.2f}")
    #st.write(f"Final Portfolio Value: ${final_value:,.2f}")
    #st.write(f"Profit: ${final_value - initial_cash:,.2f}")

    return final_value


#Buy and hold strategy function

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

    #st.write(f"Initial Cash: ${initial_cash:,.2f}")
    #st.write(f"Final Portfolio Value: ${final_value:,.2f}")
    #st.write(f"Profit: ${profit:,.2f}")
    
    #if profit >= profit_target:
      #  st.success(f"Congratulations! You have exceeded your profit target of ${profit_target:,.2f}.")
    #else:
      #  st.warning(f"You did not reach your profit target. Your profit was ${profit:,.2f}, which is ${profit_target - profit:,.2f} below the goal.")

    return final_value 

#Define pages content

#Page 1: Home Page with overview and objectives
def page1():

    st.write("") #Space

    col_overview, col_space, col_mission = st.columns([1.5, 0.2, 1.5])

    with col_overview:
        st.subheader("📈 :orange[Company overview]")
        content_overview = """
        *Neuromarket, founded in 2025*  
            
        Neuromarket is a cutting-edge fintech company specializing in predicting market movements for US companies using sophisticated ML algorithms and real-time data analysis.  

        Our automated daily trading system combines a robust **Data Analytics Module** with a user-friendly **web-based application**, providing actionable insights for informed trading decisions.
        """
        st.markdown(content_overview)

    with col_mission:
        st.subheader("🚀 :orange[Mission]")
    
        content_mission = """
        To **revolutionize** daily trading through an **automated system** powered by advanced **machine learning algorithms** and **real-time financial data analysis**.
        """
        st.markdown(content_mission)

        st.subheader("🎯 :orange[Vision]")

        content_vision = """
        To become the **leading AI-driven fintech platform**, empowering traders and investors with **accurate, real-time market insights**, fostering **financial growth** through **cutting-edge technology**.
        """
        st.markdown(content_vision)

    st.write("") #Space
    st.write("") #Space
    st.subheader("🏆 :orange[About the team]")
    st.write("") #Space
    col1,col2,col3, col4, col5 = st.columns(5)

    with col1:
        st.image("Aviv.png")
        
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 24px; font-weight: bold;">Aviv Rabinowitz</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 20px; color: gray;">
                    <em>Data Strategy Director</em>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 16px; color: gray;">Leads the acquisition, management, and integration of diverse financial datasets, ensuring the company has the most relevant and up-to-date information for predictions. Oversees the ETL process and data pipeline, crucial for feeding accurate data into the ML models.
            </span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("Mohammed.png")
     
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 24px; font-weight: bold;">Mohammed Alharkan</span>
            </div>
        """, unsafe_allow_html=True)
       
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 20px; color: gray;">
                    <em>Chief Prediction Officer</em>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 16px; color: gray;">Oversees the development and implementation of predictive models, ensuring the company’s forecasting capabilities remain cutting-edge and accurate. Leads the overall strategy for market prediction and machine learning initiatives.
            </span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.image("Guille.png")
        
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 24px; font-weight: bold;">Guillermo Medina</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 20px; color: gray;">
                    <em>Machine Learning Architect</em>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 16px; color: gray;">Designs and builds sophisticated ML algorithms to analyze market trends and predict stock movements, focusing on scalability and performance. Responsible for feature selection, model optimization, and integration of ML models into the trading system.
            </span>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.image("Fatine.png")

        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 24px; font-weight: bold;">Fatine Samodi</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 20px; color: gray;">
                    <em>UX/UI Designer</em>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 16px; color: gray;">Creates an intuitive and engaging user interface for the web application, applying neuromarketing principles to optimize user experience. Focuses on layout, color psychology, and interactive elements to enhance user engagement and drive conversions.
            </span>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.image("Camila.png")
        
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 24px; font-weight: bold;">Camila Sanabria</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 20px; color: gray;">
                    <em>Web Application Architect</em>
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
            <div style="display: flex; justify-content: center; text-align: center;">
                <span style="font-size: 16px; color: gray;">Designs the overall structure of the web application, ensuring seamless integration of all components including the ETL pipeline, ML models, and user interface. Responsible for system scalability, performance, and data flow between different modules.
            </span>
            </div>
        """, unsafe_allow_html=True)



#Page 2: Go Live Page
def page2():

    # Add a Section Description
    st.markdown("""
        ## 🚀 **Make Smarter Trading Decisions**
       
        Welcome to the **Go Live** section, where you can access **real-time stock insights** to help you make **smarter investment decisions**.  
        Select your stock, choose a date range, and get **data-driven market predictions** to guide your next move.

        Whether you're an experienced trader or just starting out, this tool helps you:  

        - 📈 **Analyze trends**  
        - 🏅 **Track performance**  
        - 🧐 **Gain insights before making your next trade**  
                
     """, unsafe_allow_html=True)

    # Add Step-by-Step Instructions
    with st.expander("📌 **How to Use This Section**", expanded=False):
        st.markdown("""
        **1️⃣ Select a Stock Ticker**  
        Choose the stock you want to analyze from the dropdown list.

        **2️⃣ Choose a Date Range**  
        - Select a **start date** to define the beginning of the historical data analysis.  
        - Select an **end date** to specify the last day of data.

        **3️⃣ Click "Fetch Data"**  
        The system will retrieve stock data, analyze past movements, and predict the next trading session.

        **4️⃣ View Predictions & Trading Signals**  
        - The model will forecast whether the stock price will **increase or decrease**.  
        - A recommendation will be displayed on whether to **BUY or SELL** based on the prediction.

        **5️⃣ Explore Historical Data and Recent News**  
        - You can view past stock price trends, open/close prices, percentage changes, and news on the stock.
        """)

    st.markdown("---")
    #User Input Parameters Section

        #Title
    st.markdown("""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined">
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="material-symbols-outlined" style="color: #E67205; font-size: 28px; margin-right: 6px;">
                edit
            </span>
            <span style="color: #E67205; font-size: 24px;">Set Prediction Criteria</span>
        </div>
    """, unsafe_allow_html=True)

    #Ticker selection
    tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]  # Example list of stocks


    #Parameters into columns
    col1, col2, col3, col4= st.columns([2, 2, 2, 1])

    with col1:
        selected_ticker = st.selectbox("Select a Stock Ticker :material/finance_chip:", tickers)

    with col2:
        start_date = st.date_input("Select Start Date :material/today:")

    with col3:
        end_date = st.date_input("Select End Date :material/event:")
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        button_fetch=st.button("Fetch Data", use_container_width=True)

    st.markdown("---")

    #Information retrieval after pressing the button
    if button_fetch:
        with st.spinner("Fetching data..."):
                
                # Initialize the LivePredictor
                predictor = LivePredictor(
                    api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b",
                    logger=logger,
                )
                                
                prediction = predictor.predict_next_day(selected_ticker, start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))
                if prediction is not None:
                    
                        # Basic Prediction (last day)

                    last_prediction = "Increase" if prediction[-1] == 1 else "Decrease"

                    #Title
                    st.markdown("""
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="material-symbols-outlined" style="color: #E67205; font-size: 28px;">
                                    online_prediction
                                </span>
                                <h3 style="color: #E67205; margin: 0; font-size: 22px; font-weight: normal;">
                                    Prediction
                                </h3>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                            <div style="
                                background-color: #F5F5F5; /* Very Light Gray */
                                padding: 15px;
                                border-radius: 10px;
                                margin-top: 5px;
                            ">
                                <p style="margin-top: 10px; font-size: 16px;">
                                    Predicted movement on the next market day of 📅 <b>{end_date}</b>:
                                </p>
                                <p style="
                                    background-color: #EAEAEA; /* Slightly Darker Gray for Emphasis */
                                    padding: 10px;
                                    border-radius: 8px;
                                    font-size: 16px;
                                    text-align: center;
                                ">
                                    The stock will most likely <b>{'Increase 🔼' if last_prediction == "Increase" else 'Decrease 🔻'}</b>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    if last_prediction == "Increase":
                        st.markdown("""
                            <div style="background-color: #E6F4EA; padding: 10px; border-radius: 8px;align-items: center;">
                                <b>🎉 We recommend you BUY the Stock 🎉</b>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="background-color: #FDEDED; padding: 10px; border-radius: 8px;align-items: center;">
                                <b>💸 We recommend you SELL the Stock 💸</b>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")    
                #else:
                    #st.warning("No data available for the selected date range.")
                
                #Historical data section
                try:
                    #Retrieve live historical data
                    py = PySimFin("339da715-7249-4c7b-9e0e-a30eef1fdf6b", configure_global_logging("temp.log"))
                    stock_data = py.get_share_prices(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

                    if not stock_data.empty:
                        
                        st.markdown(f"""
                                <div style="display: flex; align-items: center;">
                                    <span class="material-symbols-outlined" style="color: #E67205; font-size: 28px; margin-right: 6px;">
                                        history
                                    </span>
                                    <span style="color: #E67205; font-size: 24px;">Historical Stock Data for&nbsp;</span>
                                    <span style="color: #E67205; font-size: 24px;"> {selected_ticker}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.write("") #Space
                        st.write("") #Space

                        # Calculate Change Percentage & Determine Color
                        if len(stock_data) > 1:
                            change_value = round(((stock_data.iloc[-1, 3] / stock_data.iloc[-2, 3]) - 1) * 100, 2)
                            change_color = "#4CAF50" if change_value > 0 else "#E57373"  # Green for positive, Red for negative
                        else:
                            change_value = "N/A"
                            change_color = "#000000"

                        # Create Columns with Spacing
                        col_open, col_space1, col_close, col_space2, col_change, col_space3, col_high, col_space4, col_low = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1])

                        # Define Styling for Value Boxes (Lighter Gray)
                        box_style = "background-color: #F5F5F5; padding: 10px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: normal; color: black;"

                        # Define Styling for Centered Titles
                        title_style = "text-align: center; font-size: 16px; font-weight: bold; color: black;"

                        # Open
                        with col_open:
                            st.markdown(f'<p style="{title_style}">Open</p>', unsafe_allow_html=True)
                            st.markdown(f'<div style="{box_style}">{stock_data.iloc[-1, 7]:,.2f}</div>', unsafe_allow_html=True)

                        # Close
                        with col_close:
                            st.markdown(f'<p style="{title_style}">Close</p>', unsafe_allow_html=True)
                            st.markdown(f'<div style="{box_style}">{stock_data.iloc[-1, 3]:,.2f}</div>', unsafe_allow_html=True)

                        # Change (Dynamic Color)
                        with col_change:
                            st.markdown(f'<p style="{title_style}">Change</p>', unsafe_allow_html=True)
                            st.markdown(f'<div style="{box_style}; color: {change_color};">{change_value}%</div>', unsafe_allow_html=True)

                        # High
                        with col_high:
                            st.markdown(f'<p style="{title_style}">High</p>', unsafe_allow_html=True)
                            st.markdown(f'<div style="{box_style}">{stock_data.iloc[-1, 5]:,.2f}</div>', unsafe_allow_html=True)

                        # Low
                        with col_low:
                            st.markdown(f'<p style="{title_style}">Low</p>', unsafe_allow_html=True)
                            st.markdown(f'<div style="{box_style}">{stock_data.iloc[-1, 6]:,.2f}</div>', unsafe_allow_html=True)
                    
                        st.write("") #Space between rows
                        st.write("") #Space between rows

                        #Columns for graph and table
                        col1, colspace ,col2 = st.columns([1.5, 0.2, 1.5])

                        # Convert 'Date' to datetime format
                        stock_data["Date"] = pd.to_datetime(stock_data["Date"])

                        with col1:
                            # Interactive Plotly chart
                            fig = px.line(stock_data, x="Date", y="Last Closing Price", 
                                        title=f"{selected_ticker} Stock Price Trend",
                                        labels={"Date": "Date", "Last Closing Price": "Price (USD)"},
                                        template="plotly_white")
                            
                            # Remove gridlines and improve aesthetics
                            fig.update_xaxes(
                                showgrid=False,  # Removes vertical gridlines
                                rangeslider_visible=True,  # Adds a range slider for better interaction
                                title_font=dict(size=14, family="Arial", color="black"),  # Makes axis titles clearer
                                tickfont=dict(size=12, family="Arial", color="black")
                            )

                            fig.update_yaxes(
                                showgrid=False,  # Removes horizontal gridlines
                                title_font=dict(size=14, family="Arial", color="black"),  # Custom font for Y-axis
                                tickfont=dict(size=12, family="Arial", color="black")
                            )

                            # Enhance overall layout
                            fig.update_layout(
                                title=dict(
                                    text=f"{selected_ticker} Stock Price Trend",
                                    font=dict(size=18, family="Arial"), 
                                    x=0,  # Centers the title
                                ),
                                plot_bgcolor="white",  # Keeps background clean
                                margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins for a compact layout
                                hovermode="x unified"  # Improves hover interaction
                            )

                            st.plotly_chart(fig)


                        #Table Display in Streamlit
                        with col2:
                            st.markdown("##### 📊 Latest Stock Data")  # Smaller title
                            st.write(stock_data.tail(10))

                        # Get Financial Statements
                        if "financial_data" not in st.session_state:
                            st.session_state.financial_data = None  # Initialize empty

                        # Automatically Fetch Financial Statements When the User Expands the Section
                        with st.expander("📑 View Financial Statements", expanded=False):  
                            with st.spinner("Fetching financial data..."):
                                try:
                                    
                                    # Fetch financial statements and store in session state
                                    st.session_state.financial_data = py.get_financial_statement(
                                        selected_ticker, "pl", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                                    )

                                    if not st.session_state.financial_data.empty:
                                        st.success(f"✅ Financial data retrieved for {selected_ticker}")
                                        st.dataframe(st.session_state.financial_data, use_container_width=True)
                                    else:
                                        st.warning(f"⚠️ No financial data found for {selected_ticker} in this date range.")
                                
                                except Exception as e:
                                    st.error(f"❌ Error retrieving financial statements: {e}")

                        st.write("")
                        st.write("")
                        st.markdown("---")

                        # Fetch and display stock news
                        #Title
                        st.markdown(f"""
                                <div style="display: flex; align-items: center;">
                                    <span style="color: #E67205; font-size: 24px;">📢  Real-time News for&nbsp;</span>
                                    <span style="color: #E67205; font-size: 24px;"> {selected_ticker}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        #News display
                        news_articles = fetch_stock_news(selected_ticker)[:4]
                        if news_articles:
                            num_columns = 2
                            cols = st.columns(num_columns)

                            for index, article in enumerate(news_articles):
                                with cols[index % num_columns]:  # Distribute across columns
                                    st.markdown(
                                        f"""
                                        <div style="
                                            border: 1px solid #ddd;
                                            border-radius: 10px;
                                            padding: 15px;
                                            margin: 10px 0;
                                            background-color: #f9f9f9;
                                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                                        ">
                                            <h4 style="color: #333;">{article['title']}</h4>
                                            <p style="color: #555;">{article['description']}</p>
                                            <a href="{article['url']}" target="_blank" style="
                                                color: white;
                                                text-decoration: none;
                                                background-color: #FF7829;
                                                padding: 8px 12px;
                                                border-radius: 5px;
                                                display: inline-block;
                                                font-weight: bold;
                                            ">Read more</a>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.warning("No news available for this ticker.")
                    
                    else:
                        st.warning("No data available for the selected date range.")
                
                except Exception as e:
                    st.error(f"Error fetching stock data: {e}")

#Page 3: Strategy Page
def page3():

    # Add a Section Description
    st.markdown("""
        ## 💡 **Build Your Trading Strategy**
        
        Take control of your investments with a **custom trading strategy** that fits your risk level and market outlook.  
        Choose how you want to **buy, sell, or hold assets** based on market conditions, and let the system analyze your potential returns.


        Whether you're a long-term investor or an active trader, this tool helps you:
   
        - 📈 **Buy when the market shows strength.**  
        - 🔻 **Sell when the market weakens.**  
        - 🏆 **Hold to maximize long-term gains.**  

    """, unsafe_allow_html=True)

    # Add Step-by-Step Instructions
    with st.expander("📌 **How to Create a Trading Strategy**", expanded=False):
        st.markdown("""
        **1️⃣ Select Your Trading Strategy**  
        - **Buy-and-Hold** → Hold stocks for the long run, selling only when a profit target is reached.  
        - **Buy-and-Sell** → Actively trade based on short-term price movements.  

        **2️⃣ Define Your Trading Rules**  
        - Set conditions for **when to buy, sell, or hold.**  
        - Choose a **profit target** if you're using Buy-and-Hold.  

        **3️⃣ Backtest Your Strategy**  
        - Run your strategy against historical stock data.  
        - See how it would have performed in real market conditions.  

        **4️⃣ Get a Performance Summary**  
        - View potential **profits or losses** based on your chosen approach.  
        - Receive a recommendation on whether to **proceed with this strategy.**  
        """)


    st.markdown("---") 

    # Input title

    st.markdown("""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined">
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="material-symbols-outlined" style="color: #E67205; font-size: 28px; margin-right: 6px;">
                edit
            </span>
            <span style="color: #E67205; font-size: 24px;">Input Data</span>
        </div>
    """, unsafe_allow_html=True)

    tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]

    #Input

    col1, col2, col3, col4, space, col5  = st.columns([2, 2, 2, 2, 0.5, 2])

    with col1:
        selected_ticker = st.selectbox("Select a Stock Ticker :material/finance_chip:", tickers)

    with col2:
        start_date = st.date_input("Select Start Date :material/today:")

    with col3:
        end_date = st.date_input("Select End Date :material/event:")
    
    with col4:
        initial_cash = st.number_input("Initial Cash ($)", min_value=1000, value=10000, step=1000)

    with col5:
        strategy = st.radio("Select Strategy", ["Buy-and-Sell", "Buy-and-Hold"])

    if strategy == "Buy-and-Hold":
        profit_target = st.number_input("Profit Target ($)", min_value=10, value=100, step=10)

    if st.button("Fetch Data",use_container_width=True):
        with st.spinner("Fetching data..."):
            predictor = LivePredictor(api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b", logger=logger)
            stock_data = predictor.fetch_data(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            prediction = predictor.predict_next_day(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            if prediction is not None:
                if strategy == "Buy-and-Sell":
                    final_value = buy_and_sell(stock_data, prediction, initial_cash)
                else:
                    final_value = buy_and_hold_strategy(stock_data, initial_cash, profit_target)
                # Calculate profit/loss
                profit_or_loss = final_value - initial_cash
                profit_color = "#E57373" if profit_or_loss < 0 else "#81C784"  # Red if loss, Green if profit
                result_word = "profit" if profit_or_loss > 0 else "loss"  # Switch between "profit" and "loss"
                recommendation = f"If you play this strategy, you will have a **{result_word} of ${abs(profit_or_loss):,.2f}**."

                st.markdown("---") 

                # Display the results in three equal columns
                col1, col2, col3 = st.columns(3)
                
                # Initial Cash Box
                with col1:
                    st.markdown("""
                        <div style="  
                            padding: 15px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 16px;
                            font-weight: bold;">
                            Initial Cash
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="
                            background-color: #E0E0E0;
                            padding: 12px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 20px;">
                            ${initial_cash:,.2f}
                        </div>
                    """, unsafe_allow_html=True)

                # Final Portfolio Value Box
                with col2:
                    st.markdown("""
                        <div style="
                            padding: 15px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 16px;
                            font-weight: bold;">
                            Final Portfolio Value
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="
                            background-color: #E0E0E0;
                            padding: 12px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 20px;">
                            ${final_value:,.2f}
                        </div>
                    """, unsafe_allow_html=True)

                # Profit or Loss Box (Color Changes Dynamically)
                with col3:
                    st.markdown("""
                        <div style="
                            padding: 15px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 16px;
                            font-weight: bold;">
                            Profit or Loss
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="
                            background-color: {profit_color};
                            padding: 12px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 20px;">
                            ${profit_or_loss:,.2f}
                        </div>
                    """, unsafe_allow_html=True)

                st.write(" ")

                if profit_or_loss > 0:
                    st.success(f"🟢 {recommendation} We **recommend** you do this! 🚀")
                else:
                    st.error(f"🔴 {recommendation} We **do NOT recommend** doing this. ⚠️")
                    
            else:
                st.error("No data available for the selected date range. Please try again.")
            




#Make a list of the pages and add a top bar menu
pages = [
    st.Page(page1, icon=":material/home:", title="Home"),
    st.Page(page2, icon=":material/live_tv:", title="Go Live"),
    st.Page(page3, icon=":material/chess:", title="Strategize")
]

current_page = st.navigation(pages=pages, position="hidden")

st.set_page_config(layout="wide")

num_cols = max(len(pages) + 1, 8)

columns = st.columns(num_cols, vertical_alignment="bottom")

#Add logo
columns[0].image("Full_logo.png", width=500)

for col, page in zip(columns[1:], pages):
    col.page_link(page, icon=page.icon)

st.title(f"{current_page.icon} {current_page.title}")

current_page.run()
