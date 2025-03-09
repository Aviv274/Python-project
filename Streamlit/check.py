from Livepredictor import LivePredictor
import logging
import pandas_market_calendars as mcal
from datetime import datetime, timedelta

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


def get_last_open_market_date(logger, date: str = None):
        """
        Checks if the US stock market was open on the given date.
        If closed, returns the last open trading day (at max, yesterday).
        Logs the process.

        :param date: Datetime object (default is today)
        :return: Last open market date (YYYY-MM-DD)
        """
        if date is None:
            date = datetime.now()
        else:
            date = datetime.strptime(date, "%Y-%m-%d")
        # Get the NYSE trading calendar
        nyse = mcal.get_calendar('NYSE')

        # Define the latest possible start date (yesterday)
        yesterday = date - timedelta(days=1)

        # Get market schedule for the last two days
        schedule = nyse.schedule(start_date=yesterday.strftime("%Y-%m-%d"), 
                                end_date=date.strftime("%Y-%m-%d"))

        if date.strftime("%Y-%m-%d") in schedule.index:
            logger.info(f"Market was open on {date.strftime('%Y-%m-%d')}.")
            return date.strftime("%Y-%m-%d")
        
        # If today is closed, find the last open day (max: yesterday)
        last_open_date = schedule.index[-1] if not schedule.empty else yesterday.strftime("%Y-%m-%d")
        
        logger.warning(f"Market was closed on {date.strftime('%Y-%m-%d')}. Last open day: {last_open_date}.")
        return last_open_date.strftime("%Y-%m-%d")


if __name__ == "__main__":
    # Initialize logger
    logger = configure_global_logging("Streamlit/livepredictor.log")

        # Test the predictor with a sample date
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "NVDA"]

    for test_ticker in tickers:
        # Initialize predictor
        predictor = LivePredictor(
            api_key="339da715-7249-4c7b-9e0e-a30eef1fdf6b",
            model_path=f"ml/models/{test_ticker}_logistic_regression_model.pkl",
            scaler_path=f"ml/scalers/{test_ticker}_logistic_scaler.pkl",
            feature_path=f"ml/features/{test_ticker}_selected_features.json",
            logger=logger,
            
        )


        start_date = "2023-01-01"
        end_date = "2025-12-31"
        #test_date = get_last_open_market_date(logger,test_date)
        prediction_result = predictor.predict_next_day(test_ticker, start_date, end_date)
        print(f"Prediction for {test_ticker} on the next day for {end_date}: {prediction_result}")