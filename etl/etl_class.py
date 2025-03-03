import pandas as pd
import os
import json
import simfin as sf
from simfin.names import *
import logging


# Set your SimFin API Key (Replace with your actual API key)
sf.set_api_key('339da715-7249-4c7b-9e0e-a30eef1fdf6b')

# Set local data directory (for caching)
sf.set_data_dir('etl/simfin_data/')

class StockETL:
    """ETL pipeline for processing stock financial data."""
    

    def __init__(self,config_path="etl/config.json", logger=None):
        if logger is None:
            raise ValueError("Logger is mandatory. Please provide a valid logger instance.")

        self.logger = logger
        self.config_path = config_path
        self.config = self.load_config()
        self.data_folder = str(self.config.get("data_folder"))
        self.clean_folder = str(self.config.get("clean_folder"))
        os.makedirs(self.clean_folder, exist_ok=True)
        self.logger.info("StockETL instance initialized.")

    def load_config(self):
        """Loads the configuration file."""
        if not os.path.exists(self.config_path):
            self.logger.warning("Config file not found! Using default settings.")
            return {}

        try:
            with open(self.config_path, 'r') as file:
                return json.load(file) or {}
        except json.JSONDecodeError:
            self.logger.error("Failed to load config.json. Using default settings.")
            return {}

    def load_data(self, ticker):
        """Loads all datasets related to a stock ticker."""
        files = ["share_prices", "income_statement", "balance_sheet", "cash_flow","company"]
        try:
            return [pd.read_csv(os.path.join(self.data_folder, f"{ticker}_{file}.csv")) for file in files]
        except FileNotFoundError:
            self.logger.warning(f"Missing files for {ticker}. Skipping.")
        except pd.errors.EmptyDataError:
            self.logger.warning(f"Empty CSV files for {ticker}. Skipping.")
        except Exception:
            self.logger.exception(f"Error loading data for {ticker}.")
        return [None] * 4

    def clean_data(self, df, column_methods):
        """Cleans a dataset based on configuration rules."""
        if df is None:
            return None
        try:
            for col, method in column_methods.items():
                if col in df.columns:
                    df[col] = self.apply_cleaning_method(df[col], method)

            df.dropna(axis=1, how="all", inplace=True)
            return df.drop_duplicates()
        except Exception:
            self.logger.exception("Error cleaning data.")
            return df

    def apply_cleaning_method(self, column, method):
        """Applies a specific cleaning method to a column."""
        if method == "fill":
            column=column.ffill()
            return column.bfill()
        elif method == "drop":
            return None
        elif method == "zero":
            return column.fillna(0)
        return column

    def merge_data(self, df_prices, df_income, df_balance, df_cashflow, df_company, ticker):
        """Merges all stock datasets and forward-fills financial data."""
        if df_prices is None:
            self.logger.warning(f"Skipping {ticker} due to missing price data.")
            return None

        try:
            df_prices['Year'] = pd.to_datetime(df_prices['Date']).dt.year
            merged = df_prices.copy()

            for df in [df_income, df_balance, df_cashflow]:
                if df is not None:
                    merged = merged.merge(df, left_on="Year", right_on="Fiscal Year", how="left")

            merged.sort_values(by="Date", inplace=True)
            merged.ffill(inplace=True)
            merged.drop(columns=["Year"], inplace=True, errors="ignore")
            merged = merged.merge(df_company, how='cross')

            return self.clean_merged_columns(merged)
        except Exception:
            self.logger.exception(f"Error merging data for {ticker}. Skipping.")
            return None

    def clean_merged_columns(self, df):
        """Removes redundant or unwanted columns from the merged dataset."""
        try:
            drop_cols = self.config.get("drop_columns", [])
            keep_cols = set(self.config.get("keep_columns", []))

            df.drop(columns=drop_cols, inplace=True, errors="ignore")

            redundant_cols = [
                col for col in df.columns
                if ("_x" in col or "_y" in col) and col.replace("_x", "").replace("_y", "") not in keep_cols
            ]
            df.drop(columns=redundant_cols, inplace=True, errors="ignore")

            return df
        except Exception:
            self.logger.exception("Error cleaning merged columns.")
            return df

    def save_merged_data(self, df, ticker):
        """Saves the final merged dataset."""
        if df is None:
            self.logger.warning(f"Skipping {ticker} due to missing data.")
            return

        try:
            df.to_csv(os.path.join(self.clean_folder, f"{ticker}_merged_data.csv"), index=False)
            self.logger.info(f"Saved {ticker} merged dataset.")
        except Exception:
            self.logger.exception(f"Error saving data for {ticker}.")

    def save_data(self,df, filename):
        output_folder = "etl/stock_data"
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved: {filepath}")
        

    def download_data(self):
        """Downloads share prices and financial statements for configured tickers."""
        tickers = self.config.get("tickers", [])
        self.logger.info(f"Downloading data for tickers: {tickers}")
        
        try:
            df_prices = sf.load_shareprices(variant='daily', market='us',index=None)
            df_income = sf.load_income(variant='annual', market='us',index=None)
            df_balance = sf.load_balance(variant='annual', market='us',index=None)
            df_cashflow = sf.load_cashflow(variant='annual', market='us', index=None)
            df_company = sf.load_companies(market="us",index=None)
            #Need to add compenies
            for ticker in tickers:
                df_ticker = df_prices[df_prices[TICKER] == ticker]
                self.save_data(df_ticker, f"{ticker}_share_prices.csv")
                df_ticker = df_income[df_income[TICKER] == ticker]
                self.save_data(df_ticker, f"{ticker}_income_statement.csv")
                df_ticker = df_balance[df_balance[TICKER] == ticker]
                self.save_data(df_ticker, f"{ticker}_balance_sheet.csv")
                df_ticker = df_cashflow[df_cashflow[TICKER] == ticker]
                self.save_data(df_ticker, f"{ticker}_cash_flow.csv")
                df_ticker = df_company[df_company[TICKER] == ticker]
                self.save_data(df_ticker, f"{ticker}_company.csv")
                
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")


def configure_global_logging():
    """Sets up logging for the main process."""
    logging.basicConfig(
        filename="etl/etl.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def main():
    """Runs the ETL process for multiple stocks."""
    logger = configure_global_logging()
    logger.info("Starting ETL process.")

    
    etl = StockETL(logger=logger)
    tickers = etl.config.get("tickers",[])
    price_methods = etl.config.get("share_prices_cleaning_methods", {})
    income_methods = etl.config.get("income_cleaning_methods", {})
    balance_methods = etl.config.get("balance_cleaning_methods", {})
    cashflow_methods = etl.config.get("cashflow_cleaning_methods", {})
    company_methods = etl.config.get("company_cleaning_methods", {})
    etl.download_data()
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")

        df_prices, df_income, df_balance, df_cashflow, df_company = etl.load_data(ticker)
        if df_prices is None:
            logger.warning(f"Skipping {ticker} due to missing essential data.")
            continue

        df_prices = etl.clean_data(df_prices, price_methods)
        df_income = etl.clean_data(df_income, income_methods)
        df_balance = etl.clean_data(df_balance, balance_methods)
        df_cashflow = etl.clean_data(df_cashflow, cashflow_methods)
        df_company = etl.clean_data(df_company,company_methods)

        df_merged = etl.merge_data(df_prices, df_income, df_balance, df_cashflow,df_company, ticker)
        if df_merged is None:
            logger.warning(f"Skipping {ticker} due to failed merge.")
            continue

        etl.save_merged_data(df_merged, ticker)
        logger.info(f"Successfully processed {ticker}")

    logger.info("ETL process completed successfully!")


if __name__ == "__main__":
    main()

    
