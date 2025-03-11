import pandas as pd
import os
import json
import simfin as sf
from simfin.names import *
import logging
import os 

# Set your SimFin API Key
sf.set_api_key('339da715-7249-4c7b-9e0e-a30eef1fdf6b')
# Set local data directory (for caching)
sf.set_data_dir('etl/simfin_data/')

class StockETL:
    def __init__(self, config_path: str = "etl/config.json", logger: logging.Logger = None):
        """
        Initializes StockETL with a logger and configuration.

        Args:
            config_path (str): Path to the configuration file.
            logger (logging.Logger): Logger instance.

        Raises:
            ValueError: If no logger instance is provided.
        """
        if logger is None:
            raise ValueError("Logger is mandatory. Please provide a valid logger instance.")

        self.logger = logger
        self.config_path = config_path
        self.config = self.load_config()
        self.validate_config()
        
        # Load configuration settings
        self.data_folder = self.config.get("data_folder", "data")
        self.clean_folder = self.config.get("clean_folder", "clean")
        self.output_folder = self.config.get("output_folder", "output")
        
        # Create output directories if they don't exist
        for folder in [self.data_folder, self.clean_folder, self.output_folder]:
            os.makedirs(folder, exist_ok=True)
        
        self.logger.info("StockETL instance initialized with validated configuration.")

    def validate_config(self) -> None:
        """
        Validates the configuration settings.
        
        Raises:
            ValueError: If required configuration settings are missing or invalid.
        """
        required_settings = ["tickers", "data_folder", "clean_folder", "output_folder"]
        for setting in required_settings:
            if setting not in self.config:
                raise ValueError(f"Missing required configuration setting: {setting}")
        
        if not isinstance(self.config["tickers"], list) or len(self.config["tickers"]) == 0:
            raise ValueError("'tickers' must be a non-empty list")
        
        for folder in ["data_folder", "clean_folder", "output_folder"]:
            if not isinstance(self.config[folder], str):
                raise ValueError(f"'{folder}' must be a string")
        
        # Validate cleaning methods configuration
        cleaning_method_keys = [
            "share_prices_cleaning_methods",
            "income_cleaning_methods",
            "balance_cleaning_methods",
            "cashflow_cleaning_methods",
            "company_cleaning_methods"
        ]
        valid_methods = ["fill", "drop", "zero"]
        for key in cleaning_method_keys:
            methods = self.config.get(key, {})
            if not isinstance(methods, dict):
                raise ValueError(f"'{key}' must be a dictionary")
            for column, method in methods.items():
                if method not in valid_methods:
                    raise ValueError(f"Invalid cleaning method '{method}' for column '{column}' in '{key}'")
        
        self.logger.info("Configuration validated successfully.")

    def load_config(self) -> dict:
        """
        Loads the configuration file.

        Returns:
            dict: Configuration settings, or an empty dictionary if loading fails.
        """
        if not os.path.exists(self.config_path):
            self.logger.warning("Config file not found! Using default settings.")
            return {}

        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            self.logger.error("Failed to load config.json. Using default settings.")
            return {}

    def load_data(self, ticker: str) -> list[pd.DataFrame]:
        """
        Loads share prices, income statement, balance sheet, and cash flow data for a given ticker.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            list: List of pandas DataFrames containing the loaded data, or a list of None values if loading fails.
        """
        files = ["share_prices", "income_statement", "balance_sheet", "cash_flow", "company"]
        try:
            return [pd.read_csv(os.path.join(self.data_folder, f"{ticker}_{file}.csv")) for file in files]
        except FileNotFoundError:
            self.logger.warning(f"Missing files for {ticker}. Skipping.")
        except pd.errors.EmptyDataError:
            self.logger.warning(f"Empty CSV files for {ticker}. Skipping.")
        except Exception:
            self.logger.exception(f"Error loading data for {ticker}.")
        return [None] * 5  # Return a list of 5 None values to match the expected number of datasets

    def clean_data(self, df: pd.DataFrame, column_methods: dict[str, str]) -> pd.DataFrame:
        """
        Cleans a dataset based on the provided cleaning methods.

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.
            column_methods (dict): Dictionary specifying cleaning methods for each column.

        Returns:
            pd.DataFrame: Cleaned DataFrame, or the original DataFrame if cleaning fails.
        """
        if df is None:
            return None
        
        try:
            cols_to_drop = []  # Store columns that should be dropped
            for col, method in column_methods.items():

                if col in df.columns:
                    cleaned_col = self.apply_cleaning_method(df[col], method)
                    if cleaned_col is None:
                        cols_to_drop.append(col)  # Mark column for dropping
                    else:
                        df[col] = cleaned_col  # Assign cleaned column

            # Drop columns that were marked for removal
            df.drop(columns=cols_to_drop, inplace=True)


            # Remove completely empty columns (redundant check after drop)
            df.dropna(axis=1, how="all", inplace=True)

            # Remove duplicate rows
            return df.drop_duplicates()
        
        except Exception:
            self.logger.exception("Error cleaning data.")
            return df

    def apply_cleaning_method(self, column: pd.Series, method: str) -> pd.Series:
        """
        Applies a specific cleaning method to a column.

        Args:
            column (pd.Series): Column to be cleaned.
            method (str): Cleaning method to apply.

        Returns:
            pd.Series: Cleaned column.
        """
        if method == "fill":
            column = column.ffill()
            return column.bfill()
        elif method == "drop":
            return None
        elif method == "zero":
            return column.fillna(0)
        return column

    def merge_data(self, df_prices: pd.DataFrame, df_income: pd.DataFrame, 
                   df_balance: pd.DataFrame, df_cashflow: pd.DataFrame, 
                   df_company: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Merges all stock datasets and forward-fills financial data.

        Args:
            df_prices (pd.DataFrame): DataFrame containing share price data.
            df_income (pd.DataFrame): DataFrame containing income statement data.
            df_balance (pd.DataFrame): DataFrame containing balance sheet data.
            df_cashflow (pd.DataFrame): DataFrame containing cash flow data.
            df_company (pd.DataFrame): DataFrame containing company data.
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: Merged DataFrame, or None if merging fails.
        """
        if df_prices is None:
            self.logger.warning(f"Skipping {ticker} due to missing price data.")
            return None

        try:
            df_prices['Year'] = pd.to_datetime(df_prices['Date']).dt.year
            merged = df_prices.copy()

            for df in [df_income, df_balance, df_cashflow]:
                if df is not None:
                    merged = merged.merge(df, left_on="Year", right_on="Fiscal Year", how="left")

            merged = merged.merge(df_company, how='cross')
            merged = self.clean_merged_columns(merged)
            
            return merged
        except Exception:
            self.logger.exception(f"Error merging data for {ticker}. Skipping.")
            return None

    def clean_merged_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes redundant or unwanted columns from the merged dataset.

        Args:
            df (pd.DataFrame): Merged DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame, or the original DataFrame if cleaning fails.
        """
        try:

            df.sort_values(by="Date", inplace=True)
            # After Merge, forward-fill missing values (NaN) to maintain chronological order and consistency.
            df.ffill(inplace=True)
            # We Dont neet the Year column as we have dates
            df.drop(columns=["Year"], inplace=True, errors="ignore")

            redundant_cols = [
                col for col in df.columns
                if ("_x" in col or "_y" in col) 
            ]
            df.drop(columns=redundant_cols, inplace=True, errors="ignore")

            return df
        except Exception:
            self.logger.exception("Error cleaning merged columns.")
            return df

    def save_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Saves a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to be saved.
            filename (str): Name of the output CSV file.
        """
        if df is None:
            self.logger.warning(f"Skipping {file_path} due to missing data.")
            return

        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved {file_path} merged dataset.")
        except Exception:
            self.logger.exception(f"Error saving data for {file_path}.")

    def download_data(self):
        """
        Downloads share prices, income statements, balance sheets, cash flows, and company information
        for the configured tickers.
        """
        tickers = self.config.get("tickers", [])
        self.logger.info(f"Downloading data for tickers: {tickers}")

        try:
            df_prices = sf.load_shareprices(variant='daily', market='us', index=None)
            df_income = sf.load_income(variant='annual', market='us', index=None)
            df_balance = sf.load_balance(variant='annual', market='us', index=None)
            df_cashflow = sf.load_cashflow(variant='annual', market='us', index=None)
            df_company = sf.load_companies(market="us", index=None)

            for ticker in tickers:
                self.save_data(df_prices[df_prices[TICKER] == ticker], os.path.join(self.output_folder,f"{ticker}_share_prices.csv"))
                self.save_data(df_income[df_income[TICKER] == ticker], os.path.join(self.output_folder,f"{ticker}_income_statement.csv"))
                self.save_data(df_balance[df_balance[TICKER] == ticker], os.path.join(self.output_folder,f"{ticker}_balance_sheet.csv"))
                self.save_data(df_cashflow[df_cashflow[TICKER] == ticker], os.path.join(self.output_folder,f"{ticker}_cash_flow.csv"))
                self.save_data(df_company[df_company[TICKER] == ticker], os.path.join(self.output_folder,f"{ticker}_company.csv"))

        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")


