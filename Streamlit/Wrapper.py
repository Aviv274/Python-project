# %%
import requests
import pandas as pd
import logging
from datetime import datetime

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



class PySimFin:
    def __init__(self, api_key: str, logger: logging.Logger):
        """
        Constructor to initialize API interaction.
        """
        self.base_url = "https://backend.simfin.com/api/v3/"
        self.headers = {"Authorization": f"{api_key}"}
        self.logger = logger
        self.logger.info("PySimFin instance created successfully.")

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve share prices for a given ticker within a specific time range.
        """
        try:
            url = f"{self.base_url}companies/prices/verbose"
            params = {"ticker": ticker, "start": start, "end": end}
            self.logger.info(f"Fetching share prices for {ticker} from {start} to {end}...")

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and data: 
                df = pd.DataFrame(data[0]['data'])
                df['ticker'] = ticker
            else:
                self.logger.warning(f"No data returned for {ticker} from {start} to {end}.")
                df = pd.DataFrame()
            
            self.logger.info("Share prices retrieved successfully.")
            return df
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error retrieving share prices: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

    def get_financial_statement(self, ticker: str, statements: str, start: str, end: str) -> list:
        """
        Retrieve financial statements for a given ticker within a specific time range.
        If no data is available, return an empty list.
        """
        try:
            url = f"{self.base_url}companies/statements/verbose"
            params = {"ticker": ticker, "start": start, "end": end, "statements": statements}
            self.logger.info(f"Fetching financial statements {statements} for {ticker} from {start} to {end}.")

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()

            if isinstance(data, list) and data and 'statements' in data[0] and data[0]['statements']:
                self.logger.info("Financial statements retrieved successfully.")
                return pd.DataFrame(data[0]['statements'][0]['data'])
            else:
                self.logger.warning(f"No data found for {ticker} from {start} to {end}. Returning empty data frame.")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error retrieving financial statements: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

    def get_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve general company information based on the ticker symbol.
        """
        try:
            url = f"{self.base_url}companies/general/verbose"
            params = {"ticker": ticker}
            self.logger.info(f"Fetching company info for {ticker} ...")

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, dict):  # If data is a dict, wrap it in a list
                df = pd.DataFrame([data])
            elif isinstance(data, list) and data:
                df = pd.DataFrame(data)
            else:
                self.logger.warning(f"No company data found for {ticker}.")
                df = pd.DataFrame()

            self.logger.info("Company info retrieved successfully.")
            return df
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error retrieving company info: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise



