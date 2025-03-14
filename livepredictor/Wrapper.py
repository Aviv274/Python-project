# %%
import requests
import pandas as pd
import logging
import time
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

        Args:
            api_key (str): API key for authentication.
            logger (logging.Logger): Logger instance for logging.
        """
        self.base_url = "https://backend.simfin.com/api/v3/"
        self.headers = {"Authorization": f"{api_key}"}
        self.logger = logger
        self.logger.info("PySimFin instance created successfully.")

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Internal method to make API requests with retry handling.

        Args:
            endpoint (str): API endpoint.
            params (dict): Query parameters.

        Returns:
            dict: JSON response if successful.

        Raises:
            Exception: If a request fails.
        """
        url = f"{self.base_url}{endpoint}"
        print(url)
        attempts = 0
        max_retries = 3
        
        while attempts < max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 400:
                    error_msg = "Bad request: Check your URL or parameters."
                elif response.status_code == 404:
                    error_msg = "API Not Found: Verify the endpoint URL."
                elif response.status_code == 429:
                    self.logger.warning("Rate limit exceeded. Retrying in 5 seconds...")
                    time.sleep(5)
                    attempts += 1
                    continue  # Retry again
                else:
                    error_msg = f"HTTP Error {response.status_code}: {response.reason}"

                self.logger.error(error_msg)
                raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

        error_msg = "Max retry limit reached due to rate limits."
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve share prices for a given ticker within a specific time range.

        Args:
            ticker (str): Stock ticker symbol.
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing share price data.
        """
        self.logger.info(f"Fetching share prices for {ticker} from {start} to {end}...")
        params = {"ticker": ticker, "start": start, "end": end}
        
        try:
            data = self._make_request("companies/prices/verbose", params)
            if isinstance(data, list) and data: 
                df = pd.DataFrame(data[0]['data'])
                df['ticker'] = ticker
            else:
                raise Exception(f"No data returned for {ticker} from {start} to {end}.")
            
            self.logger.info("Share prices retrieved successfully.")
            return df
        except Exception as e:
            if data is None:
                return None
            raise Exception(f"Error retrieving share prices: {e}")

    def get_financial_statement(self, ticker: str, statements: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve financial statements for a given ticker within a specific time range.

        Args:
            ticker (str): Stock ticker symbol.
            statements (str): Type of financial statement (e.g., 'income', 'balance', 'cashflow').
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing financial statement data.
        """
        self.logger.info(f"Fetching {statements} statements for {ticker} from {start} to {end}.")
        params = {"ticker": ticker, "statements": statements, "start": start, "end": end}

        try:
            data = self._make_request("companies/statements/verbose", params)
            if isinstance(data, list) and data and 'statements' in data[0] and data[0]['statements']:
                self.logger.info("Financial statements retrieved successfully.")
                return pd.DataFrame(data[0]['statements'][0]['data'])
            else:
                raise Exception(f"No data found for {ticker} from {start} to {end}.")
        except Exception as e:
            raise Exception(f"Error retrieving financial statements: {e}")

    def get_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve general company information based on the ticker symbol.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame containing company information.
        """
        self.logger.info(f"Fetching company info for {ticker} ...")
        params = {"ticker": ticker}

        try:
            data = self._make_request("companies/general/verbose", params)
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list) and data:
                df = pd.DataFrame(data)
            else:
                raise Exception(f"No company data found for {ticker}.")

            self.logger.info("Company info retrieved successfully.")
            return df
        except Exception as e:
            raise Exception(f"Error retrieving company info: {e}")
