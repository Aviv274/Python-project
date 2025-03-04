import json
import joblib
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from Wrapper import PySimFin


class LivePredictor:
    def __init__(self, api_key: str, model_path: str, scaler_path: str, feature_path: str, logger):
        """
        Initializes the predictor by loading the ML model, scaler, and feature list.
        """
        self.simfin = PySimFin(api_key, logger)
        self.logger = logger
        
        self.logger.info("Loading ML model, scaler, and selected features...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(feature_path, "r") as file:
            self.selected_features = json.load(file)
        
        self.logger.info("StockPricePredictor initialized successfully.")

    def fetch_data(self, ticker: str, prediction_date: str) -> pd.DataFrame:
        """
        Fetches and merges stock price data, financial statements (latest fiscal year), and company info.
        """
        try:
            self.logger.info(f"Fetching data for {ticker} on {prediction_date}...")

            # Extract year from the prediction date
            year = datetime.strptime(prediction_date, "%Y-%m-%d").year

            # Fetch share prices
            prices_df = self.simfin.get_share_prices(ticker, prediction_date, prediction_date)
            # Fetch financial statements
            pl_df = self.simfin.get_financial_statement(ticker, "pl", year)
            bs_df = self.simfin.get_financial_statement(ticker, "bs", year)
            cf_df = self.simfin.get_financial_statement(ticker, "cf", year)

            # Fetch company info
            company_info_df = self.simfin.get_company_info(ticker)

            # Merge datasets
            df = prices_df.merge(pl_df, on=["ticker"], how="left")
            df = df.merge(bs_df, on=["ticker"], how="left")
            df = df.merge(cf_df, on=["ticker"], how="left")

            df = df[[col for col in df.columns if not col.endswith("_y")]]
            df.columns = [col.replace("_x", "") for col in df.columns]

            # Add company info
            if not company_info_df.empty:
                for col in company_info_df.columns:
                    df[col] = company_info_df[col].iloc[0]

            self.logger.info(f"Data successfully merged for {ticker} on {prediction_date}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects only the desired features and scales them.
        """
        try:
            self.logger.info("Preprocessing data...")
            # Select required features

            df = df[self.selected_features]

            # Handle missing values
            df = df.fillna(0).infer_objects(copy=False).convert_dtypes()
            # Scale the features
            scaled_features = self.scaler.transform(df)

            self.logger.info("Data preprocessing completed successfully.")
            return scaled_features
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

    def predict_next_day(self, ticker: str, prediction_date: str) -> str:
        """
        Predicts if the stock price will increase the day after the given date.
        """
        try:
            # Convert prediction date to datetime object
            prediction_date_dt = datetime.strptime(prediction_date, "%Y-%m-%d")
            next_day = (prediction_date_dt + timedelta(days=1)).strftime("%Y-%m-%d")

            # Fetch and preprocess data
            df = self.fetch_data(ticker, prediction_date)
            df = self.rename_columns_to_model(df,"Streamlit/mapping.json")
            processed_data = self.preprocess_data(df)

            # Make prediction (only the last row is needed)
            prediction = self.model.predict(processed_data[-1].reshape(1, -1))

            result = "Increase" if prediction[0] == 1 else "Decrease"
            self.logger.info(f"Prediction for {ticker} on {next_day}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error predicting stock price movement for {ticker} on {next_day}: {e}")
            raise

    def load_column_mapping(self,json_path: str) -> dict:
        """
        Load column mapping from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file containing column mappings.
        
        Returns:
            dict: Dictionary mapping model column names to API column names.
        """
        with open(json_path, 'r') as file:
            column_mapping = json.load(file)
        return column_mapping

    def rename_columns_to_model(self,df: pd.DataFrame, json_path: str) -> pd.DataFrame:
        """
        Rename DataFrame columns to match the model's expected column names.
        Only columns present in the mapping will be renamed; others remain unchanged.
        
        Args:
            df (pd.DataFrame): DataFrame whose columns need renaming.
            json_path (str): Path to the JSON file containing column mappings.
        
        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        column_mapping = self.load_column_mapping(json_path)  # Load mapping (API name -> Model name)
        df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns}, inplace=True)
        return df

   