import joblib
import logging
import pandas as pd
import os
import json



class StockPricePredictor:
    def __init__(self, stock_name: str, config: dict, logger: logging.Logger):
        """
        Initialize the StockPricePredictor with the given stock name, configuration, and logger.

        Args:
            stock_name (str): The name of the stock to predict.
            config (dict): Configuration parameters for the predictor.
            logger (logging.Logger): Logger object for logging information and errors.

        Returns:
            None
        """
        self.stock_name = stock_name
        self.logger = logger
        
        # Validate config
        self._validate_config(config)
        
        self.file_path = f"{config['data_directory']}/{stock_name}_merged_data.csv"
        self.models_directory = config["models_directory"]
        self.scalers_directory = config["scalers_directory"]
        self.columns_to_drop = list(config["columns_to_drop"])
        self.df = pd.read_csv(self.file_path)
        self.test_size = config["test_size"]
        self.model = None

    def _validate_config(self, config: dict) -> None:
        """
        Validate the configuration dictionary.

        Args:
            config (dict): Configuration parameters to validate.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        required_keys = [
            "data_directory", "models_directory", "scalers_directory",
            "columns_to_drop", "test_size"
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")
        
        if not os.path.isdir(config["data_directory"]):
            raise ValueError(f"Data directory does not exist: {config['data_directory']}")
        
        if not isinstance(config["columns_to_drop"], list):
            raise ValueError("columns_to_drop must be a list")
        
        if not 0 < config["test_size"] < 1:
            raise ValueError("test_size must be between 0 and 1")

        self.logger.info("Configuration validated successfully.")

    
    def preprocess_data(self) -> None:
        """
        Preprocess the data by converting dates and sorting the DataFrame.

        This method:
        1. Converts the 'Date' column to datetime format.
        2. Sorts the DataFrame by date.

        Args:
            None

        Returns:
            None
        """
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by=['Date'])
    
    def save_model(self) -> None:
        """
        Save the trained model and scaler with the stock name.

        This method:
        1. Creates necessary directories if they don't exist.
        2. Saves the trained model and scaler as pickle files.

        Args:
            None

        Returns:
            None
        """
            # Ensure directories exist
        os.makedirs(self.models_directory, exist_ok=True)
        os.makedirs(self.scalers_directory, exist_ok=True)
            
        model_path = f"{self.models_directory}/{self.stock_name}_model.pkl"
        scaler_path = f"{self.scalers_directory}/{self.stock_name}_scaler.pkl"
            
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
            
        self.logger.info(f"Model saved as {model_path}, Scaler saved as {scaler_path}.")

    
    def load_model(self) -> None:
        """
        Save the trained model and scaler with the stock name.

        This method:
        1. Creates necessary directories if they don't exist.
        2. Saves the trained model and scaler as pickle files.

        Args:
            None

        Returns:
            None
        """
        model_path = f"{self.models_directory}/{self.stock_name}_model.pkl"
        scaler_path = f"{self.scalers_directory}/{self.stock_name}_scaler.pkl"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        self.logger.info(f"Model and scaler for {self.stock_name} loaded successfully.")


def load_config(config_path):
    """Loads the configuration file."""
    if not os.path.exists(config_path):
        print("Config file not found! Using default settings.")
        return {}

    try:
        with open(config_path, 'r') as file:
            return json.load(file) or {}
    except json.JSONDecodeError:
        print("Failed to load config.json. Using default settings.")
        return {}

    
def configure_global_logging(config):
    """Sets up logging for the main process using config."""
    logging.basicConfig(
        filename=config["log_file"],
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(__name__)

