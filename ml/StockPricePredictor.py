import joblib
import logging
import pandas as pd
import os
import json



class StockPricePredictor:
    def __init__(self, stock_name, config, logger):
        """Initialize the Logistic Regression model, ElasticNet, and Scalers."""
        self.stock_name = stock_name
        self.file_path = f"{config['data_directory']}/{stock_name}_merged_data.csv"
        self.logger = logger
        # Directories for saving models and scalers
        self.models_directory = config["models_directory"]
        self.scalers_directory = config["scalers_directory"]
        self.features_directory = config["features_directory"]
        self.columns_to_drop = list(config["columns_to_drop"])
        self.df = pd.read_csv(self.file_path)
        self.test_size = config["test_size"]
        self.model = None

    
    def preprocess_data(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by=['Date'])

    def preprocess_data_for_training(self):
        pass


    def tune_hyperparameters(self, config):
        pass

    
    def train(self):
        pass
    
    def predict(self, X_new):
        """Predict whether the stock price will rise (1) or not (0)."""
        pass

    
    def save_model(self):
            """Save the trained model, scaler, and selected features with stock name."""
            # Ensure directories exist
            os.makedirs(self.features_directory, exist_ok=True)
            os.makedirs(self.models_directory, exist_ok=True)
            os.makedirs(self.scalers_directory, exist_ok=True)
            
            feature_file = f"{self.features_directory}/{self.stock_name}_selected_features.json"
            model_path = f"{self.models_directory}/{self.stock_name}_logistic_regression_model.pkl"
            scaler_path = f"{self.scalers_directory}/{self.stock_name}_logistic_scaler.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            with open(feature_file, "w") as f:
                json.dump(self.selected_features, f)
            
            self.logger.info(f"Model saved as {model_path}, Scaler saved as {scaler_path}, Features saved as {feature_file}.")

    
    def load_model(self):
        """Load the trained model, scaler, and selected features."""
        model_path = f"models/{self.stock_name}_logistic_regression_model.pkl"
        scaler_path = f"scalers/{self.stock_name}_scaler.pkl"
        feature_file = f"models/{self.stock_name}_selected_features.json"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Load selected features
        with open(feature_file, "r") as f:
            self.selected_features = json.load(f)

        self.logger.info(f"Model, scaler, and selected features for {self.stock_name} loaded successfully.")


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

