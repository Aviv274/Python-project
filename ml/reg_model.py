import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


class StockPricePredictor:
    def __init__(self, stock_name, config, logger):
        """Initialize the model, scalers, and preprocess data."""
        self.stock_name = stock_name
        self.file_path = f"{config['data_directory']}/{stock_name}_merged_data.csv"
        self.logger = logger

        # Directories for saving models and scalers
        self.models_directory = config["models_directory"]
        self.scalers_directory = config["scalers_directory"]
        self.features_directory = config["features_directory"]

        # Model parameters
        self.scaler = StandardScaler()
        self.elastic_net = ElasticNet(
            alpha=config["alpha"], 
            l1_ratio=config["l1_ratio"], 
            random_state=config["random_state"]
        )
        self.top_features = config["top_features"]
        self.test_size = config["test_size"]
        self.random_state = config["random_state"]

        self.columns_to_drop = list(config["columns_to_drop"])

        # Load dataset
        self.df = pd.read_csv(self.file_path)
        self.X, self.y = self.preprocess_data(self.df)

        # Perform train-test split, SMOTE, and scaling once
        self.preprocess_data_for_training()

        self.best_params = None
        self.model = None

    
    def preprocess_data(self, df):
        """Preprocess dataset: Feature selection and scaling."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df['Target'] = (df['Close'] > df['Close'].shift(-1)).astype(int)
        df = df[:-1]  # Drop last row with NaN target

        # Drop specified columns
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        
        feature_candidates = df.select_dtypes(include=['number']).columns.drop(['Target'])
        X_all = df[feature_candidates]
        y_all = df['Target']

        # Feature selection using ElasticNet
        X_scaled = MinMaxScaler().fit_transform(X_all)  # Use MinMaxScaler for ElasticNet
        self.elastic_net.fit(X_scaled, y_all)

        coef_abs = np.abs(self.elastic_net.coef_)
        top_indices = np.argsort(coef_abs)[-self.top_features:]
        self.selected_features = feature_candidates[top_indices].tolist()

        # Save selected features to a file
        feature_file = f"{self.features_directory}/{self.stock_name}_selected_features.json"
        with open(feature_file, "w") as f:
            json.dump(self.selected_features, f)

        self.logger.info(f"Selected features for {self.stock_name}: {self.selected_features}")
        return df[self.selected_features], df['Target']


    def preprocess_data_for_training(self):
        """Splits data, applies SMOTE, and scales features. Runs only once."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        # Apply SMOTE once
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(X_train, y_train)

        # Scale data once
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_resampled)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.y_test = y_test  # Store y_test for evaluation


    def tune_hyperparameters(self, config):
        """Tune hyperparameters using GridSearchCV with preprocessed data."""
        param_grid = config["hyperparameter_grid"]

        # Use preprocessed, scaled, and resampled training data
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        grid_search.fit(self.X_train_scaled, self.y_train_resampled)

        # Store the best model parameters
        self.best_params = grid_search.best_params_
        self.model = RandomForestClassifier(**self.best_params, random_state=self.random_state)

        self.logger.info(f"Best Parameters: {self.best_params}")
        self.logger.info(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")


    def train(self):
        """Train the model using best hyperparameters."""
        if self.model is None:
            self.model = LogisticRegression(solver='liblinear', random_state=42)  # Default model if tuning is not performed

        self.model.fit(self.X_train_scaled, self.y_train_resampled)

        predictions = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)

        self.logger.info(f"Model training complete for {self.stock_name}.")
        self.logger.info(f"Model Accuracy: {accuracy:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")

    def predict(self, X_new):
        """Predict whether the stock price will rise (1) or not (0)."""
        
        # Ensure X_new is a DataFrame
        if isinstance(X_new, np.ndarray):
            X_new = pd.DataFrame(X_new, columns=self.selected_features)

        # Select only the stored features
        X_new = X_new[self.selected_features]

        # Scale the data
        X_new_scaled = self.scaler.transform(X_new)

        return self.model.predict(X_new_scaled)

    
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


if __name__ == "__main__":
    
    config = load_config("ml/config_ml.json")
    logger = configure_global_logging(config)

    for ticker in config["tickers"]:
        predictor = StockPricePredictor(ticker, config, logger)
        
        predictor.tune_hyperparameters(config)
        predictor.train()
        predictor.save_model()
        sample_data = np.array([predictor.X.iloc[-1]])
        logger.info(f"{[predictor.X.iloc[-1]]}")
        logger.info(f"Prediction for the next day ({ticker}): {predictor.predict(sample_data)}")
