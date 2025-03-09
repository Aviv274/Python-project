import joblib
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
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from StockPricePredictor import StockPricePredictor, load_config, configure_global_logging


class LogisticRegrModel(StockPricePredictor):
    def __init__(self, stock_name, config, logger):
        super().__init__(stock_name,config,logger)
        """Initialize the model, scalers, and preprocess data."""
        self.scaler = StandardScaler()
        self.elastic_net = ElasticNet(
            alpha=config["alpha"], 
            l1_ratio=config["l1_ratio"], 
            random_state=config["random_state"]
        )
        self.cv = config["cv"]
        self.scoring = config["scoring"]
        self.top_features = config["top_features"]
        self.random_state = config["random_state"]
        self.columns_to_drop = list(config["columns_to_drop"])
        self.X, self.y = self.preprocess_data()
        self.preprocess_data_for_training()
        self.best_params = None
    
    def preprocess_data(self):
        """Preprocess dataset: Feature selection and scaling."""
        super().preprocess_data()
        self.df['Target'] = (self.df['Close'] > self.df['Close'].shift(-1)).astype(int)
        self.df = self.df[:-1]
        self.df = self.df.drop(columns=self.columns_to_drop, errors='ignore')
        
        feature_candidates = self.df.select_dtypes(include=['number']).columns.drop(['Target'])
        X_all = self.df[feature_candidates]
        y_all = self.df['Target']

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
        return self.df[self.selected_features], self.df['Target']


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
            cv=self.cv,
            scoring=self.scoring,
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
        raise ValueError(f"x new - {X_new.columns} selected features - {self.selected_features}")
        X_new = X_new[self.selected_features]

        # Scale the data
        X_new_scaled = self.scaler.transform(X_new)

        return self.model.predict(X_new_scaled)


if __name__ == "__main__":
    
    config = load_config("ml/config_ml.json")
    logger = configure_global_logging(config)

    for ticker in config["tickers"]:
        predictor = LogisticRegrModel(ticker, config, logger)
        
        predictor.tune_hyperparameters(config)
        predictor.train()
        predictor.save_model()
        sample_data = np.array([predictor.X.iloc[-1]])
        logger.info(f"{[predictor.X.iloc[-1]]}")
        logger.info(f"Prediction for the next day ({ticker}): {predictor.predict(sample_data)}")
