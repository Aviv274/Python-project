from StockPricePredictor import load_config, configure_global_logging
from reg_model import LogisticRegrModel
import numpy as np

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