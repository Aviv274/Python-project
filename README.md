# README: Stock Prediction System

## Overview
This project consists of an ETL pipeline, a machine learning model, and a live predictor module for stock price forecasting. The system relies on configuration and mapping files to customize data processing and model execution.

## Configuration Files
The system includes multiple configuration files that allow users to adjust parameters without modifying the source code.

### 1. ETL Configuration (`etl/config.json`)
This file contains settings for the ETL (Extract, Transform, Load) pipeline.

#### **Key Parameters:**
- **data_sources**: Defines locations of stock market data (e.g., balance sheets, cash flow, income statements, and share prices).
- **output_path**: Specifies the directory where cleaned and processed data is stored.
- **logging_level**: Configures the logging level (INFO, DEBUG, etc.).
- **date_range**: Defines the start and end dates for data extraction.

#### **How to Use:**
Modify the values in `config.json` to point to the correct data sources or change logging verbosity. The ETL pipeline will read this file when executed (`etl/run_etl.py`).

### 2. Machine Learning Configuration (`ml/config_ml.json`)
This file contains settings for training and running machine learning models.

#### **Key Parameters:**
- **model_type**: Specifies the type of model (e.g., Linear Regression, ARIMA, etc.).
- **features**: Lists the input features for training the model.
- **train_test_split_ratio**: Defines the proportion of training vs. test data.
- **output_model_path**: Specifies where trained models are saved.

#### **How to Use:**
Adjust `config_ml.json` to modify the machine learning model’s behavior. The script `ml/run_ml.py` will read this file when training or making predictions.

## Mapping File

### 3. Live Predictor Mapping (`livepredictor/mapping.json`)
This file maps stock tickers to company names and other metadata used by the live prediction module.

#### **Key Parameters:**
- **tickers**: A dictionary mapping stock ticker symbols (e.g., "AAPL") to full company names.
- **exchange**: Specifies the stock exchange (e.g., NASDAQ, NYSE).
- **data_source**: Defines where to fetch real-time stock data.

#### **How to Use:**
Modify `mapping.json` to add or update stock tickers. The live predictor (`livepredictor/Livepredictor.py`) will read this file when running stock forecasts in real-time.

## Running the System

### 1. Running the ETL Pipeline
```bash
python etl/run_etl.py
```

### 2. Training the Machine Learning Model
```bash
python ml/run_ml.py
```

### 3. Running Live Stock Predictions
```bash
streamlit run Stock predictor.py
```

Ensure that the configuration and mapping files are correctly set up before executing the scripts. Modifications to these files allow customization without altering code.

## Contact
For any issues or questions, please refer to the project documentation or contact the development team.

