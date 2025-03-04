# Stock Market Data Analysis & Machine Learning

This project includes an **ETL pipeline** for stock market data processing, a **machine learning model** for financial predictions, and a **Streamlit-based interactive app** for visualizing stock trends and predictions.

## Project Structure

```
/etl
    ├── etl_class.py       # ETL pipeline script
    ├── config.json        # Configuration file
    ├── stock_data/        # Raw stock data (CSV files)
    ├── simfin_data/       # Downloaded financial datasets
    ├── clean/             # Processed & cleaned data
/ml
    ├── reg_model.py       # Machine learning model script
    ├── config_ml.json     # ML configuration file
    ├── scalers/           # Pre-trained scalers
    ├── features/          # Selected features (JSON files)
    ├── models/            # Trained ML models
/Streamlit
    ├── Wrapper.py         # Streamlit app wrapper
    ├── Livepredictor.py   # Live stock price prediction script
    ├── mapping.json       # Configuration mapping
```

## Installation

Ensure you have Python installed (recommended version: 3.8+). Install dependencies using:

```sh
pip install -r requirements.txt
```

## How It Works

### 1. Extract, Transform, Load (ETL)
- **Extracts** stock market data from CSV and APIs.
- **Transforms** data (cleaning, feature selection, merging financial metrics).
- **Loads** processed data into structured formats for analysis.

Run the ETL pipeline:
```sh
python etl/etl_class.py
```

### 2. Machine Learning Model
- Trains models to predict stock price trends.
- Uses logistic regression and feature selection techniques.

Run the ML model:
```sh
python ml/reg_model.py
```

### 3. Streamlit Dashboard
- Visualizes stock data and predictions interactively.

Launch the Streamlit app:
```sh
streamlit run Streamlit/app.py
```

## Dataset Sources
- **SimFin** for fundamental stock data
- **Yahoo Finance** for market data

## Authors & Contributions
- ETL: [Group 6]
- ML Model: [Group 6]
- Streamlit App: [Group 6]

## License
This project is licensed under the MIT License.

