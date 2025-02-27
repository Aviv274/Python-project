# Stock Market Data Analysis & Machine Learning

This project contains an ETL pipeline for stock market data and a machine learning model for financial predictions.

## Project Structure

```
/etl
    ├── etl_class.py       # ETL script
    ├── config.json        # Configuration file
    ├── stock_data/        # Raw stock data (CSV)
    ├── simfin_data/       # Downloaded financial data
    ├── clean/             # Cleaned & merged datasets
/ml
    ├── reg_model.py       # Machine Learning model script
    ├── config_ml.json     # ML configuration file
    ├── scalers/           # Pre-trained scalers
    ├── features/          # Selected features JSON
    ├── models/            # Trained machine learning models
```

## Installation

```sh
pip install -r requirements.txt
```

## Usage

### Running ETL Process
```sh
python etl/etl_class.py
```

### Running Machine Learning Model
```sh
python ml/reg_model.py
```

## Dependencies
The required Python libraries are listed in `requirements.txt`.

## Contributors
- Your Name

## License
This project is licensed under the MIT License.
