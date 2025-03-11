from etl_class import StockETL
import logging
import os


def configure_global_logging():
    """
    Sets up logging for the main process.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        filename="etl/etl.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(__name__)


def main():
    """
    Runs the ETL process for multiple stocks.
    """
    #Change to import from global file 
    logger = configure_global_logging()
    logger.info("Starting ETL process.")

    etl = StockETL(logger=logger)
    tickers = etl.config.get("tickers", [])
    price_methods = etl.config.get("share_prices_cleaning_methods", {})
    income_methods = etl.config.get("income_cleaning_methods", {})
    balance_methods = etl.config.get("balance_cleaning_methods", {})
    cashflow_methods = etl.config.get("cashflow_cleaning_methods", {})
    company_methods = etl.config.get("company_cleaning_methods", {})
    etl.download_data()

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")

        df_prices, df_income, df_balance, df_cashflow, df_company = etl.load_data(ticker)
        if df_prices is None:
            logger.warning(f"Skipping {ticker} due to missing essential data.")
            continue

        df_prices = etl.clean_data(df_prices, price_methods)
        df_income = etl.clean_data(df_income, income_methods)
        df_balance = etl.clean_data(df_balance, balance_methods)
        df_cashflow = etl.clean_data(df_cashflow, cashflow_methods)
        df_company = etl.clean_data(df_company, company_methods)

        df_merged = etl.merge_data(df_prices, df_income, df_balance, df_cashflow, df_company, ticker)
        if df_merged is None:
            logger.warning(f"Skipping {ticker} due to failed merge.")
            continue

        etl.save_data(df_merged, os.path.join(etl.clean_folder, f"{ticker}_merged_data.csv"))
        logger.info(f"Successfully processed {ticker}")

    logger.info("ETL process completed successfully!")

if __name__ == "__main__":
    main()