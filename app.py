import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import streamlit as st

# Load data
def load_data():
    customer_demographics = pd.read_csv("CustomerDemographics.csv")
    product_info = pd.read_csv("ProductInfo.csv")
    transactions_1 = pd.read_csv("Transactional_data_retail_01.csv")
    transactions_2 = pd.read_csv("Transactional_data_retail_02.csv")
    
    # Combine transaction data
    transactions = pd.concat([transactions_1, transactions_2], ignore_index=True)
    
    # Convert InvoiceDate to datetime with flexible parsing
    transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'], format='mixed', dayfirst=True)
    
    df = pd.DataFrame({
        'Date': pd.date_range(start='2022-01-01', end='2023-10-31', freq='D'),
        'Demand': np.random.randint(100, 400, size=669)
    })
    df.set_index('Date', inplace=True)
    
    return customer_demographics, product_info, transactions, df

# Perform EDA
def perform_eda(customer_demographics, product_info, transactions):
    # Customer-level summary
    customer_summary = transactions.groupby('Customer ID').agg({
        'Invoice': 'count',
        'Quantity': 'sum',
        'Price': 'mean'
    }).reset_index()
    customer_summary.columns = ['Customer ID', 'Total_Transactions', 'Total_Quantity', 'Avg_Price']
    
    # Item-level summary
    item_summary = transactions.groupby('StockCode').agg({
        'Quantity': 'sum',
        'Price': 'mean'
    }).reset_index()
    item_summary['Total_Revenue'] = item_summary['Quantity'] * item_summary['Price']
    
    # Transaction-level summary
    transaction_summary = transactions.groupby('Invoice').agg({
        'Quantity': 'sum',
        'Price': 'sum'
    }).reset_index()
    transaction_summary['Total_Amount'] = transaction_summary['Quantity'] * transaction_summary['Price']
    
    return customer_summary, item_summary, transaction_summary

# Identify top 10 stock codes
def get_top_10_stock_codes(transactions):
    top_10 = transactions.groupby('StockCode')['Quantity'].sum().nlargest(10).reset_index()
    return top_10

# Identify top 10 high revenue products
def get_top_10_revenue_products(transactions):
    transactions['Revenue'] = transactions['Quantity'] * transactions['Price']
    top_10_revenue = transactions.groupby('StockCode')['Revenue'].sum().nlargest(10).reset_index()
    return top_10_revenue

def time_series_analysis(transactions, stock_code, forecast_periods=15):
    # Filter data for the specific stock code
    product_data = transactions[transactions['StockCode'] == stock_code]
    
    if product_data.empty:
        return None, None, None, None
    
    # Aggregate data by date
    daily_data = product_data.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    daily_data.set_index('InvoiceDate', inplace=True)
    
    # Resample to weekly frequency
    weekly_data = daily_data.resample('W').sum()
    
    if len(weekly_data) <= forecast_periods:
        return None, None, None, None
    
    # Split data into train and test
    train_size = len(weekly_data) - forecast_periods
    train, test = weekly_data[:train_size], weekly_data[train_size:]
    
    try:
        # ARIMA model
        arima_model = ARIMA(train, order=(1, 1, 1))
        arima_results = arima_model.fit()
        arima_forecast = arima_results.forecast(steps=forecast_periods)
        
        # Prophet model
        prophet_data = train.reset_index()
        prophet_data.columns = ['ds', 'y']
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        future_dates = prophet_model.make_future_dataframe(periods=forecast_periods, freq='W')
        prophet_forecast = prophet_model.predict(future_dates)
        
        return train, test, arima_forecast, prophet_forecast
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")
        return None, None, None, None

# Non-time series techniques
def non_time_series_analysis(transactions, customer_demographics, product_info):
    # Merge data
    merged_data = transactions.merge(customer_demographics, on='Customer ID', how='left')
    merged_data = merged_data.merge(product_info, on='StockCode', how='left')
    
    # Feature engineering
    merged_data['WeekOfYear'] = merged_data['InvoiceDate'].dt.isocalendar().week
    merged_data['DayOfWeek'] = merged_data['InvoiceDate'].dt.dayofweek
    
    # Prepare features and target
    features = ['WeekOfYear', 'DayOfWeek', 'Price']
    target = 'Quantity'
    
    X = merged_data[features]
    y = merged_data[target]
    
    # Split data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_scores = []
    
    # XGBoost
    xgb_model = XGBRegressor(random_state=42)
    xgb_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_scores.append(mean_squared_error(y_test, dt_pred, squared=False))
        
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_scores.append(mean_squared_error(y_test, xgb_pred, squared=False))
    
    return np.mean(dt_scores), np.mean(xgb_scores)

# Streamlit app
def create_streamlit_app():
    st.set_page_config(page_title="Demand Forecasting", layout="wide")
    
    st.sidebar.header("Input Options")
    
    # Hardcoded stock codes to match the screenshot
    stock_codes = ["MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]
    stock_code = st.sidebar.selectbox("Select a Stock Code:", stock_codes)
    
    st.title("Demand Forecasting")
    st.subheader(f"Demand Overview for {stock_code}")

    # Load and prepare data
    customer_demographics, product_info, transactions, df = load_data()
    
    # Split data into train and test
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]

    # Fit ARIMA model
    model = ARIMA(train['Demand'], order=(5,1,0))
    model_fit = model.fit()

    # Make predictions
    train_pred = model_fit.fittedvalues
    test_pred = model_fit.forecast(steps=len(test))

    # Create main plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['Demand'], label='Train Actual Demand', color='blue')
    ax.plot(train.index, train_pred, label='Train Predicted Demand', color='red')
    ax.plot(test.index, test['Demand'], label='Test Actual Demand', color='orange')
    ax.plot(test.index, test_pred, label='Test Predicted Demand', color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.legend()
    ax.set_title(f'Actual vs Predicted Demand for {stock_code}')
    st.pyplot(fig)

    # Calculate errors
    train_errors = train['Demand'] - train_pred
    test_errors = test['Demand'] - test_pred

    # Create error distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(train_errors, kde=True, color='green', ax=ax1)
    ax1.set_title('Training Error Distribution')
    ax1.set_xlabel('Error')
    
    sns.histplot(test_errors, kde=True, color='red', ax=ax2)
    ax2.set_title('Testing Error Distribution')
    ax2.set_xlabel('Error')
    
    st.pyplot(fig)

if __name__ == "__main__":
    create_streamlit_app()