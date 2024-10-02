
import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objs as go
from streamlit.components.v1 import iframe
import os
import plotly.express as px
from datetime import timedelta
import seaborn as sns
from io import StringIO
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
from datetime import datetime, timedelta
import re
import math

# API URLs
LIST_MODELS_URL = 'http://localhost:5000/list_models'
API_URL_PROPHET = 'http://localhost:5000/predict_prophet'
LIST_ENDPOINTS_URL = 'http://localhost:5000/list_user_endpoints'
DEPLOY_MODEL_URL = 'http://localhost:5000/deploy_model'
DELETE_ENDPOINT_URL = 'http://localhost:5000/delete_endpoint'
PREDICT_URL = 'http://localhost:5000/predict'
COLUMNS_API_URL = 'http://localhost:5000/get_columns'


# URL of the Flask server
FLASK_SERVER_URL = "http://localhost:5000"

def parse_date(date_data):
    possible_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", 
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
    ]
    
    def try_parse_single_date(date_str):
        for fmt in possible_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"No valid date format found for: {date_str}")
    
    # If it's a single string, parse it directly
    if isinstance(date_data, str):
        return try_parse_single_date(date_data)
    
    # If it's a Series, apply the parsing function to each element
    elif isinstance(date_data, pd.Series):
        for fmt in possible_formats:
            try:
                parsed_dates = date_data.apply(lambda x: datetime.strptime(x, fmt) if pd.notnull(x) else pd.NaT)
                
                # Check if all parsed dates are valid (no NaT)
                if not parsed_dates.isnull().any():
                    return parsed_dates  # Return the successfully parsed dates
            except ValueError:
                continue
        
        # If none of the formats work, raise an error
        raise ValueError("No valid date format found for the series.")
    
    else:
        raise TypeError("Input should be a string or a Pandas Series.")

def parse_date_prophet(date_data):
    possible_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", 
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
    ]
    
    def try_parse_single_date(date_str):
        for fmt in possible_formats:
            try:
                return datetime.strptime(date_str, fmt), fmt
            except ValueError:
                continue
        raise ValueError(f"No valid date format found for: {date_str}")
    
    # If it's a single string, parse it directly
    if isinstance(date_data, str):
        parsed_date, format_used = try_parse_single_date(date_data)
        return parsed_date, format_used
    
    # If it's a Series, apply the parsing function to each element
    elif isinstance(date_data, pd.Series):
        for fmt in possible_formats:
            try:
                parsed_dates = date_data.apply(lambda x: datetime.strptime(x, fmt) if pd.notnull(x) else pd.NaT)
                
                # Check if all parsed dates are valid (no NaT)
                if not parsed_dates.isnull().any():
                    return parsed_dates, fmt  # Return the successfully parsed dates and format
            except ValueError:
                continue
        
        # If none of the formats work, raise an error
        raise ValueError("No valid date format found for the series.")
    
    else:
        raise TypeError("Input should be a string or a Pandas Series.")
    
def parse_dates(date_series):
    """Attempt to parse dates using multiple formats and standardize to 'YYYY-MM-DD'."""
    formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", 
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
    ]
    def try_parsing(date_str, formats):
        """Try parsing a single date string with multiple formats."""
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt, errors='coerce')
            except ValueError:
                continue
        return pd.NaT  # Return NaT if no format matches

    # Initialize an empty DataFrame for results
    results = pd.Series(pd.NaT, index=date_series.index, dtype='datetime64[ns]')
    
    for fmt in formats:
        # Apply parsing for the current format
        temp_results = date_series.apply(lambda x: try_parsing(x, [fmt]))
        
        # Update results where there were NaT values previously
        results.update(temp_results[results.isna()])
        
        # Stop if no more NaT values
        if results.isna().sum() == 0:
            break
    
    # Handle any remaining NaT values if needed
    if results.isna().any():
        print("Warning: Some dates could not be parsed. They have been set to NaT.")
    
    return results, fmt


def decompose_time_series2(df, column, date_column):
    """Decomposes the time series for weekly, monthly, and yearly frequencies and plots trend and seasonality side by side."""

    # Ensure the data is sorted by date
    df = df.sort_values(by=date_column).copy()
    
    df[date_column] = df[date_column].dt.date
    if df[date_column].duplicated().any():
        # Group by the date and sum the target values
        df = df.groupby(date_column, as_index=False)[column].sum()
    df.set_index(date_column, inplace=True)
    df = df.asfreq('D')  # Ensure daily frequency
    df[column] = df[column].interpolate(method='linear')
    df[column] = (df[column] - df[column].mean()) / df[column].std()

    frequencies = {
        'Weekly': 7,
        'Monthly': 30,  # Approximate month length
        'Yearly': 365   # Approximate year length
    }

    for freq_name, period in frequencies.items():
        st.write(f"### {freq_name} Decomposition")

        # Check if there's enough data
        if len(df) < 2 * period:
            st.warning(f"Not enough data points for {freq_name} decomposition with period {period}. At least {2*period} data points are required.")
            continue

        # Perform the decomposition
        decomposition = seasonal_decompose(df[column], model='additive', period=int(period))

        # Prepare seasonal component for plotting
        seasonal = decomposition.seasonal.iloc[:period].copy()
        if freq_name == 'Weekly':
            x_labels_seasonal = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonal.index = x_labels_seasonal[:len(seasonal)]
        elif freq_name == 'Monthly':
            x_labels_seasonal = list(range(1, len(seasonal) + 1))
            seasonal.index = x_labels_seasonal
        elif freq_name == 'Yearly':
            x_labels_seasonal = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonal = seasonal.iloc[:12]  # Ensure only 12 months
            seasonal.index = x_labels_seasonal[:len(seasonal)]
        else:
            seasonal.index = seasonal.index.strftime('%Y-%m-%d')

        # Prepare trend component for plotting
        trend = decomposition.trend.dropna()
        trend_df = trend.to_frame(name='Trend')
        trend_df['Date'] = trend_df.index

        if freq_name == 'Weekly':
            trend_df['DayOfWeek'] = trend_df['Date'].dt.day_name()
            trend_grouped = trend_df.groupby('DayOfWeek')['Trend'].mean()
            x_labels_trend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            trend_grouped = trend_grouped.reindex(x_labels_trend)
        elif freq_name == 'Monthly':
            trend_df['DayOfMonth'] = trend_df['Date'].dt.day
            trend_grouped = trend_df.groupby('DayOfMonth')['Trend'].mean()
            x_labels_trend = trend_grouped.index.astype(int).astype(str)
        elif freq_name == 'Yearly':
            trend_df['Month'] = trend_df['Date'].dt.month
            trend_grouped = trend_df.groupby('Month')['Trend'].mean()
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x_labels_trend = [months_order[month - 1] for month in trend_grouped.index]
        else:
            trend_grouped = trend_df['Trend']
            x_labels_trend = trend_df['Date'].dt.strftime('%Y-%m-%d')

        # Create and display trend plot
        trend_plot = px.line(x=x_labels_trend, y=trend_grouped.values, title=f"{freq_name} Trend")
        trend_plot.update_layout(xaxis_title=freq_name, yaxis_title='Trend')
        st.plotly_chart(trend_plot, use_container_width=True)

        # Create and display seasonal plot
        seasonal_plot = px.line(x=seasonal.index, y=seasonal.values, title=f"{freq_name} Seasonality")
        seasonal_plot.update_layout(xaxis_title=freq_name, yaxis_title='Seasonal')
        st.plotly_chart(seasonal_plot, use_container_width=True)

def calculate_accuracy(y_true, y_pred, flag=None):
    """Calculates and returns common accuracy metrics."""
    
    # Clip y_true values to avoid division by zero or near-zero values for MAPE calculation
    y_true_clipped = np.clip(y_true, 1e-9, None)
    
    if flag:
        # Calculate MAPE with clipped y_true values
        mape = np.mean(np.abs((y_true_clipped - y_pred) / y_true_clipped))
        accuracy = 100 - (mape * 100)  # Accuracy formula: 100% - MAPE
        
        # Ensure accuracy does not exceed 100%
        accuracy = np.clip(accuracy, 0, 100)
        
        st.session_state.accuracy = accuracy
        st.subheader(f"Forecast Accuracy: {accuracy:.2f}%")
    else:
        # Calculate R^2 score (which ranges from -∞ to 1)
        r2 = r2_score(y_true, y_pred)
        st.session_state.accuracy = r2 * 100
        
        # Cap accuracy at 100% and ensure no negative percentage is shown
        st.session_state.accuracy = np.clip(st.session_state.accuracy, 0, 100)
        
        st.subheader(f"Forecast Accuracy: {st.session_state.accuracy:.2f}%")
        
def plot_forecast(train_df, test_df, forecast_df, date_column, target_column,forecast_horizon=None,  predict_flag=False, is_go=False):
    """Plots the forecast vs. actual using Altair and Matplotlib in separate plots."""
    """Plots the forecast vs. actual using Altair and Matplotlib in separate plots."""
    if predict_flag:
        last_date = forecast_df[date_column].max()
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='D')[1:]
        ####
    
        predictions = forecast_df['Predicted']
        future_predictions = predictions[-forecast_horizon:]

        # Create DataFrame for future predictions and store in session state
        future_df = pd.DataFrame({date_column: future_dates, 'Predicted': future_predictions})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df[date_column], y=train_df[target_column], mode='lines', name='Train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_df[date_column], y=test_df[target_column], mode='lines', name='Test', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=forecast_df[date_column], y=forecast_df['Predicted'], mode='lines', name='Predicted', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=future_df[date_column], y=future_df['Predicted'], mode='lines', name='Future Forecast', line=dict(color='red', dash='dot')))
        fig.update_layout(
            title='Actual vs Predicted with Train/Test Split and Future Forecast',
            xaxis_title='Date',
            yaxis_title='Target',
            legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)'),
            template='plotly_white'
        )
        
        st.session_state.forecast_fig = fig  # Store figure in session state
        st.plotly_chart(fig, use_container_width=True)

    else:
        if is_go:
            forecast_df = pd.DataFrame(forecast_df[[date_column, 'value']])
            train_df = train_df.rename(columns={date_column: 'date', target_column: 'value'})
            test_df = test_df.rename(columns={date_column: 'date', target_column: 'value'})
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_df['date'], y=train_df['value'], mode='lines', name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_df['date'], y=test_df['value'], mode='lines', name='Test', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=forecast_df[date_column], y=forecast_df['value'], mode='lines', name='Future Forecast', line=dict(color='red', dash='dot')))
            fig.update_layout(
                title='Actual vs Predicted with Train/Test Split and Future Forecast',
                xaxis_title=date_column,
                yaxis_title='Target',
                legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)'),
                template='plotly_white'
            )
            
            st.session_state.go_fig = fig  # Store figure in session state
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            train_df[date_column] = pd.to_datetime(train_df[date_column])
            test_df[date_column] = pd.to_datetime(test_df[date_column])
            forecast_df[date_column] = pd.to_datetime(forecast_df[date_column])
            forecast_test_df = forecast_df.iloc[:len(test_df)]
            future_forecast_df = forecast_df.iloc[len(test_df):]
            
            fig = go.Figure()

            # Plot the train data
            fig.add_trace(go.Scatter(
                x=train_df[date_column], y=train_df[target_column],
                mode='lines', name='Train',
                line=dict(color='blue')
            ))

            # Plot the test data
            fig.add_trace(go.Scatter(
                x=test_df[date_column], y=test_df[target_column],
                mode='lines', name='Test',
                line=dict(color='orange')
            ))

            # Plot the predicted values for the test period
            if not forecast_test_df.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_test_df[date_column], y=forecast_test_df['value'],
                    mode='lines', name='Predicted',
                    line=dict(color='green', dash='dash')
                ))

            # Plot the future forecast beyond the test period
            if not future_forecast_df.empty:
                fig.add_trace(go.Scatter(
                    x=future_forecast_df[date_column], y=future_forecast_df['value'],
                    mode='lines', name='Future Forecast',
                    line=dict(color='red', dash='dot')
                ))

            fig.update_layout(
                title='Actual vs Predicted with Train/Test Split and Future Forecast',
                xaxis_title='Date',
                yaxis_title=target_column,
                legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)'),
                template='plotly_white'
            )

            # Show the figure in the Streamlit app
            st.plotly_chart(fig, use_container_width=True)

def plot_actual_vs_predicted_combined(data, date_column, target_column, identifier_column=None, is_fallback=False):
    """Plot actual vs predicted values over time in one plot."""
    
    # For Prophet model, aggregate data by date using mean
    if is_fallback:
        if identifier_column and identifier_column in data.columns:
            # Aggregate data by date (mean)
            aggregated_data = data.groupby(date_column).agg({
                target_column: 'mean',
                'Predicted': 'mean'
            }).reset_index()
        else:
            aggregated_data = data.copy()
    else:
        # For non-Prophet models, use the raw data without aggregation
        aggregated_data = data.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aggregated_data[date_column],
        y=aggregated_data[target_column],
        mode='lines',
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        x=aggregated_data[date_column],
        y=aggregated_data['Predicted'],
        mode='lines',
        name='Predicted'
    ))
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title=target_column
    )
    st.plotly_chart(fig, use_container_width=True)  # Return the figure

def run_arima_plus_model():
    """Handles the forecasting with ARIMA_PLUS in BigQuery."""
    st.title("Time Series Forecasting with ARIMA_PLUS (BigQuery)")
    
    specific_dir = os.path.join(os.getcwd(), 'uploads')
    
    if not os.path.exists(specific_dir):
        os.makedirs(specific_dir)
    arima_plus_uploaded_file = None
    # Upload CSV file
    arima_plus_uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if arima_plus_uploaded_file is not None:
        clean_filename = arima_plus_uploaded_file.name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Create a valid file path with the cleaned file name
        clean_file_path = os.path.join(specific_dir, clean_filename)
        
        # Save uploaded file with the new cleaned name
        with open(clean_file_path, "wb") as f:
            f.write(arima_plus_uploaded_file.getbuffer())
        
        df = pd.read_csv(clean_file_path)

        # Clean column names (remove spaces and parentheses) and drop unnamed columns
        df.columns = df.columns.str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        st.subheader("Data preview:")
        st.write(df.head())

        # Select date and target columns
        date_column = st.selectbox("Select the date column", [None] + list(df.columns), key="arima_plus_date_column")
        target_column = st.selectbox("Select the target column", [None] + list(df.columns), key="arima_plus_target_column")
        forecast_period = st.number_input("Enter the number of periods to forecast", min_value=1, value=30, max_value=300, key="arima_plus_forecast_period")
        filter_column = st.selectbox("Select the filter column (optional)", [None] + list(df.columns), key="arima_plus_filter_column")

        if filter_column:
            # Filter by selected value from the filter column
            unique_values = df[filter_column].unique()
            filter_value = st.selectbox("Select a value to filter by", unique_values, key="filter_value")
            df = df[df[filter_column] == filter_value]

        if date_column and target_column and forecast_period:
            df[date_column] = df[date_column].apply(parse_date)
            
            df.sort_values(by=[date_column], inplace=True)
            
            # Handle hourly data by extracting only the date and grouping
            if df[date_column].duplicated().any():
                df[date_column] = df[date_column].dt.date
                df = df.groupby(date_column, as_index=False)[target_column].sum()

            # Check for gaps in the date range
            date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())
            missing_dates = date_range[~date_range.isin(df[date_column])]

            # Calculate the percentage of missing dates
            missing_percentage = len(missing_dates) / len(date_range) * 100

            # Handle missing dates
            if missing_percentage > 10:
                st.error(
                    "The dataset contains over 10% missing dates, which makes it unsuitable for ARIMA Plus. "
                    "You can try using the AutoML or Prophet model from the model library for non-daily datasets."
                )
                # st.write("Missing Dates:", missing_dates)
            else:
                max_total_length = 317
                initial_split_ratio = 0.9

                train_length = int(len(df) * initial_split_ratio)
                test_length = len(df) - train_length

                # Adjust the split if the total forecast period exceeds the max allowed length
                if (test_length + forecast_period) > max_total_length:
                    adjusted_split_ratio = (max_total_length - forecast_period) / len(df)
                    train_length = int(len(df) * adjusted_split_ratio)
                    test_length = len(df) - train_length

                # Split the data into training and test sets
                train_df, test_df = df[:train_length], df[train_length:]
                
                train_file_path = os.path.join(specific_dir, f"train_{clean_filename}")
                # test_file_path = os.path.join(specific_dir, f"test_{os.path.basename(clean_filename.name)}")
                
                train_df.to_csv(train_file_path, index=False)
                # test_df.to_csv(test_file_path, index=False)
                forecast_period_extended = forecast_period + len(test_df)

                if st.button("Start ARIMA Plus Training"):
                    # Post training data to Flask API
                    response = requests.post(
                        'http://127.0.0.1:5000/upload_data',
                        json={
                            'train_file_path': train_file_path,
                            'date_column': date_column
                        }
                    )
                    if response.ok:
                        params = {
                            'date_column': date_column,
                            'target_column': target_column,
                            'forecast_period': forecast_period_extended,
                            'train_file_path': train_file_path
                        }
                        st.success("Data uploaded successfully.")
                        
                        response = requests.get('http://127.0.0.1:5000/run_arima_plus', params=params)
                        if response.ok:
                            forecast_df = pd.DataFrame(response.json())

                            # Separate historical and forecast data
                            forecast_only_df = forecast_df[forecast_df['time_series_type'] == 'forecast']
                            history_only_df = forecast_df[forecast_df['time_series_type'] == 'history']

                            # Rename columns to align with data
                            forecast_only_df = forecast_only_df.rename(columns={'time_series_timestamp': date_column, 'time_series_data': 'value'})
                            history_only_df = history_only_df.rename(columns={'time_series_timestamp': date_column, 'time_series_data': 'value'})
                            last_train_date = train_df[date_column].max()
                            forecast_only_df[date_column] = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=len(forecast_only_df), freq='D')

                            st.subheader("Forecast Results:")
                            st.dataframe(forecast_only_df[[date_column, 'value']], use_container_width=True)

                            st.divider()
                            plot_forecast(train_df, test_df, forecast_only_df, date_column, target_column)
                            
                            st.divider()
                            trimmed_forecast_df = forecast_only_df.iloc[:len(test_df)].copy()
                            y_true = test_df[target_column].values
                            y_pred = trimmed_forecast_df['value'].values
                            calculate_accuracy(y_true,y_pred, flag=True)
                            st.divider()
                            decompose_time_series2(df, target_column, date_column)
                            st.divider()

                            # Download forecast as CSV
                            csv_forecast = forecast_only_df[[date_column, 'value']].to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv_forecast,
                                file_name='predictions.csv',
                                mime='text/csv'
                            )
                    elif response.status_code == 401:
                        error_message = response.json().get('error')
                        st.error(error_message)
                        
                        # Redirect to the login page using JavaScript
                        nav_script = """
                            <meta http-equiv="refresh" content="0; url='http://localhost:5000'">
                        """
                        st.write(nav_script, unsafe_allow_html=True)


def run_arima_model():
    """Executes ARIMA model and handles preprocessing, forecasting, and plotting."""
    st.title("Time Series Forecasting with ARIMA (BigQuery)")

    specific_dir = os.path.join(os.getcwd(), 'uploads')

    if not os.path.exists(specific_dir):
        os.makedirs(specific_dir)
    arima_uploaded_file = None
    # Upload CSV file
    arima_uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if arima_uploaded_file is not None:
        clean_filename = arima_uploaded_file.name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Create a valid file path with the cleaned file name
        clean_file_path = os.path.join(specific_dir, clean_filename)
        
        # Save uploaded file with the new cleaned name
        with open(clean_file_path, "wb") as f:
            f.write(arima_uploaded_file.getbuffer())
        
        df = pd.read_csv(clean_file_path)

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True)
        
        st.write("Data preview:")
        st.dataframe(df.head(), use_container_width=True)

        # Select date and target columns
        date_column = st.selectbox("Select the date column", [None] + list(df.columns), key="arima_date_column")
        target_column = st.selectbox("Select the target column", [None] + list(df.columns), key="arima_target_column")
        forecast_period = st.number_input("Enter the number of periods to forecast", min_value=1, max_value=317, value=30, key="arima_forecast_period")
        filter_column = st.selectbox("Select the filter column (optional)", [None] + list(df.columns), key="filter_column")

        if filter_column:
            # Get unique values from the selected filter column
            unique_values = df[filter_column].unique()
            filter_value = st.selectbox("Select a value to filter by", unique_values, key="filter_value")
            df = df[df[filter_column] == filter_value]

        if date_column and target_column:
            # Parse date column
            df[date_column] = df[date_column].apply(parse_date)

            # Check for duplicates and handle them
            if df[date_column].duplicated().any():
                df[date_column] = df[date_column].dt.date
                df = df.groupby(date_column, as_index=False)[target_column].sum()

            # Sort and reset index
            df.sort_values(by=[date_column], inplace=True)

            # Check for irregular intervals
            if df[date_column].isnull().any():
                st.error("The date column contains invalid dates.")
                return

            # Check for gaps in the date range
            date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())
            missing_dates = date_range[~date_range.isin(df[date_column])]

            # Calculate the percentage of missing dates
            missing_percentage = len(missing_dates) / len(date_range) * 100

            # Handle missing dates
            if missing_percentage > 10:
                st.error(
                    "The dataset contains over 10% missing dates, which makes it unsuitable for ARIMA. "
                    "You can try using the AutoML or Prophet model from the model library for non-daily datasets."
                )
                # st.write("Missing Dates:", missing_dates)
                return
            else:


                # Create training and test sets
                max_total_length = 317
                initial_split_ratio = 0.9

                train_length = int(len(df) * initial_split_ratio)
                test_length = len(df) - train_length

                if (test_length + forecast_period) > max_total_length:
                    adjusted_split_ratio = (max_total_length - forecast_period) / len(df)
                    train_length = int(len(df) * adjusted_split_ratio)
                    test_length = len(df) - train_length

                train_df, test_df = df[:train_length], df[train_length:]

                # Save train and test data
                train_file_path = os.path.join(specific_dir, f"train_{clean_filename}")
                # test_file_path = os.path.join(specific_dir, f"test_{os.path.basename(arima_uploaded_file.name)}")
                train_df.to_csv(train_file_path, index=False)
                # test_df.to_csv(test_file_path, index=False)

                # Extend the forecast period beyond the test period
                forecast_period_extended = forecast_period + len(test_df)

                if st.button("Start ARIMA Training"):
                    response = requests.post(
                        'http://127.0.0.1:5000/upload_data',
                        json={
                            'train_file_path': train_file_path,
                            'date_column': date_column
                        }
                    )
                    df.reset_index(drop=True, inplace=True)
                    if response.ok:
                        params = {
                            'date_column': date_column,
                            'target_column': target_column,
                            'forecast_period': forecast_period_extended,
                            'train_file_path': train_file_path
                        }
                        st.success("Data uploaded successfully.")
                        response = requests.get('http://127.0.0.1:5000/run_arima', params=params)
                        if response.ok:
                            forecast_df = pd.DataFrame(response.json())
                            forecast_df = forecast_df.rename(columns={'forecast_timestamp': date_column, 'forecast_value': 'value'})

                            # Update forecast start date to follow the last training date
                            last_train_date = train_df[date_column].max()
                            forecast_df[date_column] = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=len(forecast_df), freq='D')

                            st.subheader("Forecast Results:")
                            st.dataframe(forecast_df[[date_column, 'value']], use_container_width=True)

                            plot_forecast(train_df, test_df, forecast_df, date_column, target_column)

                            # Calculate accuracy
                            st.divider()
                            trimmed_forecast_df = forecast_df.iloc[:len(test_df)].copy()
                            y_true = test_df[target_column].values
                            y_pred = trimmed_forecast_df['value'].values
                            calculate_accuracy(y_true, y_pred, flag=True)

                            # Plot results
                            st.divider()
                            decompose_time_series2(df, target_column, date_column)
                            st.divider()
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name='predictions.csv',
                                mime='text/csv'
                            )

                        elif response.status_code == 401:
                            error_message = response.json().get('error')
                            st.error(error_message)
                            nav_script = """
                                <meta http-equiv="refresh" content="0; url='http://localhost:5000'">
                            """
                            st.write(nav_script, unsafe_allow_html=True)

                    elif response.status_code == 401:
                        error_message = response.json().get('error')
                        st.error(error_message)  # Show error message in Streamlit
                        nav_script = """
                            <meta http-equiv="refresh" content="0; url='http://localhost:5000'">
                        """
                        st.write(nav_script, unsafe_allow_html=True)


def run_times_fm():
    global_times_fm_model_response = None
    st.title("Time Series Forecasting with TimesFM")
    times_fm_uploaded_file = None
    times_fm_uploaded_file = st.file_uploader('Upload CSV file for TimesFM', type='csv')

    if times_fm_uploaded_file:
        # Load and process the CSV
        times_fm_df_actual = pd.read_csv(times_fm_uploaded_file)
        times_fm_df_actual.columns = times_fm_df_actual.columns.str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True)
        times_fm_columns = times_fm_df_actual.columns.tolist()

        # User inputs for columns
        times_fm_date_column = st.selectbox('Select Date Column', [None] + list(times_fm_columns))
        times_fm_actual_column = st.selectbox('Select Target Value Column', [None] + list(times_fm_columns))
        time_series_id_col = st.selectbox("Select the time series ID column (optional)", [None] + times_fm_columns, key="times_fm_series_id_column")
        
        # Optional regressor column
        times_fm_regressor_column = st.selectbox("Select Regressor Column (optional)", [None] + list(times_fm_columns))

        if times_fm_date_column and times_fm_actual_column:
            if time_series_id_col:
                unique_values = times_fm_df_actual[time_series_id_col].unique()
                selected_value = st.selectbox(f"Select a value for {time_series_id_col}", unique_values, key="times_fm_selected_value")
                times_fm_df_actual = times_fm_df_actual[times_fm_df_actual[time_series_id_col] == selected_value]

            # Date parsing
            times_fm_df_actual[times_fm_date_column] = times_fm_df_actual[times_fm_date_column].apply(parse_date)
            if times_fm_df_actual[times_fm_date_column].duplicated().any():
                times_fm_df_actual[times_fm_date_column] = times_fm_df_actual[times_fm_date_column].dt.date

                # Define the columns to sum
                sum_columns = [times_fm_actual_column]
                if times_fm_regressor_column:
                    sum_columns.append(times_fm_regressor_column)

                # Group by the date and sum the target and regressor values
                times_fm_df_actual = times_fm_df_actual.groupby(times_fm_date_column, as_index=False)[sum_columns].sum()

            times_fm_df_actual.sort_values(by=[times_fm_date_column], inplace=True)
            
            # Checking for missing dates
            min_date = times_fm_df_actual[times_fm_date_column].min()
            max_date = times_fm_df_actual[times_fm_date_column].max()
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            actual_dates = pd.to_datetime(times_fm_df_actual[times_fm_date_column].unique())
            missing_dates = set(expected_dates) - set(actual_dates)
            percentage_missing = len(missing_dates) / len(expected_dates) * 100
            st.write(f"Percentage of missing dates: {percentage_missing:.2f}%")

            if percentage_missing > 10:
                st.warning("Your data is not in daily form, please use AutoML as a recommended model.")
                return
            else:
                # Proceed with TimesFM training
                times_fm_df_actual[times_fm_actual_column] = pd.to_numeric(times_fm_df_actual[times_fm_actual_column], errors='coerce')
                times_fm_df_actual.dropna(subset=[times_fm_actual_column], inplace=True)

                # Set date column as index
                times_fm_df_actual.set_index(times_fm_date_column, inplace=True)

                # Ensure the data has a daily frequency
                times_fm_df_actual = times_fm_df_actual.asfreq('D').fillna(method='ffill')

                # Reset index
                times_fm_df_actual.reset_index(inplace=True)

                # Allow user to specify forecast horizon
                forecast_horizon = st.number_input("Enter the forecast horizon (number of periods to predict beyond test data):", min_value=1, max_value=256, value=30)

                # Set test size as 20% of the dataset
                test_size = max(1, int(0.2 * len(times_fm_df_actual)))

                # Check dataset length
                if len(times_fm_df_actual) < 2 * test_size:
                    st.warning("Dataset is too short for the specified test size and forecast horizon.")
                    return

                # Split into train and test sets
                train_df = times_fm_df_actual.iloc[:-test_size]
                test_df = times_fm_df_actual.iloc[-test_size:]

                if st.button('Start TimesFM Training'):
                    # Convert DataFrame to CSV buffer
                    relevant_columns = [times_fm_date_column, times_fm_actual_column]
                    if times_fm_regressor_column:
                        relevant_columns.append(times_fm_regressor_column)

                    times_fm_train_csv_buffer = StringIO()
                    train_df[relevant_columns].to_csv(times_fm_train_csv_buffer, index=False, lineterminator='\n')
                    times_fm_train_csv_buffer.seek(0)

                    # Send training data to the model
                    model_payload = {
                        'csv_data': times_fm_train_csv_buffer.getvalue(),
                        'date_column': times_fm_date_column,
                        'target_column': times_fm_actual_column,
                    }
                    if times_fm_regressor_column:
                        model_payload['regressor_column'] = times_fm_regressor_column

                    response = requests.post('http://localhost:5000/model', json=model_payload)

                    if response.status_code == 200:
                        st.success("TimesFM Model run completed. Fetching results...")
                        response = requests.get('http://localhost:5000/get_model_response')

                        if response.status_code == 200:
                            try:
                                global_times_fm_model_response = response.json()
                                st.write("TimesFM Model response received!")
                            except requests.exceptions.JSONDecodeError:
                                st.error("Failed to decode JSON response.")
                                return
                        else:
                            st.error("Failed to fetch TimesFM model response.")
                            return
                    
                    elif response.status_code == 401:
                        error_message = response.json().get('error', 'Unauthorized access. Please login again.')
                        st.error(error_message)
                        # Redirect to login page if 401 error occurs
                        nav_script = """
                            <meta http-equiv="refresh" content="0; url='http://localhost:5000'">
                        """
                        st.markdown(nav_script, unsafe_allow_html=True)
                    
                    else:
                        st.error("Failed to send data to the model.")
                        return

                    if global_times_fm_model_response:
                        forecasts = []

                        # Loop over each response (chunk of predictions)
                        for chunk_index, response_data in enumerate(global_times_fm_model_response):
                            predictions = response_data.get('predictions', [])
                            
                            if predictions:
                                for forecast in predictions:
                                    # Extract p50 from the forecast
                                    p50 = forecast.get('p70', [])
                                    
                                    # Add p50 (median) as the forecast value
                                    df_p50 = pd.DataFrame(p50, columns=[f'Forecast for Chunk {chunk_index}'])

                                    # Append the DataFrame to the list
                                    forecasts.append(df_p50)

                        if forecasts:
                            # Concatenate all p50 forecasts across all chunks
                            forecasts = pd.concat(forecasts, axis=1, ignore_index=True)
                            df_point = forecasts
                            
                            # Set the date range for the forecast (start from the beginning of the test set and extend beyond)
                            start_date = test_df[times_fm_date_column].min()  # Forecast starts from the beginning of test data
                            end_date = start_date + timedelta(days=forecast_horizon + test_size - 1)  # Extend by forecast horizon
                            date_range = pd.date_range(start=start_date, end=end_date, periods=len(df_point))

                            df_point[times_fm_date_column] = date_range

                            # Rename the value column
                            df_point.columns.values[-2] = 'value'
                            
                            st.subheader("Forecast DataFrame:")
                            st.dataframe(df_point[[times_fm_date_column, 'value']], use_container_width=True)
                            forecast_results = df_point[[times_fm_date_column, 'value']]

                            # Ensure the forecasted values aren't flat
                            if df_point['value'].nunique() == 1:
                                st.error("The forecasted values appear to be flat. Ensure variability in your model training data.")

                            times_fm_df_actual[times_fm_date_column] = pd.to_datetime(times_fm_df_actual[times_fm_date_column])

                            st.divider()
                            plot_forecast(train_df, test_df, df_point, times_fm_date_column, times_fm_actual_column, is_go=True)
                            
                            st.divider()
                            # Merge test data and forecasted data on date
                            aligned_df = pd.merge(test_df, df_point, on=times_fm_date_column, how='inner')
                            actual_values = aligned_df[times_fm_actual_column].values
                            predicted_values = aligned_df['value'].values

                            # Remove NaNs from actual_values and predicted_values
                            mask = (~np.isnan(actual_values)) & (~np.isnan(predicted_values))
                            actual_values = actual_values[mask]
                            predicted_values = predicted_values[mask]

                            if len(actual_values) == 0 or len(predicted_values) == 0:
                                st.error("No valid data points for accuracy calculation after removing NaNs.")
                            else:
                                calculate_accuracy(actual_values, predicted_values, flag=True)

                            st.divider()
                            decompose_time_series2(times_fm_df_actual, times_fm_actual_column, times_fm_date_column)

                            st.divider()
                            csv = forecast_results.to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name='predictions.csv',
                                mime='text/csv'
                            )
def run_auto_ml():
    st.subheader("AutoML")
    specific_dir = os.path.join(os.getcwd(), 'uploads')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="auto_forecast_file_uploader")
    
    if not os.path.exists(specific_dir):
        os.makedirs(specific_dir)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)            
        if df is not None:
            file_path = os.path.join(specific_dir, uploaded_file.name)

            date_column = st.selectbox("Select date column", [None] + list(df.columns), key="auto_forecast_date_column")
            target_column = st.selectbox("Select column to forecast", [None] + list(df.columns), key="auto_forecast_target_column")
            time_series_identifier = st.selectbox("Select time series identifier", [None] + list(df.columns), key="auto_forecast_time_series_identifier")
            # period = st.number_input("Forecast Period (days)", min_value=1, value=30, key="auto_forecast_period")
            
            cleaned_file_path = os.path.join(specific_dir, f"cleaned_{uploaded_file.name}")
            
            if st.button('Start AutoML Training'):
                with st.spinner('Training in progress... This may take up to 2.5 hours depending on the data. You will be notified by email once training is complete.'):
                    
                    # Clean column names and parse date column
                    df.columns = df.columns.str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True)
                    df[date_column] = df[date_column].apply(parse_date)
                    df = df.dropna(subset=[date_column])
                    st.dataframe(df)
                    df.to_csv(cleaned_file_path, index=False)

                    data = {
                        'file_path': cleaned_file_path,
                        'target_column': target_column,
                        'date_column': date_column,
                        'time_series_identifier': time_series_identifier
                    }

                    try:
                        # Try making the request to the Flask server
                        response = requests.post('http://127.0.0.1:5000/automl', data=data)
                        if response.status_code == 200:
                            st.success("Model finished training successfully")
                            response_data = response.json()
                            model_display_name = response_data.get("model_display_name")

                            st.write(f"Model {model_display_name} has been trained. You can deploy it and use it from the History Page.")
                        else:
                            st.error(f"Failed to start AutoML: {response.text}")
                    except requests.ConnectionError:
                        # Handle connection error and show the custom message
                        st.warning("The model is training and will take a few hours to complete. You will receive an email once it's done.")


def fetch_get_data_from_flask(endpoint):
    try:
        response = requests.get(f"{FLASK_SERVER_URL}/{endpoint}")
        print(response)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return {}


# Function to get required columns for an endpoint
@st.cache_data
def get_required_columns(endpoint_id):
    response = requests.get(COLUMNS_API_URL, params={"endpoint_id": endpoint_id})
    print(response)
    if response.status_code == 200:
        data = response.json()
        
        if 'columns' in data:
            return data['columns'], data['source']
        
        elif 'categorized_columns' in data:
            all_columns = data
            return all_columns, data['source']
        # return response.json().get('columns', [])
    else:
        st.error("Failed to retrieve required columns.")
        return []

# Define actions for deploy, delete, and predict
def deploy_model(endpoint_name, model):
    # endpoint_name = st.text_input("Enter endpoint name for deployment:")
    if endpoint_name:
        with st.spinner(f'Deploying {model["display_name"]}...'):
            response = requests.post(
                DEPLOY_MODEL_URL,
                json={"model_name": model['resource_name'], "endpoint_name": endpoint_name}
            )
            result = response.json()
            if response.status_code == 200:
                st.success(f"Model deployed to {result['endpoint_display_name']}")
            else:
                st.error(result.get('error', 'Deployment failed'))

def delete_endpoint(endpoint):
    with st.spinner(f'Deleting {endpoint["display_name"]}...'):
        response = requests.post(
            DELETE_ENDPOINT_URL,
            json={"endpoint_name": endpoint['resource_name']}
        )
        result = response.json()
        if response.status_code == 200:
            st.success("Endpoint deleted successfully")
        else:
            st.error(result.get('error', 'Deletion failed'))

def predict_in_batches_prophet(
    data, api_url, batch_size_limit, endpoint_id, forecast_horizon,
    date_column, target_column, identifier_column
):
    """Predict in batches using Prophet model via the API."""
    predictions = []

    try:
        unique_identifiers = data[identifier_column].unique()
        historical_dates = data[date_column].unique()
        latest_date = data[date_column].max()
        future_data = create_forecast_instances(
            unique_identifiers, historical_dates, latest_date,
            forecast_horizon, date_column, identifier_column
        )
        num_records = len(future_data)

        if num_records == 0:
            st.error("No records generated for future prediction.")
            return None

        batch_size = max(1, batch_size_limit)
        total_batches = math.ceil(num_records / batch_size)
        progress_bar = st.progress(0)

        for batch_num, i in enumerate(range(0, num_records, batch_size), start=1):
            batch_data = future_data[i:i+batch_size]
            
            try:
                # st.write(f"Processing batch {batch_num}/{total_batches}")
                batch_response = requests.post(
                    api_url,
                    json={
                        "instances": batch_data,
                        "endpoint_id": endpoint_id,
                        "forecast_horizon": forecast_horizon
                    }
                )

                if batch_response.status_code == 200:
                    batch_predictions = batch_response.json().get('predictions', [])
                    predictions.extend(batch_predictions)
                else:
                    st.error(f"Error from API (Status Code: {batch_response.status_code}): {batch_response.text}")
                    progress_bar.empty()
                    return None

                # Update the progress bar
                progress = int((batch_num / total_batches) * 100)
                progress_bar.progress(progress)

            except Exception as e:
                st.error(f"Error in batch {batch_num}: {str(e)}")
                progress_bar.empty()
                return None

        progress_bar.empty()
        return predictions

    except Exception as e:
        st.error(f"Error during batch prediction: {str(e)}")
        return None


def process_prophet_predictions(predictions, filtered_data, date_column, target_column, identifier_column):
    """Process the predictions returned by the Prophet model."""
    if predictions is None or len(predictions) == 0:
        st.error("No predictions returned from Prophet model.")
        return None, None

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Since date_column and predicted values are lists, extract the first element
    predictions_df[date_column] = predictions_df[date_column].apply(lambda x: x[0])
    predictions_df[f'predicted_{target_column}'] = predictions_df[f'predicted_{target_column}'].apply(lambda x: x[0])

    # Convert date_column to datetime and remove timezone
    predictions_df[date_column] = pd.to_datetime(predictions_df[date_column]).dt.tz_localize(None)

    # Ensure date_column in filtered_data is also datetime without timezone
    filtered_data[date_column] = pd.to_datetime(filtered_data[date_column]).dt.tz_localize(None)

    # Merge predictions with filtered_data on date_column and identifier_column
    merged_data = pd.merge(
        filtered_data,
        predictions_df[[identifier_column, date_column, f'predicted_{target_column}']],
        on=[identifier_column, date_column],
        how='left'
    )

    # Rename the predicted column to 'Predicted' in merged_data
    merged_data.rename(columns={f'predicted_{target_column}': 'Predicted'}, inplace=True)

    # For future dates (dates beyond the last date in filtered_data), create future_df
    future_predictions_df = predictions_df[~predictions_df.set_index([identifier_column, date_column]).index.isin(merged_data.set_index([identifier_column, date_column]).index)]

    # Rename the predicted column to 'Predicted' in future_predictions_df
    future_predictions_df.rename(columns={f'predicted_{target_column}': 'Predicted'}, inplace=True)

    # Select the necessary columns
    future_df = future_predictions_df[
        [identifier_column, date_column, 'Predicted']
    ]

    return merged_data, future_df


## automl predicting

def predict_in_batches(data, api_url, batch_size_limit, endpoint_id, forecast_horizon, date):
        predictions = []
        num_records = len(data)
        batch_size = int(num_records / math.ceil((data.memory_usage(index=True, deep=True).sum() / batch_size_limit)))
        
        for i in range(0, num_records, batch_size):
            batch_data = data.iloc[i:i+batch_size].to_dict(orient="records")
            
            # Ensure all dates are in the correct ISO format (%Y-%m-%d)
            for record in batch_data:
                if isinstance(record[date], pd.Timestamp):
                    record[date] = record[date].strftime('%Y-%m-%d')
            # Send the POST request with the forecast horizon included
            batch_response = requests.post(api_url, json={"data": batch_data, "endpoint_id": endpoint_id, "forecast_horizon": forecast_horizon})
            if batch_response.status_code == 200:
                batch_predictions = batch_response.json().get('predictions')
                # st.write(batch_predictions)
                predictions.extend([pred['value'] for pred in batch_predictions])
            else:
                st.error(f"Error from Flask: {batch_response.text}")
                return None
        # batch_response = requests.post(api_url, json={"data": data, "endpoint_id": endpoint_id, "forecast_horizon": forecast_horizon})
        # if batch_response.status_code == 200:
        #         batch_predictions = batch_response.json().get('predictions')
        #         predictions.extend([pred['value'] for pred in batch_predictions])
        return predictions

### Prophet prediction
def generate_future_dates(start_date, horizon):
    """Generate future dates for the forecast horizon."""
    return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, horizon + 1)]

def create_forecast_instances(identifiers, historical_dates, start_date, horizon, date_column, identifier_column):
    """Create forecast instances with historical and future dates."""
    instances = []
    future_dates = generate_future_dates(start_date, horizon)
    all_dates = list(historical_dates) + future_dates

    for identifier in identifiers:
        for date in all_dates:
            date_str = str(date) if isinstance(date, str) else date.strftime('%Y-%m-%d')
            instance = {
                date_column: [date_str],  # Use date_column variable
                identifier_column: identifier  # The identifier (e.g., cell)
            }
            instances.append(instance)
    return instances

@st.cache_data 
def filter_and_prepare_data(data, date_column, target_column, selected_category=None, category_column=None, user_selections={}):
    """Filter and prepare the data based on user selections."""
    
    # Ensure the date column is properly converted to datetime
    data[date_column], fmt = parse_dates(data[date_column])
    
    # Drop rows where date conversion failed (if any)
    data = data.dropna(subset=[date_column])

    data = data.sort_values(by=[date_column], inplace=False)

    if selected_category and category_column:
        data = data[data[category_column] == selected_category]

    # Rename columns to match the model's expected input
    renamed_columns = {date_column: date_column, target_column: target_column}
    for column in user_selections:
        renamed_columns[user_selections[column]] = column
    
    data = data.rename(columns=renamed_columns)

    # Convert other selected columns to string
    for col in user_selections:
        data[user_selections[col]] = data[user_selections[col]].astype(str)

    return data


def endpoint_predict(selected_endpoint_id):
    API_URL = 'http://localhost:5000/predict'
    BATCH_SIZE_LIMIT = 1.4 * 1024 * 1024  # 1.4MB to stay safely under the 1.5MB limit

    st.title("Predictions")
        # Initialize session state variables
    # if 'combined_data' not in st.session_state:
    st.session_state['combined_data'] = None
    # if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = None
    # if 'future_df' not in st.session_state:
    st.session_state['future_df'] = None
    # if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
    # if 'compute_triggered' not in st.session_state:
    st.session_state['compute_triggered'] = False
    # if 'decomposition_plots' not in st.session_state:
    st.session_state['decomposition_plots'] = {}
    # if 'fig' not in st.session_state:
    st.session_state['fig'] = None
        
    if selected_endpoint_id:
        required_columns, source = get_required_columns(selected_endpoint_id)

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if source == 'metadata' and uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True)
        st.subheader("Data Preview:")
        st.dataframe(data.head(),  use_container_width=True)

        columns = data.columns.tolist()
        date_column = st.selectbox("Select the Date column", columns)
        target_column = st.selectbox("Select the Target column", columns)
        category_column = st.selectbox("Select the filter column (optional)", ['None'] + columns)

        selected_category = None
        if category_column != 'None':
            unique_categories = data[category_column].unique().tolist()
            selected_category = st.selectbox(f"Select {category_column} value to filter by", unique_categories)

        user_selections = {}
        for column in required_columns:
            if column not in [date_column, target_column]:
                user_selections[column] = st.selectbox(f"Select the {column} column", columns)

        forecast_horizon = st.number_input("Enter the forecast horizon (number of future periods to predict):", min_value=1, value=7)
        if st.button("Predict"):
            st.session_state.filtered_data = None
            st.session_state.compute_triggered = None
            st.session_state.predictions = None
            st.session_state.future_df = None
            try:
                # Check if data needs to be recomputed
                if st.session_state.filtered_data is None or st.session_state.compute_triggered:
                    st.session_state.filtered_data = filter_and_prepare_data(data, date_column, target_column, selected_category, category_column, user_selections)
                    st.session_state.compute_triggered = False
                
                filtered_data = st.session_state.filtered_data

                # Handle predictions only if not already computed
                if st.session_state.predictions is None:
                    # st.session_state.predictions = handle_predictions(filtered_data, forecast_horizon, date_column, 'http://localhost:5000/predict', 1.4 * 1024 * 1024, selected_endpoint_id)
                    st.session_state.predictions  = predict_in_batches(filtered_data, API_URL, BATCH_SIZE_LIMIT, selected_endpoint_id, forecast_horizon, date_column)
                filtered_data['Predicted'] = st.session_state.predictions
                
                train_data = filtered_data[filtered_data[date_column] < filtered_data[date_column].max() - pd.Timedelta(days=90)]
                test_data = filtered_data[filtered_data[date_column] >= filtered_data[date_column].max() - pd.Timedelta(days=90)]
                # Generate future predictions DataFrame
                last_date = filtered_data[date_column].max()
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='D')[1:]
                future_predictions = st.session_state.predictions[-forecast_horizon:]
                st.session_state.future_df = pd.DataFrame({date_column: future_dates, 'Predicted': future_predictions})
                st.subheader("Forecasted Data:")
                st.dataframe(st.session_state.future_df,  use_container_width=True)          
                st.divider()
                # Compute accuracy, decompose time series, and plot results
                calculate_accuracy(filtered_data[target_column], filtered_data['Predicted'])
                st.divider()
                
                plot_forecast(train_data, test_data, filtered_data, date_column, target_column,forecast_horizon, True)
                st.divider()
                decompose_time_series2(filtered_data, target_column, date_column)
                # Combine historical and future predictions for download
                combined_data = pd.concat([filtered_data, st.session_state.future_df], ignore_index=True)

                csv = combined_data.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Data processing failed: {e}")
    #Prophet
    elif source == 'labels':
        if uploaded_file is not None:
            
            data = pd.read_csv(uploaded_file)
            st.subheader("Data Preview:")
            st.dataframe(data.head(),  use_container_width=True)
        
            # Prophet model column selection# 
            identifier_column_flag = required_columns.get('categorized_columns', {}).get('identifier_column', {})[0]
            target_column_flag = required_columns.get('categorized_columns', {}).get('target_column', [])[0]
            columns = required_columns.get('categorized_columns', {}).get('feature_columns', [])
            if target_column_flag:
                columns.append(target_column_flag)
                columns.append(identifier_column_flag)
            date_column = st.selectbox("Select the Date column", columns)
            identifier_column = st.selectbox(f"Select the Identifier column {identifier_column_flag}", columns)
            target_column = st.selectbox(f"Select the Target column {target_column_flag}", columns)
            # Optional category filter
            category_column = st.selectbox("Select the filter column (optional)", ['None'] + columns)
            selected_category = None
            if category_column != 'None':
                unique_categories = data[category_column].unique().tolist()
                selected_category = st.selectbox(f"Select {category_column} value to filter by", unique_categories)
            
            forecast_horizon = st.number_input("Enter the forecast horizon (number of future periods to predict):", min_value=1, value=30)
            select_identifier = st.checkbox("Select a specific identifier for analysis", value=False)

            # Compute button for predictions
            if st.button("Predict"):
                try:
                    data[date_column], format_used = parse_date_prophet(data[date_column])
                
                    # Convert to datetime using pd.to_datetime with the identified format
                    data[date_column] = pd.to_datetime(data[date_column], format=format_used).dt.tz_localize(None)
                
                    

                    if data[date_column].isnull().any():
                        st.error("Some dates could not be parsed. Please check the date column for inconsistencies.")
                    else:
                        # Apply category filter if selected
                        filtered_data = data.copy()
                        if selected_category:
                            filtered_data = filtered_data[filtered_data[category_column] == selected_category]

                        # Ensure target column is numeric
                        try:
                            filtered_data[target_column] = pd.to_numeric(filtered_data[target_column], errors='coerce')
                            if filtered_data[target_column].isnull().any():
                                st.error("Some target values could not be converted to numeric. Please check the target column for inconsistencies.")
                                st.write("Preview of problematic rows:")
                                st.write(filtered_data[filtered_data[target_column].isnull()])
                                raise ValueError("Target column contains non-numeric values.")
                        except Exception as e:
                            st.error(f"Failed to convert target column to numeric: {e}")
                            raise e

                        filtered_data.sort_values(by=[date_column], inplace=False)
                        
                        # Prophet predictions
                        st.write("Starting batch predictions with Prophet model...")
                        predictions = predict_in_batches_prophet(
                            filtered_data,
                            API_URL_PROPHET,
                            100,
                            selected_endpoint_id,
                            forecast_horizon,
                            date_column,
                            target_column,
                            identifier_column,
                        )
                        if predictions is not None:
                            st.write("Batch predictions completed.")

                            # Process the Prophet predictions
                            merged_data, future_df = process_prophet_predictions(
                                predictions,
                                filtered_data,
                                date_column,
                                target_column,
                                identifier_column
                            )
                            st.write("merged:")
                            st.write(merged_data)
                            st.write('future')
                            st.write(future_df)
                            

                            if merged_data is not None:
                                # Combine historical and future predictions
                                combined_data = pd.concat([merged_data, future_df], ignore_index=True)
                                combined_data.sort_values(by=[date_column], inplace=True)
                                st.session_state.combined_data = combined_data

                                # Store data in session state
                                st.session_state.filtered_data = merged_data
                                st.session_state.future_df = future_df
                                st.session_state.predictions = merged_data['Predicted']
                                st.session_state.compute_triggered = True

                                # Calculate overall error metrics
                                actual = merged_data[target_column]
                                predicted = merged_data['Predicted']

                                # Drop NaNs to compute metrics
                                valid_idx = (~actual.isna()) & (~predicted.isna())
                                actual_valid = actual[valid_idx]
                                predicted_valid = predicted[valid_idx]
                                st.divider()
                                st.write(valid_idx)
                                st.write(combined_data)
                                train_data = filtered_data[filtered_data[date_column] < filtered_data[date_column].max() - pd.Timedelta(days=90)]
                                test_data = filtered_data[filtered_data[date_column] >= filtered_data[date_column].max() - pd.Timedelta(days=90)]
                                 
                                
                                if len(actual_valid) > 0:
                                    st.divider()
                                    calculate_accuracy(merged_data[target_column], merged_data['Predicted'])
                                st.divider()
                                plot_forecast(train_data, test_data, combined_data, date_column, target_column, forecast_horizon, predict_flag=True)
                                st.divider()
                                decompose_time_series2(filtered_data, target_column, date_column)
                                st.divider()
                                csv = combined_data.to_csv(index=False)
                                st.download_button(
                                    label="Download data as CSV",
                                    data=csv,
                                    file_name='predictions.csv',
                                    mime='text/csv'
                                )
                        

                except Exception as e:
                    pass
                    st.error(f"Data processing failed: {e}")



def run_prophet_training():
    FLASK_SERVER_URL = 'http://localhost:5000'

    # Section 1: Upload CSV and specify columns
    st.title("Upload CSV and Train a Prophet Model")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read uploaded file into a pandas dataframe
        df = pd.read_csv(uploaded_file)

        # Display a preview of the uploaded data
        st.write("Preview of the uploaded data:")
        st.dataframe(df.head())

        # Input fields for target, date, and identifier columns
        target_column = st.selectbox("Select the Target Column", df.columns)
        date_column = st.selectbox("Select the Date Column", df.columns)
        identifier_column = st.selectbox("Select the Identifier Column", df.columns)

        # Input for forecast horizon (make it dynamic)
        forecast_horizon = st.number_input(
            "Enter the Forecast Horizon (in days)", min_value=1, max_value=365, value=30)

        # Feature names (exclude target, date, and identifier columns)
        feature_names = [
            col for col in df.columns if col not in[identifier_column]]

        if st.button("Upload to BigQuery and Start Training"):
            # Convert the DataFrame to CSV and send it to the Flask app
            csv_data = df.to_csv(index=False)

            # Prepare the payload for the data upload
            upload_payload = {
                "csv_data": csv_data,
                "date_column": date_column,
                "identifier_column": identifier_column,
            }

            # Step 1: Upload the data to BigQuery
            upload_response = requests.post(f'{FLASK_SERVER_URL}/upload_data_prophet', json=upload_payload)

            if upload_response.status_code == 200:
                st.success("Data uploaded to BigQuery successfully!")
                
                # Retrieve the table name from the upload response
                table_name = upload_response.json().get("table_name")

                if not table_name:
                    st.error("Failed to retrieve table name after data upload.")
                    st.stop()

                # Prepare the payload for processing and training
                process_payload = {
                    "table_name": table_name,
                    "target_column": target_column,
                    "date_column": date_column,
                    "identifier_column": identifier_column,
                    "forecast_horizon": int(forecast_horizon),
                    "feature_names": feature_names,
                }

                # Step 2: Start the processing and training pipeline
                process_response = requests.post(f'{FLASK_SERVER_URL}/process_and_train', json=process_payload)

                if process_response.status_code == 200:
                    st.success("Training pipeline started successfully!")
                    st.write("Job ID:", process_response.json().get("job_id"))
                else:
                    st.error(f"Failed to start the training pipeline: {process_response.text}")

            else:
                st.error(f"Failed to upload data to BigQuery: {upload_response.text}")



def run_prophet_batch_predictions(model_name):
    FLASK_SERVER_URL = 'http://localhost:5000'

    # Section 1: Upload CSV and specify columns
    st.title("Upload CSV and Run Batch Predictions")
    st.write(model_name)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read uploaded file into a pandas dataframe
        df = pd.read_csv(uploaded_file)

        # Display a preview of the uploaded data
        st.write("Preview of the uploaded data:")
        st.dataframe(df.head())

        # Input fields for target, date, and identifier columns
        target_column = st.selectbox("Select the Target Column", df.columns)
        date_column = st.selectbox("Select the Date Column", df.columns)
        identifier_column = st.selectbox("Select the Identifier Column", df.columns)

        # Input for forecast horizon (make it dynamic)
        forecast_horizon = st.number_input(
            "Enter the Forecast Horizon (in days)", min_value=1, max_value=365, value=30)

        # Feature names (exclude target, date, and identifier columns)
        feature_names = [
            col for col in df.columns if col not in[identifier_column]]

        if st.button("Upload to BigQuery and Start Training"):
            # Convert the DataFrame to CSV and send it to the Flask app
            csv_data = df.to_csv(index=False)

            # Prepare the payload for the data upload
            upload_payload = {
                "csv_data": csv_data,
                "date_column": date_column,
                "identifier_column": identifier_column,
            }

            # Step 1: Upload the data to BigQuery
            upload_response = requests.post(f'{FLASK_SERVER_URL}/upload_data_prophet', json=upload_payload)

            if upload_response.status_code == 200:
                st.success("Data uploaded to BigQuery successfully!")
                
                # Retrieve the table name from the upload response
                table_name = upload_response.json().get("table_name")

                if not table_name:
                    st.error("Failed to retrieve table name after data upload.")
                    st.stop()

                # Prepare the payload for processing and training
                process_payload = {
                    "model_name": model_name, 
                    "table_name": table_name,
                    "target_column": target_column,
                    "date_column": date_column,
                    "identifier_column": identifier_column,
                    "forecast_horizon": int(forecast_horizon),
                    "feature_names": feature_names,
                }

                # Step 2: Start the processing and training pipeline
                process_response = requests.post(f'{FLASK_SERVER_URL}/prophet_batch_predictions', json=process_payload)

                if process_response.status_code == 200:
                    st.success("Prediction pipeline started successfully!")
                    st.write("Job ID:", process_response.json().get("job_id"))
                else:
                    st.error(f"Failed to start the training pipeline: {process_response.text}")

            else:
                st.error(f"Failed to upload data to BigQuery: {upload_response.text}")

def deploy_model(endpoint_name, model_resource_name):
    try:
        # Send a POST request with the model resource ID and custom endpoint name
        response = requests.post(f'{FLASK_SERVER_URL}/deploy_model', json={
            "model_name": model_resource_name, 
            "endpoint_name": endpoint_name
        })
        
        if response.status_code == 200:
            try:
                endpoint_info = response.json()
                st.success("Model deployed successfully!")
                # Add a note to guide the user to the history page
                st.info("Now that the endpoint has been deployed, please go to the **History Page** to select the endpoint for prediction.")

            except ValueError:
                st.error("Failed to decode JSON response from the server.")
        else:
            # Try to parse the error message
            try:
                error_info = response.json()
                error_message = error_info.get('error', 'An error occurred while deploying the model.')
            except ValueError:
                error_message = response.text or 'An error occurred while deploying the model.'
            st.error(f"Failed to deploy model: {error_message}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
