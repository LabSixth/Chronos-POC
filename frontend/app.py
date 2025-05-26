"""
Streamlit application for stock price prediction visualization.

This module provides a web interface to visualize the results of stock price
prediction models (Prophet and Chronos), allowing users to select dates and
compare predicted values with actual prices.
"""

import os
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import aws_ingress

# Set page title
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# App title
st.title("Stock Closing Price Prediction")


@st.cache_data
def load_prediction_data():
    """
    Load and preprocess prediction data from CSV files.

    Returns:
        tuple: Prophet and Chronos prediction dataframes
    """
    prophet_predictions = pd.read_csv("data/prophet_preds.csv")
    chronos_predictions = pd.read_csv("data/chronos_preds.csv")

    # Convert date column to datetime type
    prophet_predictions['ds'] = pd.to_datetime(prophet_predictions['ds'])
    chronos_predictions['ds'] = pd.to_datetime(chronos_predictions['ds'])

    return prophet_predictions, chronos_predictions


def get_visualization_paths():
    """
    Get paths to model performance visualization images.

    Returns:
        tuple: Paths to Prophet and Chronos model visualization images
    """
    prophet_path = "data/Prophet_predicted_vs_real.png"
    chronos_path = "data/Chronos_predicted_vs_real.png"

    return prophet_path, chronos_path


# Main program
try:
    # Call AWS data ingress
    aws_ingress.get_s3_data()

    # Load data
    prophet_preds, chronos_preds = load_prediction_data()
    prophet_img_path, chronos_img_path = get_visualization_paths()

    # Create two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Model")
        model_option = st.selectbox(
            "Choose prediction model",
            ["Prophet", "Chronos"]
        )

    with col2:
        st.subheader("Select Date")
        # Get prediction date range
        min_date = prophet_preds['ds'].min().date()
        max_date = prophet_preds['ds'].max().date()

        # Date picker
        selected_date = st.date_input(
            "Select prediction date",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )

    # Display the prediction for the selected date
    st.header("Prediction Results")

    # Select appropriate prediction data
    selected_model_name = model_option  # Not a constant, based on user selection
    # pylint: disable=invalid-name
    if model_option == "Prophet":
        preds_df = prophet_preds
        img_path = prophet_img_path  # Local variable for readability
    else:
        preds_df = chronos_preds
        img_path = chronos_img_path  # Local variable for readability
    # pylint: enable=invalid-name

    # Find prediction for selected date
    # Convert to string to avoid date.date attribute error
    date_str = selected_date.strftime('%Y-%m-%d')
    selected_pred = preds_df[preds_df['ds'].dt.strftime('%Y-%m-%d') == date_str]

    if not selected_pred.empty:
        predicted_price = selected_pred['0'].values[0]

        # Display predicted price
        st.metric(
            label=f"Predicted Closing Price for {selected_date}",
            value=f"${predicted_price:.2f}"
        )

        # Create nearby dates prediction display
        st.subheader("Nearby Dates Prediction")

        # Get predictions for dates before and after the selected date
        start_date = selected_date - timedelta(days=3)
        end_date = selected_date + timedelta(days=3)

        # Create date strings for filtering
        date_strings = [
            (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range((end_date - start_date).days + 1)
        ]
        nearby_dates = preds_df[preds_df['ds'].dt.strftime('%Y-%m-%d').isin(date_strings)]

        if not nearby_dates.empty:
            nearby_dates = nearby_dates.sort_values('ds')

            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nearby_dates['ds'],
                y=nearby_dates['0'],
                mode='lines+markers',
                name='Predicted Price'
            ))

            # Highlight selected date but without a name tag
            fig.add_trace(go.Scatter(
                x=[selected_pred['ds'].values[0]],
                y=[predicted_price],
                mode='markers',
                marker={'size': 12, 'color': 'red'},
                showlegend=False
            ))

            fig.update_layout(
                title="Predicted Prices for Nearby Dates",
                xaxis_title="Date",
                yaxis_title="Closing Price ($)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No prediction data available for the selected date {selected_date}.")

    # Display model performance image
    st.header(f"{selected_model_name} Model Performance")
    if os.path.exists(img_path):
        st.image(
            img_path,
            caption=f"{selected_model_name} Model Predicted vs. Actual Values"
        )
    else:
        st.warning(
            f"Could not find performance image for {selected_model_name} model."
        )

except (FileNotFoundError, pd.errors.EmptyDataError) as file_error:
    st.error(f"Error loading data files: {file_error}")
    st.info("Please make sure all required files are in the Model_Artifacts directory.")
except ValueError as value_error:
    st.error(f"Data processing error: {value_error}")
    st.info("There might be an issue with the format of the prediction data.")
except Exception as e:  # pylint: disable=broad-except
    # Keep this as a fallback but disable the pylint warning
    st.error(f"Unexpected error: {e}")
    st.info("Please check the application logs for details.")

# Add instructions
with st.expander("Usage Instructions"):
    st.write("""
    This application displays stock price prediction results:

    1. Select the prediction model you want to use (Prophet or Chronos)
    2. Select the date for which you want to see the prediction
    3. View the predicted closing price and trend for nearby dates

    The charts show the comparison between model predictions and actual prices.
    """)