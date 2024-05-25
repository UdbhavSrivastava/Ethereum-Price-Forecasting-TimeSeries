import streamlit as st
import pandas as pd
from nixtla import NixtlaClient
import numpy as np
from sklearn.metrics import mean_squared_error

# Function to provide a descriptive paragraph about the dataset
# Function to provide a descriptive paragraph about the dataset
def dataset_description(df):
    start_date = df['ds'].min().strftime('**%B %d, %Y**')
    end_date = df['ds'].max().strftime('**%B %d, %Y**')
    num_instances = len(df)
    
    st.write("## Dataset Description")
    st.write(f"The dataset contains cryptocurrency data for Ethereum, a decentralized, open-source blockchain system. It spans from {start_date} to {end_date} and includes a total of {num_instances} instances. Ethereum is known for its smart contract functionality, which enables developers to build decentralized applications on its blockchain.")

# Function to perform partition 1 - Show dataset plot and info
def partition_1_dataset(df):
    nixtla_client = NixtlaClient(api_key ='nixt-XAcoYms3zqRSmcLprr3qm4nlVHBWgaIkmG57hcmTphfkSAi8pg1dO4FPKifn515olUDXNOklqKx4HRLh')
    
    # Display dataset description
    dataset_description(df)

    # Display dataset plot
    st.write("## Dataset Plot")
    st.plotly_chart(nixtla_client.plot(df, time_col='ds', target_col='y', engine='plotly'),use_container_width=True)
    
# Function to perform partition 2 - Model info
def partition_2_model_info(df):
    nixtla_client = NixtlaClient(api_key ='nixt-XAcoYms3zqRSmcLprr3qm4nlVHBWgaIkmG57hcmTphfkSAi8pg1dO4FPKifn515olUDXNOklqKx4HRLh')

    # Splitting data
    train = df[:-63]
    test = df[-63:]

    # Forecasting on testing data
    preds_df = nixtla_client.forecast(df=train, h=63, finetune_steps=50,freq = 'D',
   finetune_loss='default', model='timegpt-1-long-horizon', time_col='ds', target_col='y')
    preds_df = preds_df.rename(columns={'timestamp': 'ds'})
    
    rmse = np.sqrt(mean_squared_error(test['y'], preds_df['TimeGPT']))
    # Plotting actual vs predicted
    st.write("## Model Info - Actual vs Predicted")
    st.write("The model was trained on the data up until **February 29, 2024**, and tested on data until **May 2, 2024**.")
    # Displaying RMSE
    st.write(f'Root Mean Squared Error (RMSE): **{rmse}**')
    
    st.plotly_chart(nixtla_client.plot(test, preds_df, time_col='ds', target_col='y', engine='plotly'))


# Function to perform partition 3 - Future forecasting
def partition_3_forecasting(df, forecast_date):
    nixtla_client = NixtlaClient(api_key ='nixt-XAcoYms3zqRSmcLprr3qm4nlVHBWgaIkmG57hcmTphfkSAi8pg1dO4FPKifn515olUDXNOklqKx4HRLh')

    # Convert forecast_date to pd.Timestamp object
    forecast_date = pd.Timestamp(forecast_date)

    # Convert last date in the dataset to pd.Timestamp object
    last_date = df['ds'].max()

    # Calculate forecast horizon
    forecast_horizon = (forecast_date - last_date).days
    print(forecast_horizon)
    # Forecasting
    timegpt_future_df = nixtla_client.forecast(df=df, h=forecast_horizon, freq='D', time_col='ds', model='timegpt-1-long-horizon',finetune_steps=50,
        finetune_loss='default', target_col='y')
    # Renaming columns

    timegpt_future_df = timegpt_future_df[::-1]
    # Plot
    st.write("## Future Forecasting")
    st.plotly_chart(nixtla_client.plot(df, timegpt_future_df, time_col='ds', target_col='y', engine='plotly'))
    # Display future forecasting dataframe
    st.write("## Future Forecasting Data")
    st.dataframe(timegpt_future_df)
# Main Streamlit app
def main():
    st.title("Ethereum Price Forecasting using Time GPT Model")
    option = st.sidebar.selectbox("Select:", ("Dataset Plot and Info", "Model Metrics - RMSE", "Future Forecasting"))

    df = pd.read_csv('CBETHUSD.csv')
    df = df.rename(columns={'DATE': 'ds', 'CBETHUSD': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')

    if option == "Dataset Plot and Info":
        partition_1_dataset(df)
    elif option == "Model Metrics - RMSE":
        partition_2_model_info(df)
    elif option == "Future Forecasting":
        forecast_date = st.sidebar.date_input("Select Forecast Date", min_value=pd.to_datetime('now').date(), max_value=pd.to_datetime('2025-12-31').date())
        partition_3_forecasting(df, forecast_date)

if __name__ == "__main__":
    main()
