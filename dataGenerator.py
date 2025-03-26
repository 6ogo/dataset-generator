import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import plotly as plt

def analyze_distribution(data):
    """Analyze the statistical distribution of the data"""
    # Test for normality
    _, p_value = stats.normaltest(data)
    if p_value > 0.05:
        return 'normal'
    
    # Test for other distributions
    distributions = ['gamma', 'lognormal', 'exponential']
    best_fit = None
    best_kstest = float('inf')
    
    for dist in distributions:
        if dist == 'gamma':
            params = stats.gamma.fit(data)
            ks_stat = stats.kstest(data, 'gamma', params)[0]
        elif dist == 'lognormal':
            params = stats.lognorm.fit(data)
            ks_stat = stats.kstest(data, 'lognorm', params)[0]
        elif dist == 'exponential':
            params = stats.expon.fit(data)
            ks_stat = stats.kstest(data, 'expon', params)[0]
            
        if ks_stat < best_kstest:
            best_kstest = ks_stat
            best_fit = dist
            
    return best_fit

def generate_simulated_data(existing_data, column, start_date, end_date):
    """Generate realistic simulated data based on statistical analysis"""
    # Analyze existing data
    data = existing_data[column].dropna()
    
    # Basic statistics
    mean = data.mean()
    std = data.std()
    skew = stats.skew(data)
    
    # Identify seasonality (if date column exists)
    seasonality = None
    if 'date' in existing_data.columns:
        existing_data['date'] = pd.to_datetime(existing_data['date'])
        seasonal_decompose = pd.DataFrame({
            'month': existing_data['date'].dt.month,
            'value': existing_data[column]
        }).groupby('month').mean()
        seasonality = seasonal_decompose['value'].to_dict()
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_records = len(date_range)
    
    # Determine distribution
    dist_type = analyze_distribution(data)
    
    # Generate base values
    if dist_type == 'normal':
        simulated_values = np.random.normal(mean, std, num_records)
    elif dist_type == 'gamma':
        shape, loc, scale = stats.gamma.fit(data)
        simulated_values = stats.gamma.rvs(shape, loc=loc, scale=scale, size=num_records)
    elif dist_type == 'lognormal':
        shape, loc, scale = stats.lognorm.fit(data)
        simulated_values = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=num_records)
    else:  # fallback to exponential
        loc, scale = stats.expon.fit(data)
        simulated_values = stats.expon.rvs(loc=loc, scale=scale, size=num_records)
    
    # Apply seasonality if detected
    if seasonality:
        seasonal_factors = [seasonality[date.month] / mean for date in date_range]
        simulated_values *= seasonal_factors
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, std * 0.1, num_records)
    simulated_values += noise
    
    # Ensure values stay within reasonable bounds
    min_val = data.min()
    max_val = data.max()
    simulated_values = np.clip(simulated_values, min_val * 0.9, max_val * 1.1)
    
    # Create DataFrame
    simulated_data = pd.DataFrame({
        'date': date_range,
        column: simulated_values
    })
    
    return simulated_data

# Streamlit app
st.title("Advanced Data Simulator")
st.write("Upload your dataset to generate realistic simulated data based on statistical analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])

if uploaded_file:
    try:
        # Load the uploaded file
        existing_data = pd.read_excel(uploaded_file)
        st.write("Uploaded Dataset Preview:")
        st.dataframe(existing_data.head())
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        st.write(existing_data.describe())
        
        # Select column for simulation
        numeric_columns = existing_data.select_dtypes(include=[np.number]).columns
        column = st.selectbox("Select column to simulate", numeric_columns)
        
        # Select date range for simulation
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime(2024, 1, 1))
        with col2:
            end_date = st.date_input("End date", datetime(2025, 12, 31))
        
        if st.button("Generate Simulated Data"):
            with st.spinner("Analyzing data and generating simulation..."):
                simulated_data = generate_simulated_data(existing_data, column, start_date, end_date)
                
                st.subheader("Simulated Data Preview")
                st.dataframe(simulated_data.head())
                
                # Plot comparison
                st.subheader("Data Distribution Comparison")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                existing_data[column].hist(ax=ax1, bins=30)
                ax1.set_title("Original Data Distribution")
                
                simulated_data[column].hist(ax=ax2, bins=30)
                ax2.set_title("Simulated Data Distribution")
                
                st.pyplot(fig)
                
                # Download button
                csv = simulated_data.to_csv(index=False)
                st.download_button(
                    label="Download Simulated Data (CSV)",
                    data=csv,
                    file_name="simulated_data.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")