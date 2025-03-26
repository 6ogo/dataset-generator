import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import chardet
import csv
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

def detect_encoding(file_content):
    """Detect the encoding of a file"""
    result = chardet.detect(file_content)
    return result['encoding'] or 'utf-8'

def detect_delimiter(file_content, encoding):
    """Detect the delimiter of a text file"""
    content_str = file_content.decode(encoding)
    delimiters = [',', ';', '\t', '|']
    
    try:
        dialect = csv.Sniffer().sniff(content_str[:1024])
        return dialect.delimiter
    except:
        counts = {d: content_str.count(d) for d in delimiters}
        max_delimiter = max(counts, key=counts.get)
        return max_delimiter if counts[max_delimiter] > 0 else ','

def load_data(uploaded_file):
    """Load data with automatic format detection"""
    try:
        file_content = uploaded_file.read()
        encoding = detect_encoding(file_content)
        
        if uploaded_file.name.endswith('.xlsx'):
            try:
                return pd.read_excel(io.BytesIO(file_content))
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        
        elif uploaded_file.name.endswith(('.csv', '.txt')):
            delimiter = detect_delimiter(file_content, encoding)
            
            try:
                df = pd.read_csv(
                    io.BytesIO(file_content),
                    encoding=encoding,
                    sep=delimiter,
                    engine='python'
                )
                st.success(f"File loaded successfully with encoding: {encoding} and delimiter: '{delimiter}'")
                return df
            except Exception as e1:
                st.warning("First attempt failed, trying alternative loading method...")
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_content),
                        encoding=encoding,
                        sep=delimiter,
                        engine='python',
                        on_bad_lines='skip'
                    )
                    st.info("File loaded with some rows skipped due to parsing issues")
                    return df
                except Exception as e2:
                    st.error(f"Failed to load file: {str(e2)}")
                    return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def analyze_distribution_pattern(data, date_column=None):
    """Analyze the detailed distribution pattern of the data"""
    # Convert grouped data back to series if needed
    if hasattr(data, 'obj'):  # If it's a GroupBy object
        data = data.obj

    if isinstance(data.iloc[0], (str, np.str_)):
        # For string data, analyze patterns including date relationship
        if date_column is not None and isinstance(data.index, pd.DatetimeIndex):
            # Calculate frequency distribution per date
            daily_counts = data.groupby(data.index.date).value_counts(normalize=True)
            avg_daily_dist = daily_counts.groupby(level=-1).mean()
            entries_per_day = data.groupby(data.index.date).size()
            
            return {
                'type': 'categorical',
                'distribution': avg_daily_dist.to_dict(),
                'unique_values': list(avg_daily_dist.index),
                'entries_per_day': {
                    'mean': entries_per_day.mean(),
                    'std': entries_per_day.std(),
                    'min': entries_per_day.min(),
                    'max': entries_per_day.max()
                }
            }
        else:
            # Standard frequency distribution if no date relationship
            value_counts = data.value_counts(normalize=True)
            return {
                'type': 'categorical',
                'distribution': value_counts.to_dict(),
                'unique_values': list(value_counts.index),
                'entries_per_day': None
            }
    else:
        # For numeric data, analyze the distribution pattern
        try:
            # Remove outliers for better distribution fitting
            q1 = data.quantile(0.01)
            q3 = data.quantile(0.99)
            iqr = q3 - q1
            cleaned_data = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
            
            # Fit KDE to get the probability density
            kde = gaussian_kde(cleaned_data)
            x_range = np.linspace(data.min(), data.max(), 1000)
            density = kde.evaluate(x_range)
            
            # Check if data is discrete
            is_discrete = all(data.astype(int) == data)
            
            # Calculate basic statistics
            stats_info = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skew': stats.skew(cleaned_data),
                'kurtosis': stats.kurtosis(cleaned_data),
                'is_discrete': is_discrete
            }
            
            return {
                'type': 'numeric',
                'kde': kde,
                'x_range': x_range,
                'density': density,
                'stats': stats_info,
                'original_data': data.values
            }
        except Exception as e:
            st.warning(f"Error in distribution analysis: {str(e)}")
            return None

def generate_realistic_values(dist_pattern, n_samples):
    """Generate values that match the original distribution pattern"""
    if dist_pattern['type'] == 'numeric':
        try:
            stats_info = dist_pattern['stats']
            
            # Handle the case where all values are the same
            if stats_info['std'] == 0:
                return np.full(n_samples, stats_info['mean'])
            
            # Generate base samples using KDE
            kde = dist_pattern['kde']
            samples = kde.resample(n_samples)[0]
            
            # Apply original constraints
            samples = np.clip(samples, stats_info['min'], stats_info['max'])
            
            # Handle NaN values
            samples = np.nan_to_num(samples, nan=stats_info['mean'])
            
            # If the original data was discrete, round the values
            if stats_info['is_discrete']:
                samples = np.round(samples).astype(int)
            
            # Adjust the distribution moments to match original
            samples = adjust_distribution_moments(samples, stats_info)
            
            return samples
        except Exception as e:
            st.warning(f"Error in value generation: {str(e)}")
            # Fallback to uniform distribution between min and max
            return np.random.uniform(
                stats_info['min'],
                stats_info['max'],
                n_samples
            )

def adjust_distribution_moments(samples, target_stats):
    """Adjust the generated samples to match the original distribution moments"""
    # Handle the case where all samples are the same
    if np.std(samples) == 0 or np.isnan(np.std(samples)):
        return np.full_like(samples, target_stats['mean'])
    
    # Adjust mean and standard deviation
    current_mean = np.mean(samples)
    current_std = np.std(samples)
    
    # Avoid division by zero
    if current_std > 0:
        # Standardize
        standardized = (samples - current_mean) / current_std
        # Adjust to target moments
        adjusted = standardized * target_stats['std'] + target_stats['mean']
    else:
        adjusted = np.full_like(samples, target_stats['mean'])
    
    # Clip to original bounds and handle NaN values
    adjusted = np.nan_to_num(adjusted, nan=target_stats['mean'])
    adjusted = np.clip(adjusted, target_stats['min'], target_stats['max'])
    
    return adjusted

def generate_simulated_data(existing_data, columns_config, start_date, end_date):
    """Generate simulated data that matches original distributions"""
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # First, analyze date patterns in original data
        if 'date' in existing_data.columns:
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            existing_data.set_index('date', inplace=True)
        
        # Initialize empty lists for rows
        all_rows = []
        
        # Generate data for each date
        for date in date_range:
            row_data = {'date': date}
            
            for column, config in columns_config.items():
                # Analyze original distribution
                original_data = existing_data[column].dropna()
                if isinstance(original_data, pd.Series):
                    original_data = original_data.copy()  # Make a copy to avoid view warnings
                dist_pattern = analyze_distribution_pattern(original_data, date_column='date')
                
                if dist_pattern is not None:
                    # Generate values matching the distribution
                    if dist_pattern['type'] == 'categorical' and dist_pattern.get('entries_per_day'):
                        # For categorical with multiple entries per day
                        entries_info = dist_pattern['entries_per_day']
                        
                        # FIX: Use Python's round() function and properly handle clipping for a float
                        random_value = np.random.normal(entries_info['mean'], entries_info['std'])
                        rounded_value = round(random_value)
                        n_entries = int(max(entries_info['min'], min(rounded_value, entries_info['max'])))
                        
                        # Fix: Normalize probabilities to ensure they sum to 1
                        probs = list(dist_pattern['distribution'].values())
                        probs_sum = sum(probs)
                        normalized_probs = [p/probs_sum for p in probs]
                        
                        values = np.random.choice(
                            dist_pattern['unique_values'],
                            size=n_entries,
                            p=normalized_probs
                        )
                        
                        # Create multiple rows for this date
                        for value in values:
                            all_rows.append({**row_data, column: value})
                            
                    else:
                        # Single value per day
                        if dist_pattern['type'] == 'numeric':
                            value = generate_realistic_values(dist_pattern, 1)[0]
                            
                            # Add noise if enabled
                            if config.get('noise_enabled', False):
                                noise_level = config.get('noise_level', 0.1)
                                noise = np.random.normal(0, dist_pattern['stats']['std'] * noise_level)
                                value = np.clip(
                                    value + noise,
                                    dist_pattern['stats']['min'],
                                    dist_pattern['stats']['max']
                                )
                                
                                if dist_pattern['stats']['is_discrete']:
                                    value = round(value)
                                
                                # Check if decimals are allowed
                                if not config.get('allow_decimals', True):
                                    value = round(value)
                        else:
                            # Fix: Normalize probabilities to ensure they sum to 1
                            probs = list(dist_pattern['distribution'].values())
                            probs_sum = sum(probs)
                            normalized_probs = [p/probs_sum for p in probs]
                            
                            value = np.random.choice(
                                dist_pattern['unique_values'],
                                p=normalized_probs
                            )
                        
                        row_data[column] = value
                        
            if not any(dist_pattern.get('entries_per_day') for dist_pattern in 
                      [analyze_distribution_pattern(existing_data[col].dropna(), 'date') 
                       for col in columns_config if col in existing_data.columns]):
                all_rows.append(row_data)
        
        # Convert to DataFrame
        simulated_data = pd.DataFrame(all_rows)
        
        # Convert date column to datetime and format it
        simulated_data['date'] = pd.to_datetime(simulated_data['date']).dt.strftime('%Y-%m-%d')
        
        # Sort by date
        simulated_data = simulated_data.sort_values('date').reset_index(drop=True)
        
        return simulated_data
        
    except Exception as e:
        st.error(f"Error generating simulated data: {str(e)}")
        return None

def plot_distribution_comparison(original_data, simulated_data, column):
    """Create detailed distribution comparison plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ensure we're working with series, not dataframes
    if isinstance(original_data, pd.DataFrame):
        original_series = original_data[column]
    else:
        original_series = original_data
        
    if isinstance(simulated_data, pd.DataFrame):
        simulated_series = simulated_data[column]
    else:
        simulated_series = simulated_data
    
    if isinstance(original_series.iloc[0], (str, np.str_)):
        # For categorical data, plot bar charts of frequencies
        orig_freq = original_series.value_counts(normalize=True)
        sim_freq = simulated_series.value_counts(normalize=True)
        
        orig_freq.plot(kind='bar', ax=ax1, title='Original Data Distribution')
        sim_freq.plot(kind='bar', ax=ax2, title='Simulated Data Distribution')
        
    else:
        # For numeric data, plot histograms with KDE
        sns.histplot(data=original_series, kde=True, ax=ax1)
        ax1.set_title('Original Data Distribution')
        
        sns.histplot(data=simulated_series, kde=True, ax=ax2)
        ax2.set_title('Simulated Data Distribution')
    
    plt.tight_layout()
    return fig

# Streamlit app
st.title("Advanced Data Simulator")
st.write("Upload your dataset to generate realistic simulated data based on statistical analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx", "csv", "txt"])

if uploaded_file:
    # Load the data
    existing_data = load_data(uploaded_file)
    
    if existing_data is not None:
        st.write("Uploaded Dataset Preview:")
        st.dataframe(existing_data.head())
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        st.write(existing_data.describe())
        
        # Column selection and configuration
        st.subheader("Configure Simulation Parameters")
        
        # Separate numeric and string columns
        numeric_columns = existing_data.select_dtypes(include=[np.number]).columns
        string_columns = existing_data.select_dtypes(include=['object']).columns
        
        # Column selection
        selected_numeric = st.multiselect(
            "Select numeric columns to simulate",
            numeric_columns,
            default=[numeric_columns[0]] if len(numeric_columns) > 0 else []
        )
        
        include_strings = st.checkbox("Include string columns in simulation", value=False)
        if include_strings:
            selected_strings = st.multiselect(
                "Select string columns to simulate",
                string_columns,
                default=[]
            )
        
        if selected_numeric or (include_strings and selected_strings):
            # Configuration for each column
            columns_config = {}
            
            # Configure numeric columns
            for col in selected_numeric:
                st.write(f"### Configuration for {col}")
                
                # Get current min/max values
                current_min = float(existing_data[col].min())
                current_max = float(existing_data[col].max())
                
                col1, col2 = st.columns(2)
                with col1:
                    distribution = st.selectbox(
                        f"Distribution type for {col}",
                        ['uniform', 'normal', 'random'],
                        key=f"dist_{col}"
                    )
                    
                    min_value = st.number_input(
                        f"Minimum value for {col}",
                        value=current_min,
                        key=f"min_{col}"
                    )
                
                with col2:
                    allow_decimals = st.checkbox(
                        f"Allow decimal values for {col}",
                        value=True,
                        key=f"dec_{col}"
                    )
                    
                    max_value = st.number_input(
                        f"Maximum value for {col}",
                        value=current_max,
                        key=f"max_{col}"
                    )
                
                # Noise configuration
                noise_enabled = st.checkbox(
                    f"Enable noise for {col}",
                    value=True,
                    key=f"noise_enable_{col}"
                )
                
                if noise_enabled:
                    noise_level = st.slider(
                        f"Noise level for {col}",
                        0.0, 1.0, 0.1,
                        key=f"noise_{col}"
                    )
                else:
                    noise_level = 0.0
                
                columns_config[col] = {
                    'type': 'numeric',
                    'distribution': distribution,
                    'min_value': min_value,
                    'max_value': max_value,
                    'allow_decimals': allow_decimals,
                    'noise_enabled': noise_enabled,
                    'noise_level': noise_level
                }
            
            # Configure string columns
            if include_strings:
                for col in selected_strings:
                    columns_config[col] = {
                        'type': 'string',
                        'include': True
                    }
            
            # Date range selection
            st.subheader("Select Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", datetime(2024, 1, 1))
            with col2:
                end_date = st.date_input("End date", datetime(2025, 12, 31))
            
            # Generate button
            if st.button("Generate Simulated Data"):
                with st.spinner("Analyzing data and generating simulation..."):
                    simulated_data = generate_simulated_data(
                        existing_data,
                        columns_config,
                        start_date,
                        end_date
                    )
                    
                    if simulated_data is not None:
                        st.subheader("Simulated Data Preview")
                        st.dataframe(simulated_data.head())
                        
                        # Plot comparisons with enhanced visualization
                        st.subheader("Data Distribution Comparisons")
                        for column in selected_numeric + (selected_strings if include_strings else []):
                            st.write(f"### Distribution Comparison for {column}")
                            fig = plot_distribution_comparison(existing_data, simulated_data, column)
                            st.pyplot(fig)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = simulated_data.to_csv(index=False)
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name="simulated_data.csv",
                                mime="text/csv"
                            )
                        with col2:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                simulated_data.to_excel(writer, index=False)
                            st.download_button(
                                label="Download as Excel",
                                data=output.getvalue(),
                                file_name="simulated_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
        else:
            st.warning("Please select at least one column to simulate")