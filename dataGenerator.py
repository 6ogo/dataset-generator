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

# Get today's date
today = datetime.today().date()

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

def analyze_distribution_pattern(data, column_name=None):
    """Analyze the distribution pattern of the data, preserving skewness and tails."""
    # Handle string/categorical data better
    if isinstance(data.iloc[0], (str, np.str_)):
        value_counts = data.value_counts(normalize=True)
        return {
            'type': 'categorical',
            'distribution': value_counts.to_dict(),
            'unique_values': list(value_counts.index)
        }
    else:
        # For numeric data - preserve exact distribution better
        try:
            data_clean = data.dropna().values
            
            # Use a more flexible bandwidth for KDE to better capture skewed distributions
            if data_clean.std() > 0:
                # Use Silverman's rule for bandwidth selection
                bw = 0.9 * min(data_clean.std(), (np.percentile(data_clean, 75) - 
                               np.percentile(data_clean, 25))/1.34) * len(data_clean)**(-0.2)
                kde = gaussian_kde(data_clean, bw_method=bw)
            else:
                kde = gaussian_kde(data_clean)
                
            x_range = np.linspace(data_clean.min(), data_clean.max(), 1000)
            density = kde.evaluate(x_range)
            is_discrete = all(data_clean.astype(int) == data_clean)
            
            # Calculate percentiles to better capture the distribution
            percentiles = np.percentile(data_clean, [0, 10, 25, 50, 75, 90, 100])
            
            stats_info = {
                'mean': data_clean.mean(),
                'median': np.median(data_clean),
                'std': data_clean.std(),
                'min': data_clean.min(),
                'max': data_clean.max(),
                'skew': stats.skew(data_clean),
                'kurtosis': stats.kurtosis(data_clean),
                'is_discrete': is_discrete,
                'percentiles': percentiles
            }
            
            # Remember histogram data for better simulation
            hist, bin_edges = np.histogram(data_clean, bins='auto')
            
            return {
                'type': 'numeric',
                'kde': kde,
                'x_range': x_range,
                'density': density,
                'stats': stats_info,
                'original_data': data_clean,
                'histogram': {
                    'counts': hist,
                    'bins': bin_edges
                }
            }
        except Exception as e:
            st.warning(f"Error in distribution analysis: {str(e)}")
            return None

def generate_realistic_values(dist_pattern, n_samples):
    """Generate values that match the original distribution pattern"""
    if dist_pattern is None:
        return np.zeros(n_samples)
        
    if dist_pattern['type'] == 'categorical':
        # Improved categorical handling
        values = list(dist_pattern['distribution'].keys())
        probs = list(dist_pattern['distribution'].values())
        probs_sum = sum(probs)
        normalized_probs = [p / probs_sum for p in probs]
        return np.random.choice(values, size=n_samples, p=normalized_probs)
    elif dist_pattern['type'] == 'numeric':
        try:
            stats_info = dist_pattern['stats']
            
            # Handle special cases
            if stats_info['std'] == 0:
                return np.full(n_samples, stats_info['mean'])
            
            # For highly skewed distributions (use histogram sampling)
            if abs(stats_info['skew']) > 1.0:
                # Use histogram-based sampling for better preservation of skewed distributions
                hist = dist_pattern['histogram']
                bin_counts = hist['counts']
                bin_edges = hist['bins']
                bin_probs = bin_counts / bin_counts.sum()
                
                # Generate bin indices according to their probability
                bin_indices = np.random.choice(len(bin_probs), size=n_samples, p=bin_probs)
                
                # Generate uniform samples within each selected bin
                samples = np.array([np.random.uniform(bin_edges[i], bin_edges[i+1]) for i in bin_indices])
            else:
                # Use KDE for more normally distributed data
                samples = dist_pattern['kde'].resample(n_samples)[0]
            
            # Apply constraints
            samples = np.clip(samples, stats_info['min'], stats_info['max'])
            
            # Handle NaN values
            samples = np.nan_to_num(samples, nan=stats_info['mean'])
            
            # If the original data was discrete, round the values
            if stats_info['is_discrete']:
                samples = np.round(samples).astype(int)
            
            return samples
        except Exception as e:
            # Fallback to simple sampling from the original data
            if len(dist_pattern['original_data']) > 0:
                return np.random.choice(dist_pattern['original_data'], size=n_samples)
            else:
                return np.zeros(n_samples)
    else:
        return np.zeros(n_samples)

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
    """Generate simulated data with improved preservation of relationships"""
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Detect the column structure of the original data
        all_columns = existing_data.columns.tolist()
        
        # Try to detect region column, date column, and numeric columns
        region_column = None
        date_column = None
        numeric_columns = []
        
        # Check for date column
        for col in all_columns:
            if existing_data[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                try:
                    pd.to_datetime(existing_data[col])
                    date_column = col
                    break
                except:
                    pass
        
        # If we didn't find a date column but need one, assume the first column
        if date_column is None and len(all_columns) > 0:
            try:
                pd.to_datetime(existing_data[all_columns[0]])
                date_column = all_columns[0]
            except:
                pass
        
        # Try to identify region column (string column with location/region names)
        string_columns = existing_data.select_dtypes(include=['object']).columns
        for col in string_columns:
            if any(word in col.lower() for word in ['region', 'län', 'county', 'state', 'location']):
                region_column = col
                break
        
        # If no obvious region column, use the first string column that's not the date
        if region_column is None and len(string_columns) > 0:
            for col in string_columns:
                if col != date_column:
                    region_column = col
                    break
        
        # Identify numeric columns
        for col in all_columns:
            if col not in [date_column, region_column] and pd.api.types.is_numeric_dtype(existing_data[col]):
                numeric_columns.append(col)
        
        # Now we analyze distributions for each column and the relationships between them
        dist_patterns = {}
        
        # Calculate rows per day for our simulation
        if date_column is not None:
            existing_data[date_column] = pd.to_datetime(existing_data[date_column])
            rows_per_day = existing_data.groupby(date_column).size()
            rows_per_day_stats = {
                'mean': rows_per_day.mean(),
                'std': rows_per_day.std(),
                'min': max(1, rows_per_day.min()),
                'max': rows_per_day.max()
            }
        else:
            # Default to 1 row per day if no date column
            rows_per_day_stats = {'mean': 1, 'std': 0, 'min': 1, 'max': 1}
        
        # Analyze distributions for each column
        for column in all_columns:
            if column in columns_config:
                original_data = existing_data[column].dropna()
                dist_patterns[column] = analyze_distribution_pattern(original_data, column)
        
        # If we have region and numeric columns, analyze their relationship
        region_value_patterns = {}
        if region_column is not None and numeric_columns:
            for region in existing_data[region_column].unique():
                region_data = existing_data[existing_data[region_column] == region]
                region_value_patterns[region] = {}
                
                for num_col in numeric_columns:
                    if num_col in columns_config:
                        column_data = region_data[num_col].dropna()
                        if not column_data.empty:
                            region_value_patterns[region][num_col] = analyze_distribution_pattern(column_data)
        
        # Generate data with progress feedback
        all_rows = []
        total_dates = len(date_range)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date in enumerate(date_range):
            # Sample number of rows for this date
            if rows_per_day_stats['std'] > 0:
                n_rows = int(np.random.normal(rows_per_day_stats['mean'], rows_per_day_stats['std']))
                n_rows = max(rows_per_day_stats['min'], min(n_rows, rows_per_day_stats['max']))
            else:
                n_rows = int(rows_per_day_stats['mean'])
            
            # Generate rows
            for _ in range(n_rows):
                row_data = {}
                
                # Add date
                if date_column is not None:
                    row_data[date_column] = date
                
                # If we have a region column, first select a region
                if region_column is not None and region_column in dist_patterns:
                    region = generate_realistic_values(dist_patterns[region_column], 1)[0]
                    row_data[region_column] = region
                    
                    # Now generate values based on the region's specific distribution
                    if region in region_value_patterns:
                        for column in numeric_columns:
                            if column in columns_config:
                                if column in region_value_patterns[region]:
                                    region_pattern = region_value_patterns[region][column]
                                    value = generate_realistic_values(region_pattern, 1)[0]
                                else:
                                    # Fallback to overall distribution if no region-specific pattern
                                    pattern = dist_patterns.get(column)
                                    value = generate_realistic_values(pattern, 1)[0]
                                
                                # Apply noise if configured
                                config = columns_config.get(column, {})
                                if config.get('noise_enabled', False) and 'stats' in dist_patterns.get(column, {}):
                                    noise_level = config.get('noise_level', 0.1)
                                    pattern = dist_patterns.get(column)
                                    noise = np.random.normal(0, pattern['stats']['std'] * noise_level)
                                    value = np.clip(value + noise, pattern['stats']['min'], pattern['stats']['max'])
                                
                                # Round if needed
                                if not config.get('allow_decimals', True):
                                    value = round(value)
                                
                                row_data[column] = value
                else:
                    # Generate values for each configured column
                    for column, config in columns_config.items():
                        if column in dist_patterns:
                            pattern = dist_patterns[column]
                            value = generate_realistic_values(pattern, 1)[0]
                            
                            # Apply noise if configured for numeric columns
                            if config.get('type') == 'numeric' and config.get('noise_enabled', False):
                                noise_level = config.get('noise_level', 0.1)
                                if 'stats' in pattern:
                                    noise = np.random.normal(0, pattern['stats']['std'] * noise_level)
                                    value = np.clip(value + noise, pattern['stats']['min'], pattern['stats']['max'])
                            
                            # Round if needed
                            if config.get('type') == 'numeric' and not config.get('allow_decimals', True):
                                value = round(value)
                            
                            row_data[column] = value
                
                all_rows.append(row_data)
            
            # Update progress
            progress = (i + 1) / total_dates
            progress_bar.progress(progress)
            status_text.text(f"Generating data: {i + 1}/{total_dates} dates completed")
        
        # Convert to DataFrame and ensure proper formatting
        simulated_data = pd.DataFrame(all_rows)
        
        # Format dates properly
        if date_column in simulated_data.columns:
            simulated_data[date_column] = pd.to_datetime(simulated_data[date_column])
            # Match format of original data
            if date_column in existing_data.columns:
                orig_date_sample = str(existing_data[date_column].iloc[0])
                if ' 00:00:00' in orig_date_sample:
                    simulated_data[date_column] = simulated_data[date_column].dt.strftime('%Y-%m-%d 00:00:00')
                else:
                    simulated_data[date_column] = simulated_data[date_column].dt.strftime('%Y-%m-%d')
        
        # Ensure column order matches original data
        if set(all_columns).issubset(set(simulated_data.columns)):
            simulated_data = simulated_data[all_columns]
        
        # Sort by date if available
        if date_column in simulated_data.columns:
            simulated_data = simulated_data.sort_values(date_column).reset_index(drop=True)
        
        return simulated_data
    
    except Exception as e:
        st.error(f"Error generating simulated data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
                end_date = st.date_input("End date", today)
            
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
                            # CSV with utf-8-sig encoding
                            csv = simulated_data.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name="simulated_data.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Excel with xlsxwriter
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