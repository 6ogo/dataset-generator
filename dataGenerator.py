import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import chardet
import csv
import io
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# --- File Loading & Basic Helpers (Keep your existing robust versions) ---

def detect_encoding(file_content):
    """Detect the encoding of a file"""
    result = chardet.detect(file_content)
    # Provide a fallback encoding if detection confidence is low or result is None
    return result['encoding'] if result and result['encoding'] and result['confidence'] > 0.5 else 'utf-8'

def detect_delimiter(file_content, encoding):
    """Detect the delimiter of a text file"""
    content_str = file_content.decode(encoding)
    # Prioritize common delimiters
    delimiters = [',', ';', '\t', '|']
    # Use Sniffer first
    try:
        # Decode only a portion for sniffing
        sample = content_str[:2048] # Sniff a larger sample
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(delimiters))
        return dialect.delimiter
    except csv.Error:
        # Sniffer failed, fallback to counting occurrences in the sample
        counts = {d: sample.count(d) for d in delimiters}
        # Basic check: delimiter should appear relatively frequently per line
        lines = sample.splitlines()
        avg_counts = {d: counts[d] / len(lines) if len(lines) > 0 else 0 for d in delimiters}

        # Filter out delimiters that don't appear much on average
        plausible_delimiters = {d: c for d, c in avg_counts.items() if c > 0.5} # Heuristic: >0.5 times per line avg

        if plausible_delimiters:
             # Choose the most frequent among the plausible ones
             max_delimiter = max(plausible_delimiters, key=plausible_delimiters.get)
             return max_delimiter
        elif counts and max(counts.values()) > 0:
             # If no plausible delimiter, return the most frequent overall, hoping for the best
             return max(counts, key=counts.get)
        else:
            # Default fallback
            return ','


def load_data(uploaded_file):
    """Load data with automatic format detection"""
    if uploaded_file is None:
        return None
    try:
        file_content = uploaded_file.read()
        uploaded_file.seek(0) # Reset buffer pointer after reading

        if not file_content:
             st.error("Uploaded file is empty.")
             return None

        encoding = detect_encoding(file_content[:10000]) # Detect encoding on a larger sample

        if uploaded_file.name.endswith('.xlsx'):
            try:
                # Use io.BytesIO to read from memory content
                return pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                st.error(traceback.format_exc())
                return None

        elif uploaded_file.name.endswith(('.csv', '.txt')):
            delimiter = detect_delimiter(file_content, encoding)
            st.info(f"Detected encoding: {encoding}, delimiter: '{delimiter}'")

            try:
                # Use io.BytesIO for pandas
                df = pd.read_csv(
                    io.BytesIO(file_content),
                    encoding=encoding,
                    sep=delimiter,
                    engine='python', # Python engine often more flexible
                    skipinitialspace=True # Handle spaces after delimiters
                )
                st.success(f"File loaded successfully.")
                # Basic data cleaning suggestion
                df = df.rename(columns=lambda x: x.strip()) # Strip whitespace from headers
                return df
            except Exception as e1:
                st.warning(f"Initial CSV loading failed ({e1}). Trying with error skipping...")
                try:
                     # Reset buffer again before re-reading
                    uploaded_file.seek(0)
                    file_content_retry = uploaded_file.read()
                    df = pd.read_csv(
                        io.BytesIO(file_content_retry),
                        encoding=encoding,
                        sep=delimiter,
                        engine='python',
                        on_bad_lines='skip', # Skip problematic lines
                        skipinitialspace=True
                    )
                    st.info("File loaded, but some rows may have been skipped due to parsing issues.")
                    df = df.rename(columns=lambda x: x.strip())
                    return df
                except Exception as e2:
                    st.error(f"Failed to load CSV/TXT file even with skipping: {str(e2)}")
                    st.error(traceback.format_exc())
                    return None
        else:
             st.error("Unsupported file type. Please upload .xlsx, .csv, or .txt.")
             return None

    except Exception as e:
        st.error(f"Error processing file upload: {str(e)}")
        st.error(traceback.format_exc())
        return None

# --- V3 Analysis Function ---
def analyze_temporal_patterns_v3(data, date_column, region_column, columns_to_analyze):
    """
    Analyzes temporal AND regional patterns.

    Stores stats grouped by:
    - region -> time_period (monthly/daily/overall) -> stats/values/distribution
    - overall_temporal -> time_period (monthly/daily/overall) -> stats/values/distribution (Fallback)
    - overall_all -> stats/values/distribution (Final Fallback)
    - region_distribution -> time_period -> distribution (How regions themselves are distributed over time)
    """
    st.write(f"Analyzing with Date='{date_column}', Region='{region_column}', Columns={columns_to_analyze}") # Debug
    if date_column not in data.columns or data[date_column].isnull().all():
        st.error(f"Date column '{date_column}' not found or is empty.")
        return None, None
    if region_column not in data.columns or data[region_column].isnull().all():
        st.error(f"Region column '{region_column}' not found or is empty.")
        return None, None

    try:
        df = data.copy()
        # Convert date column safely
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column, region_column]) # Need valid date and region
        if df.empty:
             st.warning("No valid rows found after handling dates and regions.")
             return None, None

        df['month'] = df[date_column].dt.month
        df['dayofweek'] = df[date_column].dt.dayofweek # Monday=0, Sunday=6

        regions = df[region_column].unique()
        temporal_stats = {} # Main dictionary for column stats
        region_distribution_stats = {'monthly': {}, 'daily': {}, 'overall': {}} # For region distribution

        numeric_cols = df[columns_to_analyze].select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df[columns_to_analyze].select_dtypes(exclude=np.number).columns.tolist()
        if region_column in categorical_cols:
             categorical_cols.remove(region_column) # Don't analyze region col like others

        # --- 1. Analyze each column (Numeric and Other Categorical) ---
        for col in columns_to_analyze:
             if col == region_column: continue # Skip region column here

             temporal_stats[col] = {
                 'regions': {region: {'monthly': {}, 'daily': {}, 'overall': {}} for region in regions},
                 'overall_temporal': {'monthly': {}, 'daily': {}, 'overall': {}},
                 'overall_all': {}
             }
             is_numeric = col in numeric_cols
             valid_data_col = df[[col, region_column, 'month', 'dayofweek']].dropna(subset=[col])

             if valid_data_col.empty:
                  st.warning(f"No valid data found for column '{col}'. Skipping analysis for it.")
                  # Add placeholder empty stats
                  placeholder_numeric = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'values': np.array([])}
                  placeholder_cat = {'distribution': {}, 'count': 0}
                  temporal_stats[col]['overall_all'] = placeholder_numeric if is_numeric else placeholder_cat
                  for region in regions:
                       temporal_stats[col]['regions'][region]['overall'] = placeholder_numeric if is_numeric else placeholder_cat
                  continue

             # --- a) Overall stats for the column (final fallback) ---
             overall_values_all = valid_data_col[col].values
             if is_numeric:
                 temporal_stats[col]['overall_all'] = {
                     'mean': np.mean(overall_values_all), 'std': np.std(overall_values_all),
                     'min': np.min(overall_values_all), 'max': np.max(overall_values_all),
                     'count': len(overall_values_all), 'values': overall_values_all
                 }
             else:
                 temporal_stats[col]['overall_all'] = {
                      'distribution': pd.Series(overall_values_all).value_counts(normalize=True).to_dict(),
                      'count': len(overall_values_all)
                 }

             # --- b) Overall TEMPORAL stats (ignoring region - first fallback) ---
             for month, group_month in valid_data_col.groupby('month'):
                 values_month = group_month[col].values
                 if len(values_month) > 0:
                     if is_numeric:
                          temporal_stats[col]['overall_temporal']['monthly'][month] = {
                              'mean': np.mean(values_month), 'std': np.std(values_month),
                              'min': np.min(values_month), 'max': np.max(values_month),
                              'count': len(values_month), 'values': values_month
                          }
                     else:
                          temporal_stats[col]['overall_temporal']['monthly'][month] = {
                              'distribution': pd.Series(values_month).value_counts(normalize=True).to_dict(),
                              'count': len(values_month)
                          }
             for dow, group_dow in valid_data_col.groupby('dayofweek'):
                  values_dow = group_dow[col].values
                  if len(values_dow) > 0:
                      if is_numeric:
                           temporal_stats[col]['overall_temporal']['daily'][dow] = {
                               'mean': np.mean(values_dow), 'std': np.std(values_dow),
                               'min': np.min(values_dow), 'max': np.max(values_dow),
                               'count': len(values_dow), 'values': values_dow
                           }
                      else:
                           temporal_stats[col]['overall_temporal']['daily'][dow] = {
                               'distribution': pd.Series(values_dow).value_counts(normalize=True).to_dict(),
                               'count': len(values_dow)
                           }


             # --- c) REGION-SPECIFIC stats (primary source) ---
             for region, group_region in valid_data_col.groupby(region_column):
                  if group_region.empty: continue

                  # Overall for this region
                  overall_values_region = group_region[col].values
                  if len(overall_values_region) > 0:
                      if is_numeric:
                           temporal_stats[col]['regions'][region]['overall'] = {
                               'mean': np.mean(overall_values_region), 'std': np.std(overall_values_region),
                               'min': np.min(overall_values_region), 'max': np.max(overall_values_region),
                               'count': len(overall_values_region), 'values': overall_values_region
                           }
                      else:
                           temporal_stats[col]['regions'][region]['overall'] = {
                               'distribution': pd.Series(overall_values_region).value_counts(normalize=True).to_dict(),
                                'count': len(overall_values_region)
                           }

                  # Monthly for this region
                  for month, group_region_month in group_region.groupby('month'):
                      values_region_month = group_region_month[col].values
                      if len(values_region_month) > 0:
                          if is_numeric:
                               temporal_stats[col]['regions'][region]['monthly'][month] = {
                                   'mean': np.mean(values_region_month), 'std': np.std(values_region_month),
                                   'min': np.min(values_region_month), 'max': np.max(values_region_month),
                                   'count': len(values_region_month), 'values': values_region_month
                               }
                          else:
                               temporal_stats[col]['regions'][region]['monthly'][month] = {
                                   'distribution': pd.Series(values_region_month).value_counts(normalize=True).to_dict(),
                                   'count': len(values_region_month)
                               }

                  # Daily for this region
                  for dow, group_region_dow in group_region.groupby('dayofweek'):
                      values_region_dow = group_region_dow[col].values
                      if len(values_region_dow) > 0:
                           if is_numeric:
                                temporal_stats[col]['regions'][region]['daily'][dow] = {
                                    'mean': np.mean(values_region_dow), 'std': np.std(values_region_dow),
                                    'min': np.min(values_region_dow), 'max': np.max(values_region_dow),
                                    'count': len(values_region_dow), 'values': values_region_dow
                                }
                           else:
                                temporal_stats[col]['regions'][region]['daily'][dow] = {
                                    'distribution': pd.Series(values_region_dow).value_counts(normalize=True).to_dict(),
                                    'count': len(values_region_dow)
                                }

        # --- 2. Analyze the distribution of the REGION column itself over time ---
        region_distribution_stats['overall'] = df[region_column].value_counts(normalize=True).to_dict()
        monthly_region_dist = df.groupby('month')[region_column].value_counts(normalize=True).unstack(fill_value=0)
        region_distribution_stats['monthly'] = monthly_region_dist.to_dict('index')
        daily_region_dist = df.groupby('dayofweek')[region_column].value_counts(normalize=True).unstack(fill_value=0)
        region_distribution_stats['daily'] = daily_region_dist.to_dict('index')

        # --- 3. Analyze Rows Per Day (keep similar logic, maybe add region breakdown if needed later) ---
        rows_per_day_analysis = {'monthly': {}, 'daily': {}, 'overall': {}}
        if not df.empty:
            df['date_only'] = df[date_column].dt.date
            daily_counts = df.groupby('date_only').size()
            if not daily_counts.empty:
                rows_per_day_analysis['overall'] = {
                    'mean': daily_counts.mean(), 'std': daily_counts.std(),
                    'min': daily_counts.min(), 'max': daily_counts.max(), 'count': len(daily_counts)
                }

                # Add temporal breakdown for rows per day
                rows_per_day_temporal = df.groupby(['date_only', 'month', 'dayofweek']).size().reset_index(name='counts')
                monthly_rows = rows_per_day_temporal.groupby('month')['counts'].agg(['mean', 'std', 'min', 'max', 'count'])
                rows_per_day_analysis['monthly'] = monthly_rows.fillna(0).to_dict('index')
                daily_rows = rows_per_day_temporal.groupby('dayofweek')['counts'].agg(['mean', 'std', 'min', 'max', 'count'])
                rows_per_day_analysis['daily'] = daily_rows.fillna(0).to_dict('index')
            else: # Handle case where daily_counts is empty
                 rows_per_day_analysis['overall'] = {'mean': 1, 'std': 0, 'min': 1, 'max': 1, 'count':0}
        else: # Handle case where df is empty initially
            rows_per_day_analysis['overall'] = {'mean': 1, 'std': 0, 'min': 1, 'max': 1, 'count':0}


        st.success("Analysis Complete.") # Debug
        return temporal_stats, rows_per_day_analysis, region_distribution_stats

    except Exception as e:
        st.error(f"Error during temporal/regional analysis: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# --- Dummy plot function if not defined elsewhere ---
def plot_distribution_comparison(original_data, simulated_data, column):
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=False) # Don't share Y axis for potentially different scales
     plt.style.use('ggplot') # Use a consistent style

     # --- Plot Original ---
     try:
         if column in original_data and not original_data[column].isnull().all():
             data_orig = original_data[column].dropna()
             if pd.api.types.is_numeric_dtype(data_orig) and data_orig.nunique() > 1: # Plot histogram for numeric
                 sns.histplot(data_orig, kde=True, ax=ax1, bins=50) # Increased bins
                 ax1.set_title(f'Original: {column}')
             elif not data_orig.empty: # Plot bar chart for categorical or single-value numeric
                  # Limit number of categories shown for clarity
                  top_n = 30
                  value_counts_orig = data_orig.value_counts()
                  if len(value_counts_orig) > top_n:
                      top_counts = value_counts_orig.nlargest(top_n)
                      other_count = value_counts_orig.nsmallest(len(value_counts_orig) - top_n).sum()
                      if other_count > 0: top_counts['Other'] = other_count
                      top_counts.plot(kind='bar', ax=ax1)
                      ax1.set_title(f'Original: {column} (Top {top_n})')
                  else:
                      value_counts_orig.plot(kind='bar', ax=ax1)
                      ax1.set_title(f'Original: {column}')
                  plt.setp(ax1.get_xticklabels(), rotation=90, ha='right') # Rotate labels
             else: ax1.set_title(f'Original: {column} (Empty)')
         else: ax1.set_title(f'Original: {column} (Not Found or All Null)')
     except Exception as e:
          ax1.set_title(f'Original: {column} (Plot Error)')
          print(f"Plot error (original) for {column}: {e}") # Debug print

     # --- Plot Simulated ---
     try:
         if column in simulated_data and not simulated_data[column].isnull().all():
             data_sim = simulated_data[column].dropna()
             if pd.api.types.is_numeric_dtype(data_sim) and data_sim.nunique() > 1:
                 sns.histplot(data_sim, kde=True, ax=ax2, bins=50)
                 ax2.set_title(f'Simulated: {column}')
             elif not data_sim.empty:
                  top_n = 30
                  value_counts_sim = data_sim.value_counts()
                  if len(value_counts_sim) > top_n:
                      top_counts_sim = value_counts_sim.nlargest(top_n)
                      other_count_sim = value_counts_sim.nsmallest(len(value_counts_sim) - top_n).sum()
                      if other_count_sim > 0: top_counts_sim['Other'] = other_count_sim
                      top_counts_sim.plot(kind='bar', ax=ax2)
                      ax2.set_title(f'Simulated: {column} (Top {top_n})')
                  else:
                      value_counts_sim.plot(kind='bar', ax=ax2)
                      ax2.set_title(f'Simulated: {column}')
                  plt.setp(ax2.get_xticklabels(), rotation=90, ha='right')
             else: ax2.set_title(f'Simulated: {column} (Empty)')
         else: ax2.set_title(f'Simulated: {column} (Not Found or All Null)')
     except Exception as e:
          ax2.set_title(f'Simulated: {column} (Plot Error)')
          print(f"Plot error (simulated) for {column}: {e}") # Debug print


     plt.tight_layout(pad=2.0) # Add padding
     return fig
 
# --- V3 Generation Function ---
def generate_simulated_data_v3(existing_data, # Required even if not appending, for analysis
                               columns_config,
                               start_date_input, # The user's requested start date
                               end_date, # The user's requested end date
                               date_column, # <<< FIXED: Added date_column parameter
                               region_column, # Name of the region column
                               append_data=False, # Whether to append or replace
                               region_sampling_mode='distribute_daily', # 'distribute_daily' or 'one_per_day'
                               temporal_weight=0.7, # Weight for HYBRID categorical sampling
                               min_sample_size_direct=10): # Threshold for direct numeric sampling
    """
    Generate simulated data incorporating temporal and REGIONAL patterns.
    Can append to existing data or generate fresh data.
    Offers different region sampling modes. V3 - corrected signature.
    """
    try:
        if existing_data is None:
             st.error("Existing data is required for analysis.")
             return None
        if not date_column or date_column not in existing_data.columns:
             st.error(f"Invalid or missing Date column provided: '{date_column}'")
             return None
        if not region_column or region_column not in existing_data.columns:
             st.error(f"Invalid or missing Region column provided: '{region_column}'")
             return None


        # --- 1. Determine Actual Simulation Start Date ---
        final_start_date = start_date_input
        if append_data:
            st.info("Append mode enabled. Determining last date from existing data...")
            try:
                # Ensure date column is datetime for finding max
                existing_data[date_column] = pd.to_datetime(existing_data[date_column], errors='coerce')
                valid_dates = existing_data[date_column].dropna()
                if valid_dates.empty:
                    st.warning("No valid dates found in existing data to determine last date. Starting from user input start date.")
                    final_start_date = start_date_input
                    append_data = False # Cannot append if no valid start date
                else:
                    last_date = valid_dates.max().date()
                    final_start_date = last_date + timedelta(days=1)
                    st.success(f"Last date in existing data: {last_date}. Simulation will start from: {final_start_date}")
            except Exception as e:
                st.error(f"Error processing dates in existing data for appending: {e}. Using user input start date.")
                final_start_date = start_date_input
                append_data = False # Turn off append if error

        # Ensure final_start_date is a date object
        if isinstance(final_start_date, datetime): final_start_date = final_start_date.date()
        if isinstance(end_date, datetime): end_date = end_date.date()


        if final_start_date > end_date:
            st.warning(f"Calculated start date ({final_start_date}) is after the end date ({end_date}). No new data will be generated.")
            if append_data: return existing_data
            else: return pd.DataFrame(columns=existing_data.columns)


        # --- 2. Analyze Patterns (using the original data) ---
        st.info("Step 2: Analyzing temporal and regional patterns...")
        columns_to_analyze = list(columns_config.keys())
        try:
             temporal_stats, rows_per_day_analysis, region_distribution_stats = analyze_temporal_patterns_v3(
                 existing_data, # Analyze the original data
                 date_column,
                 region_column,
                 columns_to_analyze
             )
        except NameError:
             st.error("Error: The 'analyze_temporal_patterns_v3' function is not defined or accessible.")
             return None
        except Exception as analysis_err:
              st.error(f"An error occurred during analysis: {analysis_err}")
              st.error(traceback.format_exc())
              return None

        if temporal_stats is None or rows_per_day_analysis is None or region_distribution_stats is None:
            st.error("Analysis failed or returned no results. Cannot proceed.")
            return None

        # --- 3. Generate NEW Data ---
        st.info(f"Step 3: Generating simulated data from {final_start_date} to {end_date}...")
        simulated_rows = []
        try: date_range = pd.date_range(start=final_start_date, end=end_date, freq='D')
        except ValueError as date_err:
             st.error(f"Invalid date range for simulation: {date_err}.")
             return None

        total_dates = len(date_range)
        if total_dates == 0:
            st.info("Date range resulted in zero days for new data generation.")
            if append_data: return existing_data
            else: return pd.DataFrame(columns=existing_data.columns)

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Preparing simulation...")

        all_regions_list = list(region_distribution_stats.get('overall', {}).keys())
        if not all_regions_list:
             st.error(f"No regions found during analysis of column '{region_column}'. Cannot simulate.")
             return None


        # --- Simulation Loop ---
        for i, current_date in enumerate(date_range):
            month = current_date.month
            dayofweek = current_date.dayofweek
            current_date_dt = pd.to_datetime(current_date) # Use pandas datetime for consistency

            # --- a) Determine Number of Rows for Today ---
            n_rows = 1 # Default - use your previous working logic here
            # --- Example n_rows logic (copied from previous) ---
            daily_row_stats = rows_per_day_analysis.get('daily', {}).get(dayofweek)
            monthly_row_stats = rows_per_day_analysis.get('monthly', {}).get(month)
            overall_row_stats = rows_per_day_analysis.get('overall')
            use_row_stats = None
            if daily_row_stats and daily_row_stats.get('mean', 0) > 0: use_row_stats = daily_row_stats
            elif monthly_row_stats and monthly_row_stats.get('mean', 0) > 0: use_row_stats = monthly_row_stats
            elif overall_row_stats and overall_row_stats.get('mean', 0) > 0: use_row_stats = overall_row_stats
            if use_row_stats:
                try:
                    row_mean, row_std = float(use_row_stats.get('mean', 1)), float(use_row_stats.get('std', 0))
                    row_min, row_max = float(use_row_stats.get('min', 1)), float(use_row_stats.get('max', 1))
                    row_std = max(0, row_std)
                    n_rows = max(0, int(np.random.normal(row_mean, row_std * 0.5)))
                    final_row_min = int(max(0, row_min))
                    final_row_max = int(max(final_row_min, row_max * 1.5)) # Ensure max >= min
                    n_rows = max(final_row_min, min(n_rows, final_row_max))
                except: n_rows = 1 # Fallback on error
            else: n_rows = 1


            # --- b) Determine Region(s) for Today ---
            selected_region_for_the_day = None
            # CORRECTNESS CHECK: 'one_per_day' Logic
            if region_sampling_mode == 'one_per_day':
                # Sample ONE region for the whole day using hybrid probabilities
                # This block runs ONCE per day (outside the inner row loop)
                overall_region_dist = region_distribution_stats.get('overall', {})
                temporal_region_dist = {}
                if dayofweek in region_distribution_stats.get('daily', {}): temporal_region_dist = region_distribution_stats['daily'][dayofweek]
                elif month in region_distribution_stats.get('monthly', {}): temporal_region_dist = region_distribution_stats['monthly'][month]

                combined_region_dist = {}
                safe_overall_r_dist = overall_region_dist if isinstance(overall_region_dist, dict) else {}
                safe_temporal_r_dist = temporal_region_dist if isinstance(temporal_region_dist, dict) else {}
                all_possible_regions = set(safe_overall_r_dist.keys()) | set(safe_temporal_r_dist.keys())

                if not all_possible_regions: selected_region_for_the_day = random.choice(all_regions_list) if all_regions_list else "UNKNOWN_REGION"
                else:
                    for r in all_possible_regions:
                        prob_overall = safe_overall_r_dist.get(r, 0)
                        prob_temporal = safe_temporal_r_dist.get(r, 0)
                        combined_region_dist[r] = (temporal_weight * prob_temporal + (1 - temporal_weight) * prob_overall)

                    r_cats = list(combined_region_dist.keys())
                    r_probs = list(combined_region_dist.values())
                    r_total_prob = sum(r_probs)
                    if r_total_prob > 1e-9 and r_cats:
                        r_probs = [p / r_total_prob for p in r_probs]
                        selected_region_for_the_day = np.random.choice(r_cats, p=r_probs)
                    elif r_cats: selected_region_for_the_day = random.choice(r_cats)
                    else: selected_region_for_the_day = "UNKNOWN_REGION"


            # --- c) Generate Rows for Today ---
            for _ in range(n_rows): # Inner loop for rows within the day
                row_data = {}
                row_data[date_column] = current_date_dt # Store date

                # --- i) Determine Region for this ROW ---
                current_row_region = None
                 # CORRECTNESS CHECK: 'one_per_day' Assignment
                if region_sampling_mode == 'one_per_day':
                    # Assign the single region sampled earlier for the day
                    current_row_region = selected_region_for_the_day
                elif region_sampling_mode == 'distribute_daily':
                    # Sample region for EACH row (logic remains the same as before)
                    overall_region_dist = region_distribution_stats.get('overall', {})
                    temporal_region_dist = {}
                    if dayofweek in region_distribution_stats.get('daily', {}): temporal_region_dist = region_distribution_stats['daily'][dayofweek]
                    elif month in region_distribution_stats.get('monthly', {}): temporal_region_dist = region_distribution_stats['monthly'][month]

                    combined_region_dist = {}
                    safe_overall_r_dist = overall_region_dist if isinstance(overall_region_dist, dict) else {}
                    safe_temporal_r_dist = temporal_region_dist if isinstance(temporal_region_dist, dict) else {}
                    all_possible_regions = set(safe_overall_r_dist.keys()) | set(safe_temporal_r_dist.keys())

                    if not all_possible_regions: current_row_region = random.choice(all_regions_list) if all_regions_list else "UNKNOWN_REGION"
                    else:
                        for r in all_possible_regions:
                            prob_overall = safe_overall_r_dist.get(r, 0)
                            prob_temporal = safe_temporal_r_dist.get(r, 0)
                            combined_region_dist[r] = (temporal_weight * prob_temporal + (1 - temporal_weight) * prob_overall)
                        r_cats = list(combined_region_dist.keys())
                        r_probs = list(combined_region_dist.values())
                        r_total_prob = sum(r_probs)
                        if r_total_prob > 1e-9 and r_cats:
                            r_probs = [p / r_total_prob for p in r_probs]
                            current_row_region = np.random.choice(r_cats, p=r_probs)
                        elif r_cats: current_row_region = random.choice(r_cats)
                        else: current_row_region = "UNKNOWN_REGION"
                else:
                    current_row_region = "INVALID_MODE"

                # Assign the determined region to the row
                row_data[region_column] = current_row_region

                # --- ii) Generate Other Columns based on Date and Assigned Region ---
                for col, config in columns_config.items():
                    if col == region_column: continue # Skip region col, already assigned

                    is_numeric = config.get('type') == 'numeric'
                    value = None # Initialize
                    col_stats = temporal_stats.get(col, {})

                    # Get stats using fallback hierarchy based on current_row_region
                    source_stats = None; source_values = None

                    # Define fallback search order
                    search_order = [
                        ('regions', current_row_region, 'daily', dayofweek),
                        ('regions', current_row_region, 'monthly', month),
                        ('regions', current_row_region, 'overall', None),
                        ('overall_temporal', None, 'daily', dayofweek),
                        ('overall_temporal', None, 'monthly', month),
                        ('overall_all', None, None, None) # Overall column stats
                    ]

                    for level1, level2, level3, level4 in search_order:
                         temp_stats = col_stats.get(level1, {})
                         if level2: temp_stats = temp_stats.get(level2, {})
                         if level3: temp_stats = temp_stats.get(level3, {})
                         if level4 is not None: temp_stats = temp_stats.get(level4, {}) # Handles None for 'overall' time period

                         # Check if stats exist and meet criteria
                         if temp_stats and temp_stats.get('count', 0) >= (min_sample_size_direct if is_numeric else 1):
                              source_stats = temp_stats
                              if is_numeric: source_values = temp_stats.get('values')
                              break # Found suitable stats, stop searching

                    # --- Generate value based on selected source_stats ---
                    if source_stats:
                         if is_numeric:
                              # Prioritize direct sampling
                              if source_values is not None and len(source_values) > 0:
                                   value = random.choice(source_values)
                              else: # Fallback to mean/std
                                   mean, std = source_stats.get('mean', 0), source_stats.get('std', 0)
                                   std = max(0, std if std is not None else 0)
                                   value = np.random.normal(mean, std)
                         else: # Categorical (non-region)
                              dist = source_stats.get('distribution', {})
                              if dist and isinstance(dist, dict):
                                   cats, probs = list(dist.keys()), list(dist.values())
                                   total_p = sum(probs)
                                   if total_p > 1e-9 and cats:
                                        probs = [p / total_p for p in probs]
                                        value = np.random.choice(cats, p=probs)
                                   elif cats: value = random.choice(cats)
                                   else: value = f"no_cat_data_{col}"
                              else: value = f"missing_dist_{col}"
                    else: # No stats found anywhere
                         value = 0 if is_numeric else f"no_stats_{col}"


                    # --- iii) Apply Noise, Constraints, Rounding (Numeric only) ---
                    if is_numeric:
                        # Noise
                        if config.get('noise_enabled', False):
                             noise_level = config.get('noise_level', 0.1)
                             overall_std = col_stats.get('overall_all', {}).get('std', 0)
                             if overall_std is not None and overall_std > 0:
                                 value += np.random.normal(0, overall_std * noise_level)
                        # Clipping
                        overall_min = col_stats.get('overall_all', {}).get('min')
                        overall_max = col_stats.get('overall_all', {}).get('max')
                        if overall_min is not None and overall_max is not None:
                             value = np.clip(value, overall_min, overall_max)
                        value = np.clip(value, config.get('min_value', -np.inf), config.get('max_value', np.inf))
                        # Rounding
                        if not config.get('allow_decimals', True):
                             try: value = round(value)
                             except: pass

                    row_data[col] = value

                simulated_rows.append(row_data) # Append the completed row

            # --- Update progress ---
            if (i + 1) % max(1, total_dates // 20) == 0 or i == total_dates - 1:
                 progress = (i + 1) / total_dates
                 progress_bar.progress(progress)
                 status_text.text(f"Generating data: {current_date.strftime('%Y-%m-%d')} ({i + 1}/{total_dates})")

        # --- 4. Finalize DataFrame ---
        status_text.text("Finalizing DataFrame...")
        new_data_df = pd.DataFrame(simulated_rows)

        if append_data:
            st.info("Appending generated data to existing data...")
            try:
                 # Ensure date columns are compatible before concat
                 existing_data[date_column] = pd.to_datetime(existing_data[date_column], errors='coerce')
                 new_data_df[date_column] = pd.to_datetime(new_data_df[date_column], errors='coerce')

                 # Align columns - crucial for concat
                 # Use existing_data's columns as the reference order
                 all_final_cols = existing_data.columns.tolist()
                 existing_data_aligned = existing_data[all_final_cols]
                 # Add missing columns to new_data_df if any, fill with NaN
                 for col in all_final_cols:
                      if col not in new_data_df.columns:
                           new_data_df[col] = np.nan
                 new_data_aligned = new_data_df[all_final_cols]

                 final_df = pd.concat([existing_data_aligned, new_data_aligned], ignore_index=True)
            except Exception as concat_err:
                 st.error(f"Error during data concatenation: {concat_err}")
                 st.warning("Returning only the newly generated data due to concatenation error.")
                 final_df = new_data_df # Fallback
        else: # Not appending
            final_df = new_data_df
            # Add back non-simulated columns from original as NaNs and reorder
            simulated_cols_set = set(final_df.columns)
            original_cols = existing_data.columns.tolist()
            for col in original_cols:
                 if col not in simulated_cols_set:
                     final_df[col] = np.nan
            try: # Reorder to match original
                 final_df = final_df[original_cols]
            except KeyError:
                 st.warning("Could not reorder columns perfectly.")

        # Final sort by date
        if date_column in final_df.columns:
             try:
                 # Convert again just to be safe after potential concat issues
                 final_df[date_column] = pd.to_datetime(final_df[date_column])
                 final_df = final_df.sort_values(by=date_column).reset_index(drop=True)
             except Exception as sort_err:
                 st.warning(f"Final sorting by date failed: {sort_err}")

        progress_bar.progress(1.0)
        status_text.success("Simulation complete!")
        return final_df

    except Exception as e:
        st.error(f"An unexpected error occurred during simulation generation: {e}")
        st.error(traceback.format_exc())
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Advanced Time-Series & Regional Data Simulator (V3)")
st.write("""
Upload time-series data with a date column and a categorical region column.
This tool analyzes historical patterns (temporal and regional) to generate realistic synthetic data.
You can either generate data for a specific period or append new simulated data to your uploaded file.
""")

uploaded_file = st.file_uploader("Upload your dataset (CSV, XLSX, TXT)", type=["xlsx", "csv", "txt"])

if uploaded_file:
    # Load data using the robust loader
    existing_data = load_data(uploaded_file)

    if existing_data is not None and not existing_data.empty:
        st.subheader("Uploaded Data Preview")
        st.dataframe(existing_data.head())

        st.subheader("Basic Statistics (Original Data)")
        try:
             st.write(existing_data.describe(include='all', datetime_is_numeric=True))
        except Exception as desc_err:
             st.warning(f"Could not generate full description: {desc_err}")
             try: st.write(existing_data.describe())
             except: st.write("Could not generate basic description.")

        st.subheader("Simulation Configuration")

        col1, col2 = st.columns([1, 2]) # Adjust column widths if needed

        with col1:
            st.markdown("**Core Settings**")
             # Date Column Detection (moved from generate function for clarity)
            date_column = None
            date_options = [''] + existing_data.columns.tolist()
            selected_date_col = st.selectbox("Select the Date column:", date_options, index=0,
                                             help="Column containing dates for time-series analysis.")
            if selected_date_col:
                 try:
                     pd.to_datetime(existing_data[selected_date_col].dropna().iloc[0:5], errors='raise')
                     date_column = selected_date_col
                     st.success(f"'{date_column}' identified as Date Column.")
                 except:
                      st.error(f"Selected column '{selected_date_col}' could not be reliably interpreted as dates.")
                      date_column = None # Reset if validation fails

            # Region Column Selection
            string_cols = [''] + existing_data.select_dtypes(include=['object', 'category']).columns.tolist()
            region_column = st.selectbox("Select the primary Region column:", string_cols, index=0,
                                         help="Categorical column identifying regions/groups.")

            # Append Option
            append_data = st.checkbox("Append to uploaded data (continue from last date)", value=False,
                                     help="If checked, simulates data starting the day after the last date in your file.")

            # Region Sampling Mode
            region_sampling_mode = st.radio(
                "Region Sampling Mode:",
                options=['distribute_daily', 'one_per_day'],
                captions=["Sample region per row based on history (Allows multiple regions per day)",
                           "Sample one region for all rows per day based on history"],
                index=0, # Default to distribute_daily
                key='region_mode',
                 help="Determines how the Region column is assigned during simulation."
            )


            st.markdown("**Date Range for New Data**")
            today = datetime.today().date()
            # Default start date - show user input field regardless of append mode
            default_start = existing_data[date_column].min().date() if (date_column and not existing_data[date_column].isnull().all()) else datetime(2024, 1, 1).date()
            start_date_input = st.date_input("Start date for new data (or ignored if appending):", default_start)

            # Default end date
            default_end = today
            end_date = st.date_input("End date for new data:", default_end)

        with col2:
             st.markdown("**Column Simulation Selection**")
             numeric_columns = existing_data.select_dtypes(include=[np.number]).columns.tolist()
             other_categorical_cols = [c for c in string_cols if c and c != region_column] # Exclude region col itself

             selected_numeric = st.multiselect(
                 "Select NUMERIC columns to simulate:",
                 numeric_columns,
                 default=numeric_columns # Default to all numeric
             )

             selected_other_categorical = st.multiselect(
                 "Select OTHER CATEGORICAL columns to simulate (besides Region):",
                 other_categorical_cols,
                 default=[]
             )

             # Combine selected columns for configuration
             columns_to_simulate = selected_numeric + selected_other_categorical
             columns_config = {}

             if columns_to_simulate:
                st.markdown("**Advanced Column Configuration (Optional)**")
                with st.expander("Configure Noise, Constraints, Rounding", expanded=False):
                    # Configure numeric columns
                    for col in selected_numeric:
                        st.markdown(f"--- \n**`{col}`** (Numeric)")
                        config = {}
                        config['type'] = 'numeric'
                        try:
                            current_min = float(existing_data[col].min())
                            current_max = float(existing_data[col].max())
                            is_int = pd.api.types.is_integer_dtype(existing_data[col].dtype)
                        except: current_min, current_max, is_int = 0.0, 1.0, False

                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            config['min_value'] = cc1.number_input(f"Min Constraint", value=current_min, key=f"min_{col}", format="%g")
                            config['max_value'] = cc1.number_input(f"Max Constraint", value=current_max, key=f"max_{col}", format="%g")
                        with cc2:
                            config['allow_decimals'] = cc2.checkbox(f"Allow Decimals", value=not is_int, key=f"dec_{col}")
                        with cc3:
                            config['noise_enabled'] = cc3.checkbox(f"Add Noise", value=False, key=f"noise_en_{col}")
                            if config['noise_enabled']:
                                config['noise_level'] = cc3.slider(f"Noise Level", 0.0, 0.5, 0.05, key=f"noise_{col}", help="Relative to overall std dev")
                            else: config['noise_level'] = 0.0
                        columns_config[col] = config

                    # Configure other categorical columns (if specific options needed later)
                    for col in selected_other_categorical:
                         st.markdown(f"--- \n**`{col}`** (Categorical)")
                         columns_config[col] = {'type': 'string'} # Placeholder for future options
             else:
                 st.warning("No columns selected for simulation in the right panel.")

        # --- Generate Button ---
        st.divider()
        if st.button("🚀 Generate Simulated Data", type="primary", key="generate_button", disabled=(not date_column or not region_column or not columns_to_simulate)):
             if not date_column: st.error("Please select a valid Date column.")
             elif not region_column: st.error("Please select the primary Region column.")
             elif not columns_to_simulate: st.error("Please select at least one Numeric or Other Categorical column to simulate.")
             else:
                 # Add default configs for any selected columns not explicitly configured
                 for col in columns_to_simulate:
                     if col not in columns_config:
                         if col in selected_numeric:
                             try:
                                 min_val = float(existing_data[col].min()); max_val = float(existing_data[col].max())
                                 is_int = pd.api.types.is_integer_dtype(existing_data[col].dtype)
                             except: min_val, max_val, is_int = 0.0, 1.0, False
                             columns_config[col] = {'type': 'numeric', 'min_value': min_val, 'max_value': max_val, 'allow_decimals': not is_int, 'noise_enabled': False, 'noise_level': 0.0}
                         else: # Other categorical
                             columns_config[col] = {'type': 'string'}
                 # Add the region column itself to the config list for analysis function
                 if region_column not in columns_config:
                      columns_config[region_column] = {'type': 'string'} # Mark its type

                 with st.spinner("Hold tight! Analyzing patterns and generating simulation... This might take some time."):
                    # Call the V3 generation function
                    simulated_data = generate_simulated_data_v3(
                        existing_data=existing_data,
                        columns_config=columns_config,
                        start_date_input=start_date_input,
                        end_date=end_date,
                        date_column=date_column, # <<< Make sure this is passed
                        region_column=region_column,
                        append_data=append_data,
                        region_sampling_mode=region_sampling_mode
                        # temporal_weight=..., min_sample_size_direct=... (optional args)
                    )

                    if simulated_data is not None and not simulated_data.empty:
                        st.subheader("Simulation Results Preview")
                        st.dataframe(simulated_data.head())
                        st.dataframe(simulated_data.tail()) # Show tail, useful for append mode

                        # Distribution Plots (comparing FULL original vs FULL final)
                        st.subheader("Overall Distribution Comparison")
                        st.write("Comparing the distribution of the original data against the final (potentially appended) simulated data.")
                        plot_cols = [region_column] + selected_numeric + selected_other_categorical
                        for column in plot_cols:
                             if column in existing_data and column in simulated_data:
                                 try:
                                     st.write(f"**Distribution Comparison for `{column}`**")
                                     # Use your existing plotting function
                                     fig = plot_distribution_comparison(existing_data, simulated_data, column)
                                     st.pyplot(fig)
                                 except Exception as plot_err:
                                     st.warning(f"Could not plot distribution comparison for {column}: {plot_err}")
                             else:
                                  st.warning(f"Column '{column}' missing in original or final data. Cannot compare distribution.")


                        # Time Series Plots (Comparing original vs NEWLY generated part)
                        st.subheader("Time Series Comparison (Monthly Avg)")
                        st.write("Comparing monthly average of original data vs the *newly simulated portion*.")
                        try:
                             # Extract newly generated part for comparison
                             start_date_sim = final_start_date if append_data else start_date_input # Use the actual simulation start
                             new_part_df = simulated_data[pd.to_datetime(simulated_data[date_column]).dt.date >= start_date_sim].copy()

                             if not new_part_df.empty:
                                  for col in selected_numeric:
                                      if col in existing_data and col in new_part_df:
                                           try:
                                               fig, ax = plt.subplots(figsize=(15, 5))
                                               # Original Data Monthly Average
                                               orig_plot_data = existing_data.copy()
                                               orig_plot_data[date_column] = pd.to_datetime(orig_plot_data[date_column], errors='coerce')
                                               orig_monthly = orig_plot_data.set_index(date_column)[col].resample('M').mean()
                                               ax.plot(orig_monthly.index, orig_monthly.values, label='Original (Monthly Avg)', marker='o', linestyle='-', alpha=0.7)

                                               # Newly Simulated Data Monthly Average
                                               new_part_df[date_column] = pd.to_datetime(new_part_df[date_column], errors='coerce')
                                               sim_monthly = new_part_df.set_index(date_column)[col].resample('M').mean()
                                               ax.plot(sim_monthly.index, sim_monthly.values, label='Simulated Portion (Monthly Avg)', marker='x', linestyle='--', alpha=0.8)

                                               ax.set_title(f"Time Series Comparison for {col} (Monthly Average)")
                                               ax.set_ylabel(col); ax.set_xlabel("Date"); ax.legend(); plt.grid(True); plt.tight_layout()
                                               st.pyplot(fig)
                                           except Exception as ts_plot_err:
                                                st.warning(f"Could not plot time series for {col}: {ts_plot_err}")
                                      else: st.warning(f"Numeric column '{col}' missing for time series plot.")
                             else: st.info("No new data was generated in the specified range to plot time series comparison.")
                        except Exception as ts_outer_err:
                             st.warning(f"Could not prepare data for time series plotting: {ts_outer_err}")


                        # Download buttons
                        st.subheader("Download Results")
                        col_dl1, col_dl2 = st.columns(2)
                        try:
                            csv_output = simulated_data.to_csv(index=False, encoding='utf-8-sig')
                            col_dl1.download_button(label="Download as CSV", data=csv_output, file_name="simulated_data_final.csv", mime="text/csv")
                        except Exception as csv_err: col_dl1.error(f"CSV Generation Failed: {csv_err}")
                        try:
                            output_excel = io.BytesIO()
                            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                                simulated_data.to_excel(writer, index=False, sheet_name='SimulatedData')
                            excel_output_val = output_excel.getvalue()
                            col_dl2.download_button(label="Download as Excel", data=excel_output_val, file_name="simulated_data_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        except Exception as excel_err: col_dl2.error(f"Excel Generation Failed: {excel_err}")

                    elif simulated_data is not None and simulated_data.empty:
                         st.warning("Simulation ran but produced no data in the specified range.")
                    else: # simulated_data is None
                         st.error("Simulation failed. Please check the configuration and previous error messages.")
        else:
             # Conditional message if generate button is disabled
             if not date_column or not region_column:
                 st.warning("Please select both the Date and Region columns in the 'Core Settings' section.")
             elif not columns_to_simulate:
                 st.warning("Please select at least one Numeric or Other Categorical column to simulate in the right panel.")