#!/usr/bin/env python3
"""
Air Quality Data Preprocessing Script

This script preprocesses the measurements.csv file for time series forecasting
on Google Cloud Vertex AI AutoML.

Author: Generated for Week 5 Cloud-Native ML Platform Project
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime
import os
import sys
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("/home/huyvu/Projects/week5/logs/preprocessing.log", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function} | {message}", 
           level="DEBUG", rotation="10 MB")

def load_data(file_path):
    """Load the CSV file and perform initial inspection."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def remove_unnecessary_columns(df):
    """Keep only relevant columns for forecasting."""
    logger.info("Removing unnecessary columns")
    
    # Define columns to keep
    columns_to_keep = ['location_id', 'datetimeUtc', 'value', 'latitude', 'longitude']
    
    # Check if all required columns exist
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Keep only relevant columns
    df_cleaned = df[columns_to_keep].copy()
    logger.info(f"Kept columns: {list(df_cleaned.columns)}")
    logger.info(f"Shape after column removal: {df_cleaned.shape}")
    
    return df_cleaned

def remove_duplicate_headers(df):
    """Remove any duplicate header rows that appear in the data."""
    logger.info("Removing duplicate header rows")
    
    # Find rows where location_id equals 'location_id' (header rows)
    header_mask = df['location_id'] == 'location_id'
    num_header_rows = header_mask.sum()
    
    if num_header_rows > 0:
        logger.info(f"Found {num_header_rows} duplicate header rows. Removing them.")
        df_cleaned = df[~header_mask].copy()
    else:
        logger.info("No duplicate header rows found")
        df_cleaned = df.copy()
    
    logger.info(f"Shape after header removal: {df_cleaned.shape}")
    return df_cleaned

def ensure_numeric_target(df):
    """Ensure the target column (value) is numeric."""
    logger.info("Ensuring target column is numeric")
    
    # Convert value column to numeric, errors will be converted to NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Check for NaN values after conversion
    nan_count = df['value'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} non-numeric values in 'value' column, converted to NaN")
    
    # Log some statistics
    logger.info(f"Value column statistics:")
    logger.info(f"  Min: {df['value'].min()}")
    logger.info(f"  Max: {df['value'].max()}")
    logger.info(f"  Mean: {df['value'].mean():.2f}")
    
    return df

def check_timestamp_format(df):
    """Check and fix timestamp format."""
    logger.info("Checking timestamp format")
    
    try:
        # Try to parse datetimeUtc column
        df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])
        
        # Convert to ISO 8601 format with UTC timezone
        df['datetimeUtc'] = df['datetimeUtc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        logger.info("Timestamp format validated and standardized to ISO 8601")
        logger.info(f"Date range: {df['datetimeUtc'].min()} to {df['datetimeUtc'].max()}")
        
    except Exception as e:
        logger.error(f"Error parsing timestamps: {e}")
        sys.exit(1)
    
    return df

def ensure_timestamp_column(df, col='datetimeUtc'):
    """
    Parse and enforce RFC3339 timestamps in `col` for AutoML compatibility.
    - Coerce invalid values to NaT, drop those rows.
    - Format to 'YYYY-MM-DDTHH:MM:SSZ' (UTC).
    - Ensure proper datetime parsing and validation.
    """
    logger.info(f"Ensuring proper timestamp format for AutoML in column '{col}'")
    
    # Normalize and parse - handle both string and existing datetime types
    if df[col].dtype == 'object':
        # If it's string, clean and parse
        df[col] = pd.to_datetime(df[col].astype(str).str.strip(), utc=True, errors='coerce')
    else:
        # If already datetime, ensure UTC
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    
    # Report and drop unparsable rows
    invalid_count = int(df[col].isna().sum())
    if invalid_count > 0:
        logger.warning(f"{invalid_count} rows have unparsable '{col}'; dropping them before export")
        initial_rows = len(df)
        df = df.loc[df[col].notna()].copy()
        logger.info(f"Dropped {initial_rows - len(df)} rows with invalid timestamps")
    
    # Format as RFC3339 with Z (required for AutoML)
    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Validate the final format
    sample_timestamps = df[col].head(3).tolist()
    logger.info(f"Sample formatted timestamps: {sample_timestamps}")
    
    return df

def handle_missing_values(df):
    """Handle missing values using scikit-learn."""
    logger.info("Handling missing values")
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    logger.info("Missing values per column:")
    for col, count in missing_summary.items():
        if count > 0:
            logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # For numeric columns (latitude, longitude, value), use median imputation
    numeric_cols = ['latitude', 'longitude', 'value']
    
    for col in numeric_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Use SimpleImputer with median strategy
                imputer = SimpleImputer(strategy='median')
                df[col] = imputer.fit_transform(df[[col]]).flatten()
                logger.info(f"Imputed {missing_count} missing values in {col} using median")
    
    # For categorical columns (location_id), remove rows with missing values
    categorical_cols = ['location_id', 'datetimeUtc']
    for col in categorical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df = df.dropna(subset=[col])
                logger.info(f"Removed {missing_count} rows with missing {col}")
    
    logger.info(f"Shape after handling missing values: {df.shape}")
    return df

def remove_duplicates(df):
    """Remove duplicate rows based on location_id and datetimeUtc."""
    logger.info("Removing duplicate rows")
    
    initial_count = len(df)
    
    # Remove duplicates based on location_id and datetimeUtc
    df_deduped = df.drop_duplicates(subset=['location_id', 'datetimeUtc'], keep='first')
    
    removed_count = initial_count - len(df_deduped)
    logger.info(f"Removed {removed_count} duplicate rows")
    logger.info(f"Shape after deduplication: {df_deduped.shape}")
    
    return df_deduped

def save_cleaned_data(df, output_path):
    """Save the cleaned data as UTF-8 CSV without index for AutoML compatibility."""
    logger.info(f"Saving cleaned data to {output_path}")
    
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as UTF-8 CSV without index (critical for AutoML)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Cleaned data saved successfully. Final shape: {df.shape}")
        
        # Validate the saved file for AutoML compatibility
        logger.info("Validating saved file for AutoML compatibility...")
        test_df = pd.read_csv(output_path, nrows=5)
        logger.info(f"Column dtypes in saved file: {test_df.dtypes.to_dict()}")
        
        # Log final statistics
        logger.info("Final data summary:")
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Unique locations: {df['location_id'].nunique()}")
        logger.info(f"  Date range: {df['datetimeUtc'].min()} to {df['datetimeUtc'].max()}")
        logger.info(f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """Main preprocessing pipeline."""
    # Ensure logs directory exists
    os.makedirs("/home/huyvu/Projects/week5/logs", exist_ok=True)
    
    logger.info("Starting air quality data preprocessing")
    
    # Define file paths
    input_file = "/home/huyvu/Projects/week5/data/air_quality_dataset/measurements.csv"
    output_file = "/home/huyvu/Projects/week5/data/air_quality_dataset/measurements_cleaned.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Load data
    df = load_data(input_file)
    
    # Step 1: Remove unnecessary columns
    df = remove_unnecessary_columns(df)
    
    # Step 2: Remove duplicate headers
    df = remove_duplicate_headers(df)
    
    # Step 3: Ensure target column is numeric
    df = ensure_numeric_target(df)
    
    # Step 4: Check and fix timestamp format
    df = check_timestamp_format(df)
    
    # Step 4.5: Ensure timestamp column is properly formatted for AutoML
    df = ensure_timestamp_column(df, 'datetimeUtc')
    
    # Step 5: Handle missing values
    df = handle_missing_values(df)
    
    # Step 6: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 7: Save cleaned data (keeping original data order)
    save_cleaned_data(df, output_file)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()
