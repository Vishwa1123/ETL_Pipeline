# Zomato Data ETL Pipeline
# Task 1: Data Pipeline Development for CODTECH Internship
# This script creates a data processing pipeline for restaurant data

import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
from sklearn.preprocessing import StandardScaler  # For scaling numerical values
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import matplotlib.pyplot as plt  # For creating charts
import seaborn as sns  # For better-looking charts
import os  # For file and directory operations
from datetime import datetime  # For timestamps in logs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Create a directory to store our processed data
output_dir = 'processed_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Also create a directory for visualizations
vis_dir = os.path.join(output_dir, 'visualizations')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# Set up a log file to track what our pipeline is doing
log_file = os.path.join(output_dir, 'pipeline_log.txt')

def log_message(message):
    """Write a message to our log file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)  # Also print to console
    
    # Write to log file
    with open(log_file, 'a') as f:
        f.write(log_message + "\n")

# STEP 1: EXTRACT - Load data from the source file
def extract_data(file_path):
    """Read the data from a CSV file"""
    log_message(f"Loading data from {file_path}")
    
    try:
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        log_message(f"Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        return None

# STEP 2: TRANSFORM - Clean and prepare the data
def transform_data(data):
    """Clean and transform the raw data"""
    log_message("Starting data transformation")
    
    # Make a copy so we don't change the original data
    df = data.copy()
    
    # 2.1: Handle missing values
    log_message("Handling missing values")
    
    # For number columns: fill missing values with median (middle value)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:  # If there are missing values
            median_val = df[col].median()  # Get the median value
            df[col].fillna(median_val, inplace=True)  # Fill missing values with median
            log_message(f"  Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")
    
    # For text columns: fill missing values with "Unknown"
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:  # If there are missing values
            df[col].fillna("Unknown", inplace=True)  # Fill with "Unknown"
            log_message(f"  Filled {df[col].isnull().sum()} missing values in {col} with 'Unknown'")
    
    # 2.2: Create new helpful columns (feature engineering)
    log_message("Creating new features")
    
    # If we have a price range column, create friendly labels
    if 'Price range' in df.columns:
        # Map numeric price ranges to descriptive categories
        price_map = {1: 'Budget', 2: 'Mid-range', 3: 'High-end', 4: 'Luxury'}
        df['Price Category'] = df['Price range'].map(price_map)
        log_message("  Created 'Price Category' column from 'Price range'")
    
    # If we have cuisine information, extract the main cuisine
    if 'Cuisines' in df.columns:
        # Get the first cuisine listed (the primary one)
        df['Primary Cuisine'] = df['Cuisines'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notnull(x) else 'Unknown')
        log_message("  Created 'Primary Cuisine' column from 'Cuisines'")
    
    # 2.3: Handle extreme values (outliers)
    log_message("Handling outliers in numerical columns")
    
    # For columns with costs or ratings, cap extreme values
    for col in ['Average Cost for two', 'Aggregate rating']:
        if col in df.columns:
            # Get reasonable lower and upper limits (1% and 99% values)
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            
            # Count how many outliers we have
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            
            if outliers > 0:
                # Replace outliers with the boundary values
                df[col] = df[col].clip(lower=lower, upper=upper)
                log_message(f"  Fixed {outliers} outliers in '{col}'")
    
    # 2.4: Create some simple visualizations to understand the data
    log_message("Creating visualizations")
    
    # Chart 1: Distribution of ratings
    if 'Aggregate rating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Aggregate rating'], kde=True)
        plt.title('Distribution of Restaurant Ratings')
        plt.savefig(os.path.join(vis_dir, 'rating_distribution.png'))
        plt.close()
        log_message("  Created rating distribution chart")
    
    # Chart 2: Average cost distribution
    if 'Average Cost for two' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Average Cost for two'], kde=True, bins=30)
        plt.title('Distribution of Average Cost for Two People')
        plt.savefig(os.path.join(vis_dir, 'cost_distribution.png'))
        plt.close()
        log_message("  Created cost distribution chart")
    
    log_message(f"Transformation complete. Final shape: {df.shape}")
    return df

# STEP 3: LOAD - Prepare data for modeling and save results
def load_data(df, target_column='Aggregate rating'):
    """Prepare data for modeling and save processed files"""
    log_message("Preparing data for modeling and saving results")
    
    # 3.1: Split data into features (X) and target variable (y)
    if target_column in df.columns:
        log_message(f"Using '{target_column}' as the target variable")
        X = df.drop(columns=[target_column])  # Features (everything except target)
        y = df[target_column]  # Target variable
    else:
        log_message(f"Target column '{target_column}' not found. Skipping model preparation.")
        X = df.copy()
        y = None
    
    # 3.2: Keep only numerical columns for simplicity
    # In a real project, we would properly encode categorical variables
    X_numeric = X.select_dtypes(include=['number'])
    log_message(f"Selected {X_numeric.shape[1]} numerical features for modeling")
    
    # 3.3: Split into training and testing sets
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42)
        log_message(f"Split data into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")
        
        # Save train and test data
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
        
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        log_message("Saved training and testing datasets")
        
        return X_train, X_test, y_train, y_test
    else:
        # Return the numeric features and None for the other values
        return X_numeric, None, None, None
    
    # 3.4: Save the fully processed data
    df.to_csv(os.path.join(output_dir, 'zomato_processed.csv'), index=False)
    log_message(f"Saved complete processed dataset to {os.path.join(output_dir, 'zomato_processed.csv')}")
    
    # 3.5: Create and save a data profile with column information
    column_profile = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    column_profile.to_csv(os.path.join(output_dir, 'column_profile.csv'), index=False)
    log_message("Saved column profile information")
    
    return X_train, X_test, y_train, y_test if y is not None else (X_numeric, None, None, None)

# Main function to run the entire pipeline
def run_etl_pipeline(data_path):
    """Run the complete ETL pipeline"""
    log_message("Starting Zomato data ETL pipeline")
    
    # Step 1: Extract data
    data = extract_data(data_path)
    if data is None:
        log_message("Pipeline stopped due to data extraction error")
        return
    
    # Step 2: Transform data
    transformed_data = transform_data(data)
    
    # Step 3: Load data (prepare for modeling and save)
    X_train, X_test, y_train, y_test = load_data(transformed_data)
    
    log_message("ETL pipeline completed successfully!")
    return transformed_data

# To run the pipeline, call this function with the path to your data file
if __name__ == "__main__":
    # Change this to the path where you saved the Zomato dataset
    data_path = "Zomato-data-.csv"
    
    # Run the pipeline
    processed_data = run_etl_pipeline(data_path)
    
    if processed_data is not None:
        print(f"Successfully processed {processed_data.shape[0]} restaurant records!")
        print(f"Check the '{output_dir}' folder for all outputs")