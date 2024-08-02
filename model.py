# model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import joblib  # Use joblib for saving models

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df[df['price'] <= 125000000]
    df = df[df['noi'] <= 5000000]
    df = df[df['cap_rate'] <= 0.073]
    return df

# Define features and target
def prepare_data(df):
    X_real = df[['noi']]
    y_real = df['price']
    
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    X_real = sm.add_constant(X_real)

    # Train linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_real, y_train_real)
    
    # Train XGBoost model
    X = df[['multi_single_tenant', 'total_sf', 'federal_funds_rate', 'inflation', 'building_age', 'interest_rate']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    
    # Save models
    joblib.dump(lr_model, 'lr_model.pkl')
    joblib.dump(xgb_model, 'xgb_model.pkl')

# Function to generate synthetic data
def generate_synthetic_data(df, num_rows):
    synthetic_data = pd.DataFrame()
    a = df['price'].mean() / df['noi'].mean()  
    b = df['price'].mean() - a * df['noi'].mean()
    
    noi_min = df['noi'].min()
    noi_max = df['noi'].max()
    synthetic_data['noi'] = np.random.uniform(noi_min, noi_max, num_rows)
    
    noise_std = df['price'].std() * 0.1  
    synthetic_data['price'] = a * synthetic_data['noi'] + b + np.random.normal(0, noise_std, num_rows)
    
    for column in df.columns:
        if column not in ['price', 'noi']:
            if df[column].dtype in ['float64', 'int64']:
                min_val = df[column].min()
                max_val = df[column].max()
                min_val = max(0, min_val)
                synthetic_data[column] = np.random.uniform(min_val, max_val, num_rows)
            else:
                synthetic_data[column] = np.random.choice(df[column].dropna().unique(), num_rows)
    
    return synthetic_data

if __name__ == "__main__":
    df = load_and_preprocess_data('/Users/isaiaherb/Desktop/Northbow/Whole Foods/Data/WholeFoodsTrain.csv')
    df_synthetic = generate_synthetic_data(df, 1000)
    prepare_data(df_synthetic)
    print("Models saved successfully.")
