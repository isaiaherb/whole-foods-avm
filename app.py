import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Use joblib for loading models
import matplotlib.pyplot as plt
import xgboost as xgb
from model import generate_synthetic_data, load_and_preprocess_data

st.logo('northbow.png')

# Load and prepare the data
df = load_and_preprocess_data('/Users/isaiaherb/Desktop/Northbow/Whole Foods/Data/WholeFoodsTrain.csv')
df_synthetic = generate_synthetic_data(df, 1000)

# Load saved models
lr_model = joblib.load('lr_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Streamlit app
st.title("Whole Foods Auto-Valuation Model")
st.sidebar.header("Model Inputs")

total_sf = st.sidebar.number_input("Total Square Footage", value=141818)
noi = st.sidebar.number_input("NOI", value=2065573.03)
multi_single_tenant = st.sidebar.number_input("Multi-Single Tenant", value=1)
federal_funds_rate = st.sidebar.number_input("Federal Funds Rate", value=0.10)
interest_rate = st.sidebar.number_input("Interest Rate", value=3.16)
inflation = st.sidebar.number_input("Inflation Rate", value=4.45)
building_age = st.sidebar.number_input("Building Age", value=31)

# Prepare input data for prediction
input_data = {
    'noi': noi,
    'multi_single_tenant': multi_single_tenant,
    'total_sf': total_sf,
    'federal_funds_rate': federal_funds_rate,
    'inflation': inflation,
    'building_age': building_age,
    'interest_rate': interest_rate
}

input_df = pd.DataFrame([input_data])

# Make prediction
lr_prediction = lr_model.predict(input_df[['noi']])
lr_prediction = np.round(lr_prediction, 2)  # Round LR prediction to 2 decimal places

# Create a DMatrix for XGBoost
dmatrix = xgb.DMatrix(input_df)

# Make XGBoost prediction
xgb_prediction = xgb_model.predict(dmatrix)

# Combine predictions
final_prediction = np.round(0.7 * lr_prediction + 0.3 * xgb_prediction,0)

# Display prediction
st.subheader(f"Predicted Price: ${final_prediction[0]:,.0f}")
# st.write(f"Predicted Price: ${final_prediction[0]:,.0f}")

# Calculate adjusted cap rate
adjusted_cap_rate = noi / final_prediction[0]
st.write(f"Adjusted Cap Rate: {adjusted_cap_rate:.2%}")

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df['price'], bins=30, alpha=0.7, label='Price Distribution')
axes[0].axvline(final_prediction[0], color='r', linestyle='dashed', linewidth=2, label='Predicted Price')
axes[0].set_title('Price Distribution')
axes[0].legend()

adjusted_cap_rates = df['noi'] / df['price']
axes[1].hist(adjusted_cap_rates, bins=30, alpha=0.7, label='Cap Rate Distribution')
axes[1].axvline(adjusted_cap_rate, color='r', linestyle='dashed', linewidth=2, label='Adjusted Cap Rate')
axes[1].set_title('Cap Rate Distribution')
axes[1].legend()

st.pyplot(fig)

# Reprocess DataFrame for display
df['lr_predictions'] = np.round(lr_model.predict(df[['noi']]), 0)  # Round LR predictions

dmatrix = xgb.DMatrix(df[['noi', 'multi_single_tenant', 'total_sf', 'federal_funds_rate', 'inflation', 'building_age', 'interest_rate']])
df['xgb_predictions'] = xgb_model.predict(dmatrix)
df['combined_prediction'] = np.round(df['lr_predictions'] * 0.7 + df['xgb_predictions'] * 0.3, 0)

df = df.drop(columns=['10_year_treasury', 'gdp', 'year_sold'])
# Display DataFrame
st.dataframe(df[['property_name', 'city', 'state', 'price', 'lr_predictions', 'xgb_predictions', 'combined_prediction'] + [col for col in df.columns if col not in ['property_name', 'city', 'state', 'price', 'lr_predictions', 'xgb_predictions', 'combined_prediction']]])

