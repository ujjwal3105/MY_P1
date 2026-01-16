import numpy as np 
import pandas as pd 
import streamlit as st 
import joblib
import os
MODEL_PATH=os.path.join(os.path.dirname(__file__),"BEST.joblib")

@st.cache_resource
def load_model():
    return joblib.load("BEST.joblib")
model=load_model()

model=joblib.load("BEST.joblib")

st.title("HOUSE PRICE PREDECTION SYSTEM")
st.title(" Bangalore House Price Prediction App")
st.write("Predict house prices using a trained Machine Learning model")

st.markdown("---")

# ---------------------------------------
# USER INPUTS
# ---------------------------------------
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1)

with col2:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    total_sqft = st.number_input(
        "Total Square Feet", min_value=300, max_value=10000, value=1200
    )

area_type = st.selectbox(
    "Area Type",
    [
        "Built-up Area",
        "Super built-up Area",
        "Plot Area",
        "Carpet Area"
    ]
)

# ---------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------
input_data = pd.DataFrame({
    "bath": [bath],
    "balcony": [balcony],
    "total_sqft_int": [total_sqft],
    "bhk": [bhk]
})

# ---------------------------------------
# ONE HOT ENCODING (AREA TYPE)
# ---------------------------------------
area_columns = [
    "area_typeBuilt-up Area",
    "area_typeCarpet Area",
    "area_typePlot Area",
    "area_typeSuper built-up Area"
]

for col in area_columns:
    input_data[col] = 0

input_data[f"area_type{area_type}"] = 1

# ---------------------------------------
# MATCH MODEL FEATURES (CRITICAL STEP)
# ---------------------------------------
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns exactly like training
input_data = input_data[model_features]

# ---------------------------------------
# PREDICTION
# ---------------------------------------
st.markdown("---")

if st.button(" Predict House Price"):
    prediction = model.predict(input_data)[0]

    st.success(f" Estimated Price: ₹ {prediction:,.0f}")
    st.caption("Prediction is based on historical Bangalore housing data")

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
