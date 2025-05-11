import streamlit as st
import joblib
import numpy as np

# Load model and scaler (replace paths with your actual files)
model = joblib.load("customer_segmentation.pkl")
scaler = joblib.load("scaler.pkl")  # If you used scaling during training

st.title("Customer Segmentation Model 🛒")

# Input fields (only Age and Income)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (Scale of 10)", min_value=1, max_value=100, value=5)  # Income scaled to 10

if st.button("Segment Customer"):
    # Prepare input (only Age and Income, in correct order)
    input_data = np.array([[age, income]])  # Shape: (1, 2)
    
    # Scale if needed (use the same scaler from training)
    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Map prediction to segments (update based on your model's classes)
    segments = {
        'Low': "Budget Shoppers",
        'Medium': "Mid-Range Spenders", 
        'High': "High-Income Customers"
    }
    
    st.success(f"Predicted Segment: **{segments[prediction[0]]}**")

# Optional: Add model explanation
st.markdown("---")
st.write("Note: Income is scaled to a factor of 10 (e.g., 5 = $50,000)")