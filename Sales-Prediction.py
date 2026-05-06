# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("Sales Prediction System")

st.write("Enter advertising budget to predict sales")

# Load dataset
data = pd.read_csv("Advertising.csv")

# Features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# User Inputs
st.sidebar.header("Input Features")

tv = st.sidebar.slider("TV Advertising", 0, 300, 100)
radio = st.sidebar.slider("Radio Advertising", 0, 50, 25)
newspaper = st.sidebar.slider("Newspaper Advertising", 0, 100, 20)

# Create input dataframe
input_data = pd.DataFrame([[tv, radio, newspaper]],
                          columns=['TV', 'Radio', 'Newspaper'])

# Prediction
prediction = model.predict(input_data)

# Output
st.subheader("📈 Predicted Sales")
st.success(f"Predicted Sales: {prediction[0]:.2f}")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(data)

# Visualization
st.subheader("📊 Data Visualization")
st.line_chart(data[['TV', 'Radio', 'Newspaper']])
