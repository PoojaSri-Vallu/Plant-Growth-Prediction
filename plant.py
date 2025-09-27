# %%
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Plant Growth Prediction App")

# Load the trained logistic regression model
try:
    with open('plant.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'plant.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Yes', 'No'])  # ✅ Adjust based on model training classes

# Sidebar for user inputs
st.sidebar.header("Enter Plant Details")

# Numerical features
Sunlight_Hours = st.sidebar.slider("Sunlight Hours", min_value=0, max_value=12, value=6)
Temperature = st.sidebar.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
Humidity = st.sidebar.slider("Humidity (1-100)", min_value=1, max_value=100, value=50)

# Categorical features
Soil_Type = st.sidebar.selectbox("Soil Type", options=["Clay", "Sandy", "Loamy"])
Water_Frequency = st.sidebar.selectbox("Water Frequency", options=["Daily", "Weekly", "Bi-Weekly"])
Fertilizer_Type = st.sidebar.selectbox("Fertilizer Type", options=["organic", "chemical", "none"])


# Function to preprocess input data
def preprocess_input(Sunlight_Hours, Temperature, Humidity, Soil_Type, Water_Frequency, Fertilizer_Type):
    # Create a DataFrame with numerical features
    data = {
        'Sunlight Hours': Sunlight_Hours,
        'Temperature': Temperature,
        'Humidity': Humidity
    }
    df = pd.DataFrame([data])

    # One-hot encode Soil Type
    soil_types = ["Clay", "Sandy", "Loamy"]
    for soil in soil_types:
        df[f'Soil_Type_{soil}'] = 1 if Soil_Type == soil else 0

    # One-hot encode Water Frequency
    water_freqs = ["Daily", "Weekly", "Bi-Weekly"]
    for water in water_freqs:
        df[f'Water_Frequency_{water}'] = 1 if Water_Frequency == water else 0

    # One-hot encode Fertilizer Type
    fertilizers = ["organic", "chemical", "none"]
    for fertilizer in fertilizers:
        df[f'Fertilizer_Type_{fertilizer.capitalize()}'] = 1 if Fertilizer_Type.lower() == fertilizer else 0

    # Ensure all expected columns are present in the correct order
    expected_columns = [
        'Sunlight Hours', 'Temperature', 'Humidity',
        'Soil_Type_Clay', 'Soil_Type_Sandy', 'Soil_Type_Loamy',
        'Water_Frequency_Daily', 'Water_Frequency_Weekly', 'Water_Frequency_Bi-Weekly',
        'Fertilizer_Type_Organic', 'Fertilizer_Type_Chemical', 'Fertilizer_Type_None'
    ]

    df = df.reindex(columns=expected_columns, fill_value=0)
    return df


# Button to make prediction
if st.sidebar.button("Predict"):
    # Preprocess the input
    input_df = preprocess_input(
        Sunlight_Hours, Temperature, Humidity,
        Soil_Type, Water_Frequency, Fertilizer_Type
    )

    # Make prediction
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted plant growth is: **{predicted_label}**")
        if predicted_label == "No":
            st.write("No plant growth predicted.")
        else:
            st.write(f"The plant may have {predicted_label}.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter plant details such as sunlight hours, temperature, and humidity.
2. Choose the soil type, water frequency, and fertilizer type from the dropdown menus.
3. Click the 'Predict' button to see whether the plant is likely to grow successfully.
""")
