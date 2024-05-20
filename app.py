import streamlit as st
import pandas as pd
import pickle

# Load the cleaned dataset
df1_no_duplicates = pd.read_csv(
    r"C:\Users\hp\Downloads\car_price_prediction.csv\car_price_prediction.csv")

# Define the features for independent variables (X)
features = ["Manufacturer", "Model", "Category", "Leather interior",
            "Fuel type", "Engine volume", "Mileage", "Cylinders",
            "Gear box type", "Drive wheels", "Doors", "Wheel",
            "Color", "Airbags"]

# Create a dictionary to map categorical features to numerical values
categorical_mappings = {
    "Manufacturer": {manufacturer: idx for idx, manufacturer in enumerate(df1_no_duplicates["Manufacturer"].unique())},
    "Model": {model: idx for idx, model in enumerate(df1_no_duplicates["Model"].unique())},
    "Color": {color: idx for idx, color in enumerate(df1_no_duplicates["Color"].unique())},
    # Fixed category mapping
    "Category": {"Sedan": 1, "Coupe": 2, "Hatchback": 3},
    "Leather interior": {"Yes": 1, "No": 0},
    "Fuel type": {fuel_type: idx for idx, fuel_type in enumerate(df1_no_duplicates["Fuel type"].unique())},
    "Gear box type": {gear_box_type: idx for idx, gear_box_type in enumerate(df1_no_duplicates["Gear box type"].unique())}
}

# Load the trained linear regression model from the pickle file


def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to preprocess input features


def preprocess_input(features_dict):
    processed_input = []
    for feature in features:
        if feature in ["Manufacturer", "Model", "Color", "Category"]:
            # For categorical features, map the selected value to its corresponding numeric value
            processed_input.append(categorical_mappings[feature].get(
                features_dict[feature], 0))  # Use .get() to handle missing keys
        elif feature in categorical_mappings:
            # For other categorical features, use the selected value directly
            processed_input.append(
                categorical_mappings[feature][features_dict[feature]])
        else:
            # For numeric features, use the value directly
            processed_input.append(features_dict[feature])
    return [processed_input]


# Load the trained model
model = load_model()

# Function to predict price


def predict_price(input_values):
    input_features = preprocess_input(input_values)
    return model.predict(input_features)[0]

# Streamlit app


def main():
    st.title("Car Price Prediction")
    st.write("Please provide the following information:")

    # Collect input values from the user
    input_values = {}
    for feature in features:
        if feature not in categorical_mappings:
            input_values[feature] = st.number_input(f"{feature}:")
        else:
            input_values[feature] = st.selectbox(
                f"{feature}:", options=df1_no_duplicates[feature].unique())

    # Predict price on button click
    if st.button("Predict Price"):
        price_prediction = predict_price(input_values)
        st.write(f"The predicted price is: ${price_prediction:.2f}")


# Run the app
if __name__ == "__main__":
    main()
