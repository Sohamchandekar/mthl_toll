import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

import pandas as pd


def prepare_data_for_model(data):
    records = []
    # Define the official toll prices for each vehicle type
    official_tolls = {
        'CAR': 250,
        'LCV\\MINIBUS': 400,
        'BUS\\2-AXLETRUCK': 830,
        'MAV(3AXLE)': 905,
        'MAV(4-6AXLE)': 1300,
        'OVERSIZE': 1580
    }

    for date, vehicles in data.items():
        for vehicle_type, methods in vehicles.items():
            for payment_type, details in methods.items():
                if payment_type in ['fasttag', 'cash', 'upi']:
                    records.append({
                        'Date': date,
                        'VehicleType': vehicle_type,
                        'PaymentType': payment_type,
                        'Traffic': details['Traffic'],
                        'Revenue': details['Revenue']
                    })

    df = pd.DataFrame(records)

    # Add official toll column based on the vehicle type
    df['OfficialToll'] = df['VehicleType'].apply(lambda x: official_tolls.get(x, 0))  # Default to 0 if not found

    # Daily metrics
    daily_agg = df.groupby(['Date', 'VehicleType']).agg({
        'Traffic': 'sum',
        'Revenue': 'sum',
        'OfficialToll': 'mean'  # Average toll for each vehicle type
    }).reset_index()

    # Add average toll per vehicle, handle divide by zero
    daily_agg['AvgToll'] = daily_agg.apply(
        lambda row: row['Revenue'] / row['Traffic'] if row['Traffic'] > 0 else 0, axis=1
    )

    return daily_agg

def add_custom_features(data):
    # Add Price Sensitivity based on assumptions
    price_sensitivity_map = {
        'CAR': 0.3,  # Cars are less price-sensitive
        'BUS\\2-AXLETRUCK': 0.5,
        'LCV\\MINIBUS': 0.5,
        'MAV(3AXLE)': 0.8,
        'MAV(4-6AXLE)': 0.8,
        'OVERSIZE': 0.9  # Heavily price-sensitive
    }
    data['PriceSensitivity'] = data['VehicleType'].map(price_sensitivity_map)
    return data


def prepare_data(data):
    # Handle missing values
    data['AvgToll'] = data['AvgToll'].fillna(0)
    data['OfficialToll'] = data['OfficialToll'].fillna(0)

    # Add custom features
    data = add_custom_features(data)

    # Feature Matrix (X) and Targets (y)
    X = data[['VehicleType', 'AvgToll', 'OfficialToll', 'PriceSensitivity']]
    y = data[['Traffic', 'Revenue']]

    return X, y


# Train Robust Models
def train_robust_model(X, y):
    # One-Hot Encoding for VehicleType
    categorical_features = ['VehicleType']
    categorical_transformer = OneHotEncoder(drop='first')  # Avoid multicollinearity

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Gradient Boosting Model with MultiOutput
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(GradientBoostingRegressor()))
    ])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.2f}")
    print(f"Test R^2 Score: {test_score:.2f}")

    return model


def simulate_scenarios(model, vehicle_type, official_toll):
    """
    Simulate the effect of changing the official toll on predicted traffic and revenue.
    """
    # Price sensitivity mapping
    price_sensitivity_map = {
        'CAR': 0.2, 'BUS\\2-AXLETRUCK': 0.5, 'LCV\\MINIBUS': 0.5,
        'MAV(3AXLE)': 0.8, 'MAV(4-6AXLE)': 0.8, 'OVERSIZE': 0.9
    }
    price_sensitivity = price_sensitivity_map.get(vehicle_type, 0.5)

    # Predict traffic and revenue
    input_data = pd.DataFrame({
        'VehicleType': [vehicle_type],
        'AvgToll': [official_toll],  # AvgToll equals OfficialToll for simulation
        'OfficialToll': [official_toll],
        'PriceSensitivity': [price_sensitivity]
    })
    prediction = model.predict(input_data)  # Predict Traffic and Revenue
    predicted_traffic = prediction[0][0]
    predicted_revenue = prediction[0][1]

    # Adjusted traffic based on assumptions
    if official_toll > 500:  # Example threshold
        predicted_traffic *= (1 - price_sensitivity)  # Reduce traffic

    # Recalculate revenue
    recalculated_revenue = predicted_traffic * official_toll
    return predicted_traffic, predicted_revenue, recalculated_revenue


# Adjusted Back-Calculation for Target Revenue
def required_toll_for_revenue(model, vehicle_type, target_revenue):
    """
    Calculate the required toll to achieve a target revenue dynamically.
    """
    best_toll_price = None
    best_difference = float('inf')
    predicted_traffic_at_best_toll = 0

    for toll_price in np.linspace(250, 800, 500):  # Adjust range as per data
        predicted_traffic, predicted_revenue, recalculated_revenue = simulate_scenarios(
            model, vehicle_type, toll_price
        )
        difference = abs(recalculated_revenue - target_revenue)

        if difference < best_difference:
            best_difference = difference
            best_toll_price = toll_price
            predicted_traffic_at_best_toll = predicted_traffic

    return best_toll_price, predicted_traffic_at_best_toll


# Example Usage
# Assuming `prediction_base_data` is your dataset with the new `OfficialToll` column
# Prepare data with additional features

# # prediction_base_data = add_custom_features(prediction_base_data)
# X, y = prepare_data(prediction_base_data)
#
# # Train the model
# model = train_robust_model(X, y)
#
# # Scenario 1: Simulate traffic flow with different official tolls
# vehicle_type = 'CAR'
# official_toll = 200
# predicted_traffic, predicted_revenue, recalculated_revenue = simulate_scenarios(
#     model, vehicle_type, official_toll
# )
# print(
#     f"For {vehicle_type} with OfficialToll {official_toll}: Predicted Traffic = {predicted_traffic:.2f}, Predicted Revenue = {predicted_revenue:.2f}")
#
# # Scenario 2: Calculate required toll for target revenue
# target_revenue = 5000000
# required_toll, predicted_traffic = required_toll_for_revenue(model, vehicle_type, target_revenue)
# print(f"To achieve Target Revenue {target_revenue} for {vehicle_type}, Required AvgToll = {required_toll:.2f}")