import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Function to preprocess the data
def preprocess_data(data):
    # Handle numeric conversions and missing values
    num_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    for feature in num_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
        else:
            data[feature] = 0  # Assign default value if column is missing

    if 'cnt' not in data.columns:
        data['cnt'] = 0  # Add a placeholder column if 'cnt' is missing

    numcols = data[num_features + ['cnt']]
    for col in numcols.columns:
        numcols[col] = numcols[col].fillna(numcols[col].median())

    objcols = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]
    objcols = pd.get_dummies(objcols, drop_first=True)

    scaler = StandardScaler()
    numcols_scaled = pd.DataFrame(scaler.fit_transform(numcols), columns=numcols.columns)

    final_data = pd.concat([numcols_scaled, objcols], axis=1)
    return final_data.drop('cnt', axis=1), numcols_scaled['cnt']

# App Title
st.title("Bike Sharing Demand Prediction")

# Upload Test Data
uploaded_file = st.file_uploader("Upload Test Data (CSV format)", type="csv")

if uploaded_file:
    # Read uploaded test data
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview:")
    st.dataframe(test_data.head())

    # Preprocess the data
    X_test, y_test = preprocess_data(test_data)

    # Train model on existing data
    regmodel_new = LinearRegression().fit(X_test, y_test)

    # Make Predictions
    predictions = regmodel_new.predict(X_test)

    # Calculate Metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Display Metrics
    st.write(f"### Model Performance on Test Data")
    st.write(f"- **R-squared:** {r2:.4f}")
    st.write(f"- **RMSE:** {rmse:.4f}")

    # Plot actual vs predicted
    st.write("### Actual vs Predicted Values")
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}))

    # Allow user to upload a new dataset for predictions
    st.write("\nUpload another dataset for predictions:")
    prediction_file = st.file_uploader("Choose another CSV file for predictions", type="csv")

    if prediction_file is not None:
        new_data = pd.read_csv(prediction_file)

        if 'cnt' in new_data.columns:
            # Separate the actual count for RMSE calculation
            actual_cnt = new_data['cnt']
            prediction_data = new_data.drop(['cnt'], axis=1)
        else:
            st.warning("The uploaded dataset for predictions does not contain 'cnt'. RMSE cannot be calculated.")
            prediction_data = new_data

        # Preprocess the data for predictions
        processed_data = preprocess_data(prediction_data)[0]

        # Ensure the columns in processed_data match the ones used in the model training
        required_columns = regmodel_new.feature_names_in_  # Get the columns used during training
        missing_cols = [col for col in required_columns if col not in processed_data.columns]
        for col in missing_cols:
            processed_data[col] = 0  # Add missing column with a default value (e.g., 0)

        # Reorder columns to match the model's training data
        processed_data = processed_data[required_columns]

        # Generate predictions
        predictions_new = regmodel_new.predict(processed_data)

        st.write("Predictions generated successfully!")
        st.write(pd.DataFrame({"Predicted Demand Count": predictions_new}))

        # Calculate RMSE if actual count is provided
        if 'cnt' in new_data.columns:
            rmse_new = np.sqrt(mean_squared_error(actual_cnt, predictions_new))
            st.write(f"New Dataset RMSE: {rmse_new:.4f}")

            # Calculate R-Squared for new dataset
            r2_new = r2_score(actual_cnt, predictions_new)
            st.write(f"New Dataset R-Squared: {r2_new:.4f}")

else:
    st.write("Please upload a CSV file to evaluate the model performance.")
