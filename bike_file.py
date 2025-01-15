import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Model building and preprocessing functions
def preprocess_data(df):
    """
    Preprocesses the data by separating dependent and independent variables,
    and encoding categorical columns.
    """
    numcols = df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = df[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]

    # Dummy encode categorical columns
    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])

    # Combine numerical and dummy-encoded categorical data
    df_final = pd.concat([numcols, objcols_dummy], axis=1)

    # Split into features (X) and target (y)
    X = df_final.drop('cnt', axis=1)
    y = df_final['cnt']

    # Drop multicollinear columns
    X = X.drop(['atemp', 'registered'], axis=1)

    return X, y

def build_model(X, y):
    """
    Builds and trains a linear regression model.
    """
    model = LinearRegression().fit(X, y)
    return model

def main():
    st.title("Bike Count Prediction")

    st.sidebar.header("Model Options")
    st.sidebar.markdown("Upload datasets to calculate metrics and make predictions.")

    # Upload training dataset
    st.header("Upload Training Data")
    train_file = st.file_uploader("Upload the training dataset (CSV format)", type="csv")

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("Training Data:", train_data.head())

        # Preprocess training data
        X_train, y_train = preprocess_data(train_data)

        # Train model
        model = build_model(X_train, y_train)

        # Calculate metrics
        train_r2 = model.score(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

        st.subheader("Training Metrics")
        st.write(f"R-Square: {train_r2:.4f}")
        st.write(f"RMSE: {train_rmse:.4f}")

        # Save the trained model
        joblib.dump(model, "trained_model.pkl")

    # Upload test dataset
    st.header("Upload Test Data")
    test_file = st.file_uploader("Upload the test dataset (CSV format)", type="csv")

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data:", test_data.head())

        if train_file is not None:
            # Load the saved model
            model = joblib.load("trained_model.pkl")

            # Preprocess test data
            X_test, y_test = preprocess_data(test_data)

            # Calculate RMSE on test data
            test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

            st.subheader("Test Metrics")
            st.write(f"Test RMSE: {test_rmse:.4f}")
        else:
            st.warning("Please upload the training data first to build the model.")

if __name__ == "__main__":
    main()