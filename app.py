import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.title("Transaction Amount Prediction")

data = pd.read_csv("transactions.csv")

st.subheader("Dataset Preview")
st.write(data.head())


data['t_amt'] = pd.to_numeric(data['t_amt'], errors='coerce')
data = data.dropna()


encoder_services = LabelEncoder()
encoder_products = LabelEncoder()
encoder_city = LabelEncoder()
encoder_state = LabelEncoder()
encoder_details = LabelEncoder()

data['services'] = encoder_services.fit_transform(data['services'])
data['products_used'] = encoder_products.fit_transform(data['products_used'])
data['city'] = encoder_city.fit_transform(data['city'])
data['state'] = encoder_state.fit_transform(data['state'])
data['t_details'] = encoder_details.fit_transform(data['t_details'])


X = data[['services','products_used','city','state','t_details']]
y = data['t_amt']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


st.subheader("Predict Transaction Amount")

services = st.selectbox(
    "Select Service",
    encoder_services.classes_
)

products = st.selectbox(
    "Select Product Used",
    encoder_products.classes_
)

city = st.selectbox(
    "Select City",
    encoder_city.classes_
)

state = st.selectbox(
    "Select State",
    encoder_state.classes_
)

payment = st.selectbox(
    "Payment Type",
    encoder_details.classes_
)

if st.button("Predict Transaction Amount"):

    input_data = np.array([[
        encoder_services.transform([services])[0],
        encoder_products.transform([products])[0],
        encoder_city.transform([city])[0],
        encoder_state.transform([state])[0],
        encoder_details.transform([payment])[0]
    ]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Transaction Amount: ${prediction[0]:.2f}")

y_pred = model.predict(X_test)

st.subheader("Model Evaluation")

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

st.write(results.head())

st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))