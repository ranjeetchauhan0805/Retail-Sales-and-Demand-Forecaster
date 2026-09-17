import streamlit as st
import pandas as pd
import joblib

model = joblib.load("C:\\Users\\newbl\\OneDrive\\Desktop\\Sale and demand forecaster\\models\\gradient_boosting_model.pkl")

st.set_page_config(
page_title="📊 Retail Sales & Demand Forecaster",
layout="wide"
)

st.title("📊 Retail Sales & Demand Forecaster")
st.write("Predict the expected quantity of products sold using a machine learning model.")

with st.sidebar:
    st.header("About the Project")
    st.write("This project uses retail transaction data and machine learning "
    "to predict product quantity."
)

st.subheader("Model")
st.write("Gradient Boosting Regressor")

st.subheader("Evaluation")
st.write("MAE: 4.81")
st.write("RMSE: 5.64")
st.write("R²: 0.094")

st.divider()
st.caption("Built with Python, Pandas, Scikit-learn and Streamlit")

st.header("Enter Transaction Details")

st.subheader("Sales Information")

amount = st.number_input("Amount", min_value=0.0, value=500.0)
profit = st.number_input("Profit", value=50.0)

category_subcategories = {
    "Electronics": ["Electronic Games", "Laptops", "Phones", "Printers"],
    "Furniture": ["Bookcases", "Chairs", "Sofas", "Tables"],
    "Office Supplies": ["Binders", "Markers", "Paper", "Pens"]
}

category = st.selectbox(
    "Category",
    list(category_subcategories.keys())
)

sub_category = st.selectbox(
    "Sub-Category",
    category_subcategories[category]
)

payment_mode = st.selectbox(
    "Payment Mode",
    ["COD", "UPI", "Debit Card", "Credit Card", "EMI"]
)

st.divider()

st.subheader("Location & Date")

state = st.text_input("State", "Maharashtra")
city = st.text_input("City", "Mumbai")

month_name = st.selectbox(
    "Month",
    [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
)

year = st.number_input(
    "Year",
    min_value=2020,
    max_value=2030,
    value=2025
)

month = st.number_input(
    "Month Number",
    min_value=1,
    max_value=12,
    value=3
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

day_of_week = st.number_input(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=5
)

week = st.number_input(
    "Week",
    min_value=1,
    max_value=53,
    value=11
)

is_weekend = st.selectbox(
    "Is Weekend?",
    [0, 1]
)

st.divider()

if st.button("Predict Quantity", use_container_width=True):

    input_data = pd.DataFrame([{
        "Amount": amount,
        "Profit": profit,
        "Category": category,
        "Sub-Category": sub_category,
        "PaymentMode": payment_mode,
        "State": state,
        "City": city,
        "Month_Name": month_name,
        "Year": year,
        "Month": month,
        "Day": day,
        "Day_of_Week": day_of_week,
        "Week": week,
        "Is_Weekend": is_weekend
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Quantity: {prediction:.2f} units")

    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Quantity", f"{prediction:.2f}")
    c2.metric("Category", category)
    c3.metric("Sub-Category", sub_category)

    if prediction < 8:
        st.info("The model predicts relatively low demand for this transaction.")
    elif prediction < 14:
        st.info("The model predicts moderate demand for this transaction.")
    else:
        st.info("The model predicts relatively high demand for this transaction.")

st.divider()

st.subheader("Model Performance")

performance = pd.DataFrame({
    "Model": [
    "Naive Baseline",
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
    ],
    "MAE": [5.2135, 5.3138, 4.9916, 4.8114],
    "RMSE": [5.9449, 6.0893, 5.7889, 5.6370],
    "R²": [-0.0081, -0.0577, 0.0441, 0.0936]
})

st.dataframe(performance, use_container_width=True)

st.caption(
"Note: Model performance is based on the chronological test set. "
"The relatively low R² indicates that transaction-level quantity "
"is difficult to predict using the available features."
)