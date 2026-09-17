# Retail Sales & Demand Forecaster

A machine learning project that analyzes retail transaction data and predicts product quantity using regression models.

## Project Overview

The goal of this project is to explore whether retail transaction information can be used to predict the quantity of products sold.

The project includes data preprocessing, feature engineering, categorical encoding, multiple regression models, chronological validation, and model evaluation.

## Dataset

The project uses a retail sales dataset containing information about:

- Sales amount
- Profit
- Quantity
- Category
- Sub-Category
- Payment Mode
- Order Date
- State
- City

Date-based features were extracted from `Order Date`, including:

- Year
- Month
- Month Name
- Day
- Day of Week
- Week
- Weekend indicator

## Machine Learning Workflow

1. Data cleaning
2. Exploratory data analysis
3. Feature engineering
4. Numerical and categorical feature separation
5. One-hot encoding of categorical variables
6. Train-test splitting
7. Model training
8. Model evaluation
9. Time-aware validation
10. Model comparison
11. Model saving
12. Streamlit deployment

## Models Tested

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

A naive baseline was also used for comparison.

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Naive Baseline | 5.2135 | 5.9449 | -0.0081 |
| Linear Regression | 5.3138 | 6.0893 | -0.0577 |
| Random Forest | 4.9916 | 5.7889 | 0.0441 |
| Gradient Boosting | 4.8114 | 5.6370 | 0.0936 |

Gradient Boosting produced the strongest test-set performance among the evaluated models.

## Important Finding

The models showed limited predictive power for transaction-level quantity. This indicates that the available transaction features do not fully explain variations in quantity.

A monthly time-series analysis was also explored. Lag and rolling features were tested, but the resulting forecasting model did not outperform the simpler baseline.

This highlights an important machine learning lesson: increasing model complexity does not necessarily improve predictive performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## Project Structure

```text
Sale and demand forecaster/
│
├── data/
│    └── Sales Dataset.csv
│
├── model/
│   └── gradient_boosting_model.pkl
│
├── app.py
├── requirements.txt
└── README.md