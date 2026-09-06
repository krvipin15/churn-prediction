"""Dataset Validation Schemas.

This module defines Pandera schemas for validating the structure, data types,
and value constraints of customer churn datasets. It ensures data quality
by enforcing strict rules on nullability, uniqueness, and allowed categories
across both raw and processed data stages.
"""

import pandera.pandas as pa

RAW_BASE_SCHEMA = pa.DataFrameSchema(
    {
        "gender": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Male", "Female"]),
        ),
        "SeniorCitizen": pa.Column(
            pa.Int,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "Partner": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No"]),
        ),
        "Dependents": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No"]),
        ),
        "tenure": pa.Column(
            pa.Int,
            nullable=True,
            checks=pa.Check.ge(0),
        ),
        "PhoneService": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No"]),
        ),
        "MultipleLines": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No phone service"]),
        ),
        "InternetService": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["DSL", "Fiber optic", "No"]),
        ),
        "OnlineSecurity": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "OnlineBackup": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "DeviceProtection": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "TechSupport": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingTV": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingMovies": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Contract": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Month-to-month", "One year", "Two year"]),
        ),
        "PaperlessBilling": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(["Yes", "No"]),
        ),
        "PaymentMethod": pa.Column(
            pa.String,
            nullable=True,
            checks=pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.ge(0.0),
        ),
        "TotalCharges": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.ge(0.0),
        ),
    },
    name="RawCustomerFeatures",
    strict=False,
    coerce=True,
)

RAW_SCHEMA = RAW_BASE_SCHEMA.add_columns(
    {
        "Churn": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check.isin(["Yes", "No", "0", "1"]),
        ),
    }
)

PROCESSED_BASE_SCHEMA = pa.DataFrameSchema(
    {
        "gender": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "SeniorCitizen": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "Partner": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "Dependents": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "tenure": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.ge(0),
        ),
        "PhoneService": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "MultipleLines": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "InternetService": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "OnlineSecurity": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "OnlineBackup": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "DeviceProtection": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "TechSupport": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "StreamingTV": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "StreamingMovies": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "Contract": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2]),
        ),
        "PaperlessBilling": pa.Column(
            pa.Int8,
            nullable=True,
            checks=pa.Check.isin([0, 1]),
        ),
        "PaymentMethod": pa.Column(
            pa.Int16,
            nullable=True,
            checks=pa.Check.isin([-1, 0, 1, 2, 3]),
        ),
        "MonthlyCharges": pa.Column(
            pa.Float32,
            nullable=True,
            checks=pa.Check.ge(0.0),
        ),
        "TotalCharges": pa.Column(
            pa.Float32,
            nullable=True,
            checks=pa.Check.ge(0.0),
        ),
    },
    name="ProcessedCustomerFeatures",
    strict=False,
    coerce=True,
)

PROCESSED_SCHEMA = PROCESSED_BASE_SCHEMA.add_columns(
    {
        "Churn": pa.Column(
            pa.Int8,
            nullable=False,
            checks=pa.Check.isin([0, 1]),
        ),
    }
)
