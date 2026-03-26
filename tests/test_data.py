"""
Data quality tests — Stage 1 of the CI/CD pipeline.

These tests run BEFORE training. If the data is bad, there is no point
training a model from it. Failing here stops the pipeline immediately
and alerts the developer to fix the data source first.

Run with:
    pytest tests/test_data.py -v
"""

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match generate_data.py
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS  = {"amount", "hour", "day_of_week", "merchant_category", "is_fraud"}
VALID_CATEGORIES  = {"grocery", "restaurant", "retail", "online", "travel"}
MIN_TRAIN_ROWS    = 1_000
MAX_AMOUNT        = 100_000.0   # Sanity cap: no single transaction should exceed this
MIN_FRAUD_RATIO   = 0.001       # At least 0.1% fraud (otherwise data is suspicious)
MAX_FRAUD_RATIO   = 0.50        # No more than 50% fraud (otherwise data is suspicious)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — shared test data loaded once per test session
# A pytest fixture is a reusable setup block. Instead of loading the CSV
# in every test, we define it once here and inject it where needed.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_data() -> pd.DataFrame:
    """Load training data once and share it across all tests in this module."""
    return pd.read_csv("data/train.csv")


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Load test data once and share it across all tests in this module."""
    return pd.read_csv("data/test.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS
# Grouping tests in a class keeps them organized and allows shared fixtures.
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQuality:
    """Validate that the training and test datasets meet all requirements."""

    # ── Schema checks ────────────────────────────────────────────────────────

    def test_train_has_all_required_columns(self, train_data: pd.DataFrame):
        """All expected columns must be present in training data."""
        missing = REQUIRED_COLUMNS - set(train_data.columns)
        assert not missing, (
            f"Missing columns in train.csv: {missing}\n"
            f"Check generate_data.py to ensure these columns are created."
        )

    def test_test_has_all_required_columns(self, test_data: pd.DataFrame):
        """All expected columns must be present in test data."""
        missing = REQUIRED_COLUMNS - set(test_data.columns)
        assert not missing, f"Missing columns in test.csv: {missing}"

    # ── Size checks ──────────────────────────────────────────────────────────

    def test_train_has_enough_rows(self, train_data: pd.DataFrame):
        """Training data must have at least 1,000 rows to be meaningful."""
        assert len(train_data) >= MIN_TRAIN_ROWS, (
            f"Training set is too small: {len(train_data)} rows. "
            f"Need at least {MIN_TRAIN_ROWS}."
        )

    def test_test_is_smaller_than_train(self, train_data, test_data):
        """Test set should be smaller than training set."""
        assert len(test_data) < len(train_data), (
            "Test set is larger than or equal to training set. "
            "Check the train/test split ratio."
        )

    # ── Null value checks ────────────────────────────────────────────────────

    def test_no_null_values_in_train(self, train_data: pd.DataFrame):
        """No column in the training data should have any null values."""
        for col in REQUIRED_COLUMNS:
            null_count = train_data[col].isnull().sum()
            assert null_count == 0, (
                f"Column '{col}' in train.csv has {null_count} null values. "
                f"Check the data generation step."
            )

    def test_no_null_values_in_test(self, test_data: pd.DataFrame):
        """No column in the test data should have any null values."""
        for col in REQUIRED_COLUMNS:
            null_count = test_data[col].isnull().sum()
            assert null_count == 0, (
                f"Column '{col}' in test.csv has {null_count} null values."
            )

    # ── Value range checks ───────────────────────────────────────────────────

    def test_amounts_are_positive(self, train_data: pd.DataFrame):
        """Transaction amounts must be greater than zero."""
        invalid = (train_data["amount"] <= 0).sum()
        assert invalid == 0, (
            f"Found {invalid} non-positive transaction amounts in train.csv. "
            f"All amounts must be > 0."
        )

    def test_amounts_are_below_maximum(self, train_data: pd.DataFrame):
        """Transaction amounts must be below the sanity cap."""
        invalid = (train_data["amount"] > MAX_AMOUNT).sum()
        assert invalid == 0, (
            f"Found {invalid} amounts exceeding ${MAX_AMOUNT:,.0f} in train.csv."
        )

    def test_hours_are_valid(self, train_data: pd.DataFrame):
        """Hour values must be integers between 0 and 23 inclusive."""
        invalid = train_data[
            (train_data["hour"] < 0) | (train_data["hour"] > 23)
        ]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} invalid hour values in train.csv. "
            f"Hours must be 0–23."
        )

    def test_days_are_valid(self, train_data: pd.DataFrame):
        """Day-of-week values must be integers between 0 and 6 inclusive."""
        invalid = train_data[
            (train_data["day_of_week"] < 0) | (train_data["day_of_week"] > 6)
        ]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} invalid day_of_week values. "
            f"Days must be 0 (Monday) through 6 (Sunday)."
        )

    def test_merchant_categories_are_known(self, train_data: pd.DataFrame):
        """All merchant category values must be from the known set."""
        actual  = set(train_data["merchant_category"].unique())
        unknown = actual - VALID_CATEGORIES
        assert not unknown, (
            f"Unknown merchant categories found: {unknown}. "
            f"Valid categories are: {VALID_CATEGORIES}"
        )

    # ── Fraud ratio checks ───────────────────────────────────────────────────

    def test_fraud_ratio_is_realistic(self, train_data: pd.DataFrame):
        """
        Fraud ratio must be between 0.1% and 50%.

        If fraud is 0%, the model has nothing to learn from.
        If fraud is > 50%, the dataset is unrealistic (and training will fail).
        """
        ratio = train_data["is_fraud"].mean()
        assert MIN_FRAUD_RATIO <= ratio <= MAX_FRAUD_RATIO, (
            f"Fraud ratio is {ratio:.2%}, which is outside the expected "
            f"range of {MIN_FRAUD_RATIO:.1%}–{MAX_FRAUD_RATIO:.0%}."
        )

    # ── Data integrity checks ────────────────────────────────────────────────

    def test_train_and_test_have_different_sizes(self, train_data, test_data):
        """
        A basic leakage check: train and test should not be the same dataset.
        If they are equal size, something went wrong in the split.
        """
        assert len(train_data) != len(test_data), (
            "Train and test datasets have identical row counts. "
            "The train/test split may have failed."
        )

    def test_label_column_is_binary(self, train_data: pd.DataFrame):
        """is_fraud must only contain 0 and 1."""
        unique_values = set(train_data["is_fraud"].unique())
        assert unique_values <= {0, 1}, (
            f"is_fraud column contains unexpected values: {unique_values}. "
            f"Only 0 and 1 are allowed."
        )