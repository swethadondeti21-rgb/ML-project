"""
Model performance tests — Stage 2 of the CI/CD pipeline.

These tests run AFTER training. They act as a quality gate:
only a model that meets all performance thresholds moves forward
to the Docker build and deployment stages.

If any test fails, the pipeline stops and the developer is notified.
The current model in production is NOT replaced.

Run with:
    pytest tests/test_model.py -v
"""

import pickle
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE THRESHOLDS
#
# These are the minimum acceptable values for each metric.
# Think of them as a contract: the model must meet these standards
# before it is allowed into production.
#
# How to set these values:
#   1. Train your first baseline model
#   2. Record its metrics
#   3. Set thresholds slightly below baseline (allow for small variation)
#   4. Raise thresholds over time as the model improves
# ─────────────────────────────────────────────────────────────────────────────

MIN_ACCURACY  = 0.90   # Must correctly classify at least 90% of transactions
MIN_F1        = 0.40   # F1 is the primary metric for imbalanced data
MIN_PRECISION = 0.40   # At least 40% of fraud flags must be real fraud
MIN_RECALL    = 0.30   # Must catch at least 30% of actual fraud cases

FEATURE_COLS = ["amount", "hour", "day_of_week", "merchant_encoded"]
TARGET_COL   = "is_fraud"


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_encoder():
    """Load the trained model and encoder from disk once for all tests."""
    try:
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pytest.fail(
            "models/model.pkl not found. "
            "Run 'python src/train.py' before running these tests."
        )


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Load test data and add the encoded merchant column."""
    return pd.read_csv("data/test.csv")


@pytest.fixture(scope="module")
def prepared_test_data(model_and_encoder, test_data):
    """Return X_test and y_test ready for model evaluation."""
    model, encoder = model_and_encoder
    df = test_data.copy()
    df["merchant_encoded"] = encoder.transform(df["merchant_category"])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TestModelPerformance:
    """Verify that the saved model meets all production quality standards."""

    # ── Basic sanity checks ──────────────────────────────────────────────────

    def test_model_file_loads(self, model_and_encoder):
        """The model pickle file must load without errors."""
        model, encoder = model_and_encoder
        assert model   is not None, "Model object is None after loading"
        assert encoder is not None, "Encoder object is None after loading"

    def test_model_has_predict_method(self, model_and_encoder):
        """The model must expose a .predict() method."""
        model, _ = model_and_encoder
        assert hasattr(model, "predict"), "Model does not have a predict() method"

    def test_model_has_predict_proba_method(self, model_and_encoder):
        """The model must be able to output probabilities, not just labels."""
        model, _ = model_and_encoder
        assert hasattr(model, "predict_proba"), (
            "Model does not support predict_proba(). "
            "Probability scores are required for the fraud detection use case."
        )

    def test_prediction_count_matches_input(self, model_and_encoder, prepared_test_data):
        """Model must return exactly one prediction per input row."""
        model, _ = model_and_encoder
        X, y     = prepared_test_data
        preds    = model.predict(X)
        assert len(preds) == len(X), (
            f"Model returned {len(preds)} predictions for {len(X)} inputs."
        )

    # ── Output validity checks ───────────────────────────────────────────────

    def test_predictions_are_binary(self, model_and_encoder, prepared_test_data):
        """All predictions must be either 0 (legit) or 1 (fraud)."""
        model, _ = model_and_encoder
        X, _     = prepared_test_data
        preds    = set(model.predict(X))
        assert preds <= {0, 1}, f"Model produced unexpected prediction values: {preds}"

    def test_model_predicts_both_classes(self, model_and_encoder, prepared_test_data):
        """
        Model must predict both classes on the test set.

        If the model only ever predicts 0 or only ever predicts 1,
        it has completely failed to learn the task — even if accuracy
        looks acceptable due to class imbalance.
        """
        model, _ = model_and_encoder
        X, _     = prepared_test_data
        preds    = set(model.predict(X))
        assert 0 in preds, "Model never predicts 'legitimate' (class 0). Model is broken."
        assert 1 in preds, "Model never predicts 'fraud' (class 1). Model learned nothing."

    def test_probabilities_are_in_valid_range(self, model_and_encoder, prepared_test_data):
        """All probability scores must be between 0.0 and 1.0 inclusive."""
        model, _ = model_and_encoder
        X, _     = prepared_test_data
        probs    = model.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0, f"Negative probability found: {probs.min()}"
        assert probs.max() <= 1.0, f"Probability > 1.0 found: {probs.max()}"

    # ── Performance threshold checks ─────────────────────────────────────────

    def test_accuracy_meets_threshold(self, model_and_encoder, prepared_test_data):
        """Test set accuracy must be at least MIN_ACCURACY."""
        model, _ = model_and_encoder
        X, y     = prepared_test_data
        acc      = accuracy_score(y, model.predict(X))
        assert acc >= MIN_ACCURACY, (
            f"Accuracy {acc:.4f} is below threshold {MIN_ACCURACY}. "
            f"The model may be undertrained or the data may have changed."
        )

    def test_f1_meets_threshold(self, model_and_encoder, prepared_test_data):
        """
        F1 score must be at least MIN_F1.

        This is the most important test. F1 balances precision and recall
        for the fraud class, making it suitable for imbalanced datasets.
        """
        model, _ = model_and_encoder
        X, y     = prepared_test_data
        f1       = f1_score(y, model.predict(X), zero_division=0)
        assert f1 >= MIN_F1, (
            f"F1 score {f1:.4f} is below threshold {MIN_F1}. "
            f"The model is not performing well enough on the minority class."
        )

    def test_precision_meets_threshold(self, model_and_encoder, prepared_test_data):
        """Precision must be at least MIN_PRECISION to avoid too many false alarms."""
        model, _ = model_and_encoder
        X, y     = prepared_test_data
        prec     = precision_score(y, model.predict(X), zero_division=0)
        assert prec >= MIN_PRECISION, (
            f"Precision {prec:.4f} is below threshold {MIN_PRECISION}. "
            f"Too many legitimate transactions are being flagged as fraud."
        )

    def test_recall_meets_threshold(self, model_and_encoder, prepared_test_data):
        """Recall must be at least MIN_RECALL to ensure enough fraud is caught."""
        model, _ = model_and_encoder
        X, y     = prepared_test_data
        rec      = recall_score(y, model.predict(X), zero_division=0)
        assert rec >= MIN_RECALL, (
            f"Recall {rec:.4f} is below threshold {MIN_RECALL}. "
            f"Too much actual fraud is slipping through undetected."
        )

    # ── Encoder checks ───────────────────────────────────────────────────────

    def test_encoder_knows_all_categories(self, model_and_encoder):
        """The encoder must recognize all five merchant categories."""
        _, encoder = model_and_encoder
        expected   = {"grocery", "restaurant", "retail", "online", "travel"}
        actual     = set(encoder.classes_)
        assert actual == expected, (
            f"Encoder does not match expected categories.\n"
            f"Expected: {expected}\n"
            f"Got:      {actual}"
        )