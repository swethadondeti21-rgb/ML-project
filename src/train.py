"""
Train a Random Forest fraud detection model.

This script is the heart of our ML pipeline. It is called by the
CI/CD workflow every time code is pushed, so it must be:
  1. Reproducible  -- same inputs always produce the same model
  2. Self-contained -- no manual steps required
  3. Clearly logged -- so we can debug failures in CI

How to run:
-----------
    python src/train.py

    (Run generate_data.py first to create the CSV files)

Output:
-------
    models/model.pkl  -- Pickled tuple of (model, encoder)
                         Both are needed together at inference time.
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# All tunable parameters are at the top so they are easy to find and change.
# ─────────────────────────────────────────────────────────────────────────────

# Paths
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
MODEL_PATH = "models/model.pkl"

# Feature columns (must match the columns in generate_data.py)
# We add "merchant_encoded" in code — it replaces "merchant_category"
FEATURE_COLS = ["amount", "hour", "day_of_week", "merchant_encoded"]
TARGET_COL   = "is_fraud"

# Model hyperparameters
# n_estimators : Number of decision trees in the forest
#                More trees = more stable but slower
# max_depth    : Maximum depth of each tree
#                Deeper = can learn more but risks overfitting
# random_state : Fixed seed so training is reproducible
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth":    10,
    "random_state": 42,
    "n_jobs":       -1,   # Use all available CPU cores
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test CSV files from disk.

    Raises a clear error if the files do not exist, with instructions
    to run generate_data.py first.
    """
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Run 'python src/generate_data.py' to create it first."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    print(f"  Loaded training data : {len(train_df):,} rows")
    print(f"  Loaded test data     : {len(test_df):,} rows")
    print(f"  Training fraud ratio : {train_df[TARGET_COL].mean():.2%}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder]:
    """
    Encode categorical features and split into X (features) and y (labels).

    Why LabelEncoder?
    -----------------
    Machine learning models work with numbers, not strings. "online", "grocery"
    etc. must be converted to integers: 0, 1, 2, 3, 4.

    We fit the encoder on TRAINING data only, then use the same mapping
    to transform both train and test. This is critical: if we fit on all
    data, we leak test information into training (a form of data leakage).

    We also SAVE the encoder alongside the model, because at inference time
    we need to apply the exact same mapping to new transactions.

    Returns:
        X_train, X_test, y_train, y_test, encoder
    """
    encoder = LabelEncoder()

    # Fit on training data only, then transform both splits
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df["merchant_encoded"] = encoder.fit_transform(
        train_df["merchant_category"]
    )
    test_df["merchant_encoded"] = encoder.transform(
        test_df["merchant_category"]
    )

    category_map = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"  Merchant encoding: {category_map}")

    X_train = train_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    y_test  = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test, encoder


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Why Random Forest?
    ------------------
    - Handles mixed numeric and categorical (encoded) features well
    - Naturally resistant to overfitting compared to single decision trees
    - Gives us feature importance scores for free
    - Does not require feature scaling (no need to normalize amounts)
    - Fast to train and easy to interpret

    Args:
        X_train : Feature matrix (rows = transactions, cols = features)
        y_train : Labels (0 = legitimate, 1 = fraud)

    Returns:
        A trained RandomForestClassifier
    """
    print(f"\n  Training Random Forest with params: {MODEL_PARAMS}")

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("  Training complete!")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:   RandomForestClassifier,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Why these metrics?
    ------------------
    - Accuracy alone is misleading for imbalanced data (2% fraud).
      A model that always says "not fraud" gets 98% accuracy but is useless.

    - Precision = "Of all transactions flagged as fraud, how many were?"
      High precision = fewer false alarms for customers.

    - Recall = "Of all actual fraud, how many did we catch?"
      High recall = fewer fraud cases slip through undetected.

    - F1 Score = harmonic mean of precision and recall.
      This is our most important metric for imbalanced classification.

    Args:
        model  : Trained classifier
        X_test : Test feature matrix
        y_test : True labels for test set

    Returns:
        Dictionary of metric names to float values
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of fraud

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred,    zero_division=0),
        "f1":        f1_score(y_test, y_pred,        zero_division=0),
    }

    # Print detailed evaluation report
    print("\n  ── Model Evaluation on Test Set ───────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}  <-- Most important metric")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  (legit, correctly flagged legit) : {cm[0][0]:,}")
    print(f"    False Positives (legit, wrongly flagged fraud)   : {cm[0][1]:,}")
    print(f"    False Negatives (fraud, missed!)                 : {cm[1][0]:,}")
    print(f"    True Positives  (fraud, correctly caught)        : {cm[1][1]:,}")

    print(f"\n  Feature Importance:")
    for name, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(imp * 40)
        print(f"    {name:<20} {imp:.4f}  {bar}")

    print("  ────────────────────────────────────────────────────────")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: RandomForestClassifier, encoder: LabelEncoder) -> None:
    """
    Save the trained model and encoder together as a single pickle file.

    Why save them together?
    -----------------------
    At inference time, both objects are needed:
      1. The encoder converts 'online' -> 3 (integer)
      2. The model uses the integer to make a prediction
    Saving them as a tuple in one file prevents them from getting out of sync.

    Args:
        model   : Trained RandomForestClassifier
        encoder : Fitted LabelEncoder (trained on merchant categories)
    """
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, encoder), f)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n  Model saved to {MODEL_PATH}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Fraud Detection Model Training ===\n")

    print("Step 1/4: Loading data...")
    train_df, test_df = load_data()

    print("\nStep 2/4: Preprocessing...")
    X_train, X_test, y_train, y_test, encoder = preprocess(train_df, test_df)

    print("\nStep 3/4: Training model...")
    model = train_model(X_train, y_train)

    print("\nStep 4/4: Evaluating and saving...")
    evaluate_model(model, X_test, y_test)
    save_model(model, encoder)

    print("\n=== Training complete! Run 'pytest tests/' to validate. ===\n")


if __name__ == "__main__":
    main()