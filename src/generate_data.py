"""
Generate a synthetic credit card fraud detection dataset.

Why synthetic data?
-------------------
Real fraud datasets are confidential and hard to obtain. This script
creates fake-but-realistic data so the tutorial is fully self-contained.
The data mimics real patterns:
  - Most transactions are legitimate (98%)
  - Fraudulent ones are larger, happen late at night, and are online/travel

How to run:
-----------
    python src/generate_data.py

Output:
-------
    data/train.csv  -- 8,000 transactions for training
    data/test.csv   -- 2,000 transactions for testing
"""

import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# These control the shape of the dataset. Change them to experiment.
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_SAMPLES = 10_000          # Total number of transactions to generate
FRAUD_RATIO   = 0.02            # 2% of transactions will be fraud
TRAIN_SPLIT   = 0.80            # 80% train, 20% test
RANDOM_SEED   = 42              # Fixed seed so results are reproducible

# The five merchant categories in our dataset
MERCHANT_CATEGORIES = ["grocery", "restaurant", "retail", "online", "travel"]


def generate_legitimate_transactions(n: int, seed: int) -> pd.DataFrame:
    """
    Generate 'n' realistic legitimate (non-fraud) transactions.

    Legitimate transactions have these characteristics:
      - Amounts follow a log-normal distribution (mean ~$33, some larger)
      - Hours are spread evenly across the whole day
      - Mostly everyday merchants: grocery, restaurant, retail

    Args:
        n    : Number of legitimate transactions to generate
        seed : Random seed for reproducibility

    Returns:
        DataFrame with columns: amount, hour, day_of_week,
                                merchant_category, is_fraud
    """
    rng = np.random.default_rng(seed)

    return pd.DataFrame({
        # Log-normal gives us: mostly small amounts, a few larger ones
        # mean=3.5 -> exp(3.5) ≈ $33 average
        "amount": rng.lognormal(mean=3.5, sigma=1.2, size=n).round(2),

        # Transactions happen throughout the day (0 to 23)
        "hour": rng.integers(low=0, high=24, size=n),

        # Spread across all days of the week (0=Monday, 6=Sunday)
        "day_of_week": rng.integers(low=0, high=7, size=n),

        # Weighted toward everyday shopping categories
        # Grocery 30%, Restaurant 25%, Retail 25%, Online 15%, Travel 5%
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES,
            size=n,
            p=[0.30, 0.25, 0.25, 0.15, 0.05]
        ),

        # Label: 0 = legitimate
        "is_fraud": np.zeros(n, dtype=int),
    })


def generate_fraudulent_transactions(n: int, seed: int) -> pd.DataFrame:
    """
    Generate 'n' fraudulent transactions with suspicious patterns.

    Fraudulent transactions have these characteristics:
      - Higher amounts (fraudsters target big purchases)
      - Late-night hours (less oversight)
      - Mostly online and travel (hard to verify)

    Args:
        n    : Number of fraudulent transactions to generate
        seed : Random seed for reproducibility

    Returns:
        DataFrame with columns: amount, hour, day_of_week,
                                merchant_category, is_fraud
    """
    rng = np.random.default_rng(seed + 1)  # Different seed from legitimate

    return pd.DataFrame({
        # Fraudsters tend to charge larger amounts
        # mean=5.5 -> exp(5.5) ≈ $245 average
        "amount": rng.lognormal(mean=5.5, sigma=1.5, size=n).round(2),

        # Fraud happens disproportionately late at night
        "hour": rng.choice([0, 1, 2, 3, 4, 5, 23], size=n),

        # No strong day-of-week pattern for fraud
        "day_of_week": rng.integers(low=0, high=7, size=n),

        # Fraud is concentrated in online and travel (60% + 20%)
        # These are harder to verify in real time
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES,
            size=n,
            p=[0.05, 0.05, 0.10, 0.60, 0.20]
        ),

        # Label: 1 = fraud
        "is_fraud": np.ones(n, dtype=int),
    })


def generate_dataset(
    total: int = TOTAL_SAMPLES,
    fraud_ratio: float = FRAUD_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Combine legitimate and fraudulent transactions into one shuffled dataset.

    Args:
        total       : Total number of transactions
        fraud_ratio : Fraction of transactions that are fraud (e.g. 0.02 = 2%)
        seed        : Random seed for reproducibility

    Returns:
        A shuffled DataFrame with all transactions
    """
    n_fraud = int(total * fraud_ratio)
    n_legit = total - n_fraud

    print(f"  Generating {n_legit:,} legitimate transactions...")
    legit = generate_legitimate_transactions(n_legit, seed)

    print(f"  Generating {n_fraud:,} fraudulent transactions...")
    fraud = generate_fraudulent_transactions(n_fraud, seed)

    # Stack them together and shuffle so fraud is not all at the bottom
    combined = pd.concat([legit, fraud], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


def split_and_save(df: pd.DataFrame, train_ratio: float = TRAIN_SPLIT) -> None:
    """
    Split the dataset into training and test sets, then save to CSV files.

    We use a fixed random_state so the split is identical every time.
    This is critical for reproducibility: the same rows are always in
    the training set, regardless of who runs the script or when.

    Args:
        df          : The full dataset to split
        train_ratio : Fraction to use for training (e.g. 0.8 = 80%)
    """
    os.makedirs("data", exist_ok=True)

    train_df = df.sample(frac=train_ratio, random_state=RANDOM_SEED)
    test_df  = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",  index=False)

    print(f"\n  Saved data/train.csv  ({len(train_df):,} rows)")
    print(f"  Saved data/test.csv   ({len(test_df):,} rows)")


def print_summary(df: pd.DataFrame) -> None:
    """Print key statistics about the generated dataset."""
    fraud    = df[df["is_fraud"] == 1]
    legit    = df[df["is_fraud"] == 0]

    print("\n  ── Dataset Summary ────────────────────────────────────")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraudulent         : {len(fraud):,}  ({len(fraud)/len(df):.1%})")
    print(f"  Legitimate         : {len(legit):,}  ({len(legit)/len(df):.1%})")
    print(f"\n  Average amount (legit) : ${legit['amount'].mean():.2f}")
    print(f"  Average amount (fraud) : ${fraud['amount'].mean():.2f}")
    print(f"\n  Merchant distribution (fraud):")
    dist = fraud["merchant_category"].value_counts(normalize=True)
    for cat, pct in dist.items():
        print(f"    {cat:<12} {pct:.0%}")
    print("  ────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# When you run `python src/generate_data.py`, this block executes.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Generating Synthetic Fraud Detection Dataset ===\n")

    df = generate_dataset()
    print_summary(df)
    split_and_save(df)

    print("\n=== Done! Data is ready in the data/ folder. ===\n")