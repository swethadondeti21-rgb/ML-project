"""
FastAPI server for serving fraud detection predictions.

This API is the public-facing component of our ML system.
It accepts transaction data and returns a fraud prediction.

How to run locally:
-------------------
    uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000

Then open: http://localhost:8000/docs  (interactive documentation)

Endpoints:
----------
    GET  /health     -- Returns server status (used by CI/CD smoke test)
    POST /predict    -- Accepts transaction data, returns fraud prediction
"""

import pickle
from fastapi  import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# The model is loaded once when the server starts, not on every request.
# This is important for performance — loading a pickle file takes ~100ms.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/model.pkl"

print(f"Loading model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, "rb") as f:
        model, encoder = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run 'python src/train.py' first."
    )

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Predict whether a credit card transaction is fraudulent.",
    version="1.0.0",
)

# Valid merchant categories (must match training data)
VALID_CATEGORIES = {"grocery", "restaurant", "retail", "online", "travel"}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS (Pydantic)
# Pydantic validates incoming JSON automatically.
# If a field is wrong type or missing, FastAPI returns a 422 error.
# ─────────────────────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    """Input schema: the data needed to evaluate one transaction."""
    amount:            float = Field(..., gt=0,       description="Transaction amount in USD (must be > 0)", example=150.00)
    hour:              int   = Field(..., ge=0, le=23, description="Hour of the day (0–23)",                 example=14)
    day_of_week:       int   = Field(..., ge=0, le=6,  description="Day of week (0=Mon, 6=Sun)",             example=3)
    merchant_category: str   = Field(...,              description="One of: grocery, restaurant, retail, online, travel", example="online")


class PredictionResponse(BaseModel):
    """Output schema: the fraud prediction result."""
    is_fraud:          bool  = Field(description="True if predicted as fraud")
    fraud_probability: float = Field(description="Probability of fraud, 0.0–1.0")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Used by:
      - CI/CD smoke tests to verify the container started correctly
      - Docker HEALTHCHECK instruction
      - Load balancers in production
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.

    Validates input, encodes features, and returns the prediction.
    Returns HTTP 400 if merchant_category is not recognized.
    """
    data = transaction.model_dump()

    # Additional validation: check the merchant category is known
    if data["merchant_category"] not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown merchant_category: '{data['merchant_category']}'. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}"
            ),
        )

    # Encode the merchant category using the saved encoder
    data["merchant_encoded"] = int(
        encoder.transform([data["merchant_category"]])[0]
    )

    # Build the feature vector in the correct order
    X = [[
        data["amount"],
        data["hour"],
        data["day_of_week"],
        data["merchant_encoded"],
    ]]

    prediction   = model.predict(X)[0]
    probability  = model.predict_proba(X)[0][1]

    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4),
    )