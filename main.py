# ──────────────────────────────────────────────────────────────────────────────
# main.py — News Category Classifier API
# Author: Sameer Singh
#
# This FastAPI application serves the fine-tuned DistilBERT model as a REST API.
# It accepts a news headline/text via HTTP POST and returns the predicted category.
#
# Architecture flow:
# HTTP Request (JSON) → FastAPI → Tokenizer → DistilBERT → Softmax → JSON Response
# ──────────────────────────────────────────────────────────────────────────────

# FastAPI: the web framework that turns Python functions into API endpoints
# It automatically generates interactive docs at /docs (Swagger UI)
from fastapi import FastAPI, HTTPException

# BaseModel: Pydantic class for defining the shape of request/response data
# Pydantic automatically validates incoming data — if someone sends an integer
# instead of a string, FastAPI rejects it before our code even runs
from pydantic import BaseModel

# numpy: for calculating probabilities from logits
import numpy as np

# AutoTokenizer: loads the same tokenizer used during training
# Critical: must use the exact same tokenizer, or token IDs won't match
from transformers import AutoTokenizer

# ORTModelForSequenceClassification: loads ONNX model for sequence classification
from optimum.onnxruntime import ORTModelForSequenceClassification
import os
import logging

# Configure logging so we can see what's happening when the server runs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Using ONNX runtime for reduced memory footprint instead of PyTorch

# ── App initialization ────────────────────────────────────────────────────────

# Create the FastAPI application instance
# title and description appear in the auto-generated /docs page
app = FastAPI(
    title="News Category Classifier",
    description="""
    ## Fine-tuned DistilBERT for News Classification
    
    Classifies news headlines/articles into 4 categories:
    - **World** — international news, politics, diplomacy
    - **Sports** — sports events, results, athlete news  
    - **Business** — markets, economy, corporate news
    - **Science/Tech** — technology, science, innovation
    
    **Model:** DistilBERT fine-tuned on AG News dataset (20,000 samples)
    
    **Author:** Sameer Singh
    """,
    version="1.0.0"
)

# ── Model loading ─────────────────────────────────────────────────────────────
#
# CRITICAL: We load the model ONCE when the server starts, not on every request.
#
# Why? Loading a model takes 2-3 seconds and uses significant memory.
# If we loaded it per request, every API call would be 2-3 seconds slow
# and memory would spike then crash under multiple concurrent requests.
#
# By loading once at startup, the model stays in memory and each request
# just does inference (milliseconds) rather than loading (seconds).
#
# This is a fundamental pattern for serving ML models in production.

# Path to the local ONNX model folder
MODEL_PATH = "onnx_model/"

logger.info(f"Loading ONNX model from {MODEL_PATH}...")

# Load tokenizer from the saved path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the ONNX model
# low_cpu_mem_usage=True reduces memory spikes during loading
model = ORTModelForSequenceClassification.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)

logger.info(f"ONNX Model loaded successfully")

# Label mapping — loaded directly from model config
id2label = model.config.id2label

# ── Request/Response models ───────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Defines the shape of data this API expects to receive.
    
    Pydantic validates this automatically — if 'text' is missing or
    not a string, FastAPI returns a 422 error before our code runs.
    """
    text: str  # the news headline or article text to classify
    
    class Config:
        # Example shown in the /docs Swagger UI
        json_schema_extra = {
            "example": {
                "text": "NASA discovers new Earth-like exoplanet with potential for liquid water"
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the shape of data this API sends back.
    
    Documenting the response shape helps API consumers know what to expect.
    """
    text: str           # the original input text (echoed back for confirmation)
    predicted_class: str    # the predicted news category (e.g. "Science/Tech")
    confidence: float       # probability of the predicted class (0.0 to 1.0)
    all_scores: dict        # probability distribution across all 4 classes

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """
    Root endpoint — confirms the API is running.
    
    GET /  →  returns status and model info
    
    Used by deployment platforms (Render) to check if the service is alive.
    This is a GET request because we're just reading status, not sending data.
    """
    return {
        "status": "online",
        "model": "DistilBERT News Classifier",
        "author": "Sameer Singh",
        "categories": list(id2label.values()),
        "docs": "/docs"  # remind users where to find the interactive docs
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint — classifies a news text into a category.
    
    POST /predict  →  accepts JSON with 'text', returns prediction
    
    This is a POST request (not GET) because:
    - We're sending data (the news text) to be processed
    - GET requests should only retrieve data, not submit it for processing
    - POST allows us to send structured JSON in the request body
    
    Args:
        request: PredictionRequest containing the text to classify
        
    Returns:
        PredictionResponse with predicted class, confidence, and all scores
    """
    
    # Validate input — don't process empty strings
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    # Tokenize the input text
    # return_tensors='np' returns numpy arrays for ONNX Runtime
    # max_length=64 for faster inference and lower memory usage
    inputs = tokenizer(
        request.text,
        return_tensors='np',
        truncation=True,
        padding=True,
        max_length=64
    )
    
    # Filter out 'token_type_ids' — DistilBERT does not use it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    # Run ONNX inference
    outputs = model(**inputs)
    
    # Convert logits to probabilities using softmax with numpy
    logits = outputs.logits
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Get the first (and only) batch item
    probs = probabilities[0]
    
    # Find the class with highest probability
    # argmax returns the index of the maximum value
    predicted_class_id = probs.argmax()
    
    # Convert index to human-readable label using our mapping
    predicted_class_name = id2label[int(predicted_class_id)]
    
    # Build the full scores dict — show probability for every class
    # This helps the user understand how confident the model was
    # and whether it was close between two categories
    all_scores = {
        id2label[i]: round(float(probs[i]), 4)
        for i in range(len(id2label))
    }
    
    return PredictionResponse(
        text=request.text,
        predicted_class=predicted_class_name,
        confidence=round(float(probs[predicted_class_id]), 4),
        all_scores=all_scores
    )

@app.post("/predict/batch")
def predict_batch(texts: list[str]):
    """
    Batch prediction endpoint — classify multiple texts in one request.
    
    More efficient than calling /predict multiple times because:
    - Single tokenization call for all texts
    - Single forward pass through the model (parallelized on GPU)
    - One network round trip instead of N round trips
    
    Args:
        texts: list of news texts to classify
        
    Returns:
        list of predictions, one per input text
    """
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(texts) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 texts per batch")
    
    # Tokenize all texts at once
    # return_tensors='np' returns numpy arrays for ONNX Runtime
    inputs = tokenizer(
        texts,
        return_tensors='np',
        truncation=True,
        padding=True,
        max_length=64
    )
    # Filter out 'token_type_ids' — DistilBERT does not accept it
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    outputs = model(**inputs)
    
    # Compute softmax probabilities
    logits = outputs.logits
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    results = []
    for i, text in enumerate(texts):
        probs = probabilities[i]
        predicted_id = probs.argmax()
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_class": id2label[int(predicted_id)],
            "confidence": round(float(probs[predicted_id]), 4)
        })
    
    return {"predictions": results, "count": len(results)}
