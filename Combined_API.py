from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import os

# --------------------------------------------------
# Initialize app
# --------------------------------------------------
app = Flask(__name__)

# Enable CORS for production
CORS(app, origins=['*'])  # You can restrict this to your frontend domain later

# Production configuration
if os.environ.get('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
else:
    app.config['DEBUG'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load Models with Error Handling
# --------------------------------------------------
try:
    churn_model, churn_scaler = joblib.load(
        os.path.join(BASE_DIR, "churn_model.pkl")
    )
    print("✅ Churn model loaded successfully")
except Exception as e:
    print(f"❌ Error loading churn model: {e}")
    churn_model, churn_scaler = None, None

try:
    rules_product = joblib.load(os.path.join(BASE_DIR, "rules_product.pkl"))
    rules_aisle = joblib.load(os.path.join(BASE_DIR, "rules_aisle.pkl"))
    rules_department = joblib.load(os.path.join(BASE_DIR, "rules_department.pkl"))
    print("✅ Association rules loaded successfully")
except Exception as e:
    print(f"❌ Error loading association rules: {e}")
    rules_product = rules_aisle = rules_department = None

try:
    sentiment_model, sentiment_vectorizer = joblib.load(
        os.path.join(BASE_DIR, "sentiment_model.pkl")
    )
    print("✅ Sentiment model loaded successfully")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")
    sentiment_model, sentiment_vectorizer = None, None

try:
    segmentation_model = joblib.load(
        os.path.join(BASE_DIR, "customer_segmentation_pipeline.pkl")
    )
    print("✅ Segmentation model loaded successfully")
except Exception as e:
    print(f"❌ Error loading segmentation model: {e}")
    segmentation_model = None

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def validate_fields(data, required_fields):
    """Check for missing required fields in request JSON."""
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, {"error": f"Missing fields: {missing}"}
    return True, None

# --------------------------------------------------
# 1) Churn Prediction
# --------------------------------------------------
CHURN_FEATURES = ["recency", "frequency", "monetary", "avg_payment_value", "avg_review_score"]

@app.route("/predict-churn", methods=["POST"])
def predict_churn():
    if churn_model is None or churn_scaler is None:
        return jsonify({"error": "Churn model not available"}), 503
    
    try:
        data = request.get_json()
        valid, error = validate_fields(data, CHURN_FEATURES)
        if not valid:
            return jsonify(error), 400

        input_array = np.array([data[f] for f in CHURN_FEATURES]).reshape(1, -1)
        input_scaled = churn_scaler.transform(input_array)

        prediction = churn_model.predict(input_scaled)[0]
        probability = churn_model.predict_proba(input_scaled)[0, 1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 2) Product Mining Recommendation
# --------------------------------------------------
def recommend(cart_items, rules, top_n=5):
    if rules is None:
        return []
    
    cart_items = set(cart_items)
    recs = []
    for _, row in rules.iterrows():
        if row["antecedents"].issubset(cart_items):
            for consequent in row["consequents"]:
                if consequent not in cart_items:
                    reason = (f"Because you bought {', '.join(row['antecedents'])}, "
                              f"customers also often buy {consequent}")
                    recs.append({
                        "item": consequent,
                        "reason": reason,
                        "confidence": float(row["confidence"]),
                        "lift": float(row["lift"]),
                    })
    recs = sorted(recs, key=lambda x: (x["confidence"], x["lift"]), reverse=True)

    seen, final = set(), []
    for r in recs:
        if r["item"] not in seen:
            final.append(r)
            seen.add(r["item"])
        if len(final) >= top_n:
            break
    return final

@app.route("/recommend", methods=["POST"])
def recommend_all():
    try:
        data = request.json
        cart_items = data.get("cart", [])
        result = {
            "product_recommendations": recommend(cart_items, rules_product),
            "aisle_recommendations": recommend(cart_items, rules_aisle),
            "department_recommendations": recommend(cart_items, rules_department),
        }

        print("Request cart_items:", cart_items)
        print("Response:", result)

        return jsonify(result)
    except Exception as e:
        print("Error in /recommend:", e)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 3) Sentiment Analysis
# --------------------------------------------------
@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    if sentiment_model is None or sentiment_vectorizer is None:
        return jsonify({"error": "Sentiment model not available"}), 503
        
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        X_input = sentiment_vectorizer.transform([cleaned])
        sentiment = sentiment_model.predict(X_input)[0]

        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 4) Customer Segmentation
# --------------------------------------------------
cluster_labels = {
    0: "High Spenders",
    1: "Budget-Conscious",
    2: "Trend Seekers",
    3: "Loyal Mid-Lifers",
}

SEGMENT_FEATURES = [
    "AgeGroup", "Education_Encoded", "Marital_Status",
    "Income", "Has_Children",
    "Purchases", "Spending",
    "Recency", "Response",
]

SEGMENT_DEFAULTS = {
    "AgeGroup": 1, "Education_Encoded": 0, "Marital_Status": 0,
    "Income": 0, "Has_Children": 0,
    "Purchases": 0, "Spending": 0,
    "Recency": 0, "Response": 0,
}

@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    if segmentation_model is None:
        return jsonify({"error": "Segmentation model not available"}), 503
        
    try:
        data_json = request.get_json()
        data = pd.DataFrame(data_json)

        for col in SEGMENT_FEATURES:
            if col not in data.columns:
                data[col] = SEGMENT_DEFAULTS[col]
            data[col] = data[col].fillna(SEGMENT_DEFAULTS[col])

        preds = segmentation_model.predict(data[SEGMENT_FEATURES])
        data["Predicted_Cluster"] = preds
        data["Cluster_Label"] = data["Predicted_Cluster"].map(cluster_labels)

        return jsonify(data.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Health Check and Root
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    models_status = {
        "churn_model": churn_model is not None,
        "sentiment_model": sentiment_model is not None,
        "segmentation_model": segmentation_model is not None,
        "rules_available": all([rules_product is not None, rules_aisle is not None, rules_department is not None])
    }
    
    return jsonify({
        "status": "healthy",
        "models": models_status,
        "message": "Customer Analytics API is running 🚀"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚀 Customer Analytics API v1.0",
        "status": "running",
        "endpoints": {
            "churn_prediction": "/predict-churn",
            "recommendations": "/recommend", 
            "sentiment_analysis": "/sentiment",
            "customer_segmentation": "/predict-segment",
            "health_check": "/health"
        },
        "usage": {
            "churn": "POST to /predict-churn with recency, frequency, monetary, avg_payment_value, avg_review_score",
            "recommend": "POST to /recommend with cart array",
            "sentiment": "POST to /sentiment with text field",
            "segment": "POST to /predict-segment with customer features"
        }
    })

# --------------------------------------------------
# Error Handlers
# --------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config['DEBUG'])
