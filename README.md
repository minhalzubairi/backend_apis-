# Customer Analytics API 🚀

An AI-driven Customer Insights and Engagement Platform that provides:
- Customer Segmentation
- Churn Prediction  
- Product Recommendations
- Sentiment Analysis

## 🔗 Live API
**Deployed on:** `https://your-app-name.ondigitalocean.app`

## 📋 API Endpoints

### 1. Customer Churn Prediction
```bash
POST /predict-churn
Content-Type: application/json

{
  "recency": 10,
  "frequency": 5, 
  "monetary": 100,
  "avg_payment_value": 50,
  "avg_review_score": 4.0
}
