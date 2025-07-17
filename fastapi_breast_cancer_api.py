from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# CORS settings for cross-platform requests (important for iOS frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models
log_reg = joblib.load("logistic_regression_model.pkl")
rf = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
svm = joblib.load("svm_model.pkl")
mlp = joblib.load("mlp_model.pkl")
voting = joblib.load("voting_classifier_model.pkl")

# Define input schema
class InputData(BaseModel):
    features: list[float]  # Must be a list of 30 floats

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running."}

# Predict route
@app.post("/predict")
def predict(input: InputData):
    try:
        data = np.array(input.features).reshape(1, -1)
        predictions = {
            "LogisticRegression": int(log_reg.predict(data)[0]),
            "RandomForest": int(rf.predict(data)[0]),
            "XGBoost": int(xgb_model.predict(data)[0]),
            "SVM": int(svm.predict(data)[0]),
            "MLPClassifier": int(mlp.predict(data)[0]),
            "VotingClassifier": int(voting.predict(data)[0])
        }
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
