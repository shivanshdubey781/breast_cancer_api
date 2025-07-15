from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load models
log_reg = joblib.load("logistic_regression_model.pkl")
rf = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
svm = joblib.load("svm_model.pkl")
mlp = joblib.load("mlp_model.pkl")
voting = joblib.load("voting_classifier_model.pkl")

# Define input schema
class InputData(BaseModel):
    features: list  # Expecting 30 float values

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running."}

# Predict route
@app.post("/predict")
def predict(input: InputData):
    data = np.array(input.features).reshape(1, -1)

    predictions = {
        "LogisticRegression": int(log_reg.predict(data)[0]),
        "RandomForest": int(rf.predict(data)[0]),
        "XGBoost": int(xgb_model.predict(data)[0]),
        "SVM": int(svm.predict(data)[0]),
        "MLPClassifier": int(mlp.predict(data)[0]),
        "VotingClassifier": int(voting.predict(data)[0])
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://5658f4a7-b04f-4673-9411-4f1892992251.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    return {"predictions": predictions}
