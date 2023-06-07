from fastapi import FastAPI, Depends, Request
import os
from sqlmodels import  Churn, CreateUpdateChurn
from database import engine, get_db, create_db_and_tables
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
import joblib
import numpy as np
import pandas as pd

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000/'

# Learn, decide and get model from mlflow model registry
model_name = "ChurnRandomForestModel"
model_version = 1
model = load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

app = FastAPI()

# Creates all the tables defined in models module
create_db_and_tables()

def insert_churn(request, prediction, client_ip, db):
    new_churn = Churn(
        CreditScore=request["CreditScore"],
        Geography=request["Geography"],
        Gender=request['Gender'],
        Age=request['Age'],
        Tenure=request['Tenure'],
        Balance=request['Balance'],
        NumOfProducts=request['NumOfProducts'],
        HasCrCard=request['HasCrCard'],
        IsActiveMember=request['IsActiveMember'],
        EstimatedSalary=request['EstimatedSalary'],
        prediction=prediction,
        client_ip=client_ip
    )

    with db as session:
        session.add(new_churn)
        session.commit()
        session.refresh(new_churn)

    return new_churn


# prediction function
def make_churn_prediction(model, request):
    # parse input from request
    CreditScore=request["CreditScore"],
    Geography=request["Geography"],
    Gender=request["Gender"],
    Age=request['Age'],
    Tenure=request['Tenure'],
    Balance=request['Balance'],
    NumOfProducts=request['NumOfProducts'],
    HasCrCard=request['HasCrCard'],
    IsActiveMember=request['IsActiveMember'],
    EstimatedSalary=request['EstimatedSalary'],

    print(CreditScore)

    # Make an input vector
    info = [[CreditScore[0], Geography[0], Gender[0], Age[0], Tenure[0], 
             Balance[0], NumOfProducts[0], HasCrCard[0], IsActiveMember[0], EstimatedSalary[0]]]
    
    print(info)

    df = pd.DataFrame(data=info , columns = model.feature_names_in_)
    # Predict
    prediction = model.predict(df)
    # prediction = model.predict(np.array(info, dtype=float))

    return prediction[0].item()



# Churn Prediction endpoint
@app.post("/prediction/churn")
def predict_churn(request: CreateUpdateChurn, fastapi_req: Request,  db: Session = Depends(get_db)):
    prediction = make_churn_prediction(model=model, request=request.dict())
    db_insert_record = insert_churn(request=request.dict(), prediction=prediction,
                                          client_ip=fastapi_req.client.host,
                                          db=db)
    return {"prediction": prediction, "db_record": db_insert_record}

# Welcome page
@app.get("/")
async def root():
    return {"data":"Welcome to MLOps API"}