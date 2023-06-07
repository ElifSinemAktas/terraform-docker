import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from urllib.parse import urlparse
import mlflow.sklearn 
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

# Read data
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
print(df.head())

print(df.info())

# Feature matrix
X = df.iloc[:, 3:13]
print(X.shape)
print(X[:3])

# Output variable
y = df.iloc[:, 13]
print(y.shape)
print(y[:6])

# split test train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

experiment_name = "ChurnModelTerraDocker"

if mlflow.get_experiment_by_name(experiment_name):
    pass
else:
    mlflow.set_experiment(experiment_name)

registered_model_name="ChurnXGModel"

client = MlflowClient()
exp_id = client.get_experiment_by_name("ChurnModelTerraDocker")._experiment_id

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy = accuracy_score(y_true=actual, y_pred=pred)
    return rmse, mae, r2, accuracy

with mlflow.start_run(run_name="RandomForest", experiment_id=exp_id) as run:
    n_estimators=50
    pipeline = Pipeline([
        ('ct-ohe', ColumnTransformer([('ct', OneHotEncoder(handle_unknown='ignore', categories='auto'), [1,2])], remainder='passthrough')),
        ('scaler', StandardScaler(with_mean=False)),
        ('estimator', RandomForestClassifier(n_estimators=n_estimators))
    ])
    
    pipeline.fit(X_train, y_train)
    print(X_train[:5])
    print(X_test)
    y_pred = pipeline.predict(X_test)

    (rmse, mae, r2, accuracy) = eval_metrics(y_test, y_pred)

    # mlflow.log_param("")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        mlflow.sklearn.log_model(pipeline, "model")
        # mlflow.sklearn.log_model(estimator, "model",registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(pipeline, "model")
