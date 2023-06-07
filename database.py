import os
from dotenv import load_dotenv
from sqlmodel import create_engine, SQLModel, Session
import boto3
import pickle

load_dotenv()  # take environment variables from .env.
SQLALCHEMY_DATABASE_URL = os.getenv('SQLALCHEMY_DATABASE_URL')
# print(SQLALCHEMY_DATABASE_URL)

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

# MINIO_KEY=os.getenv('MINIO_KEY')
# MINIO_SECRET=os.getenv('MINIO_SECRET')
# s3_cli = boto3.client('s3')
# s3_res = boto3.resource('s3')

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_db():
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()


# def read_encoder(bucket,key, s3_res=s3_res):
#     response = s3_cli.get_object(Bucket=bucket, Key=key)
#     body = response['Body'].read()
#     data = pickle.loads(body)
#     return data


# def write_encoder(bucket, key, model, s3_res=s3_res):
#     pickle_byte_obj = pickle.dumps(model)
#     s3_res.Object(bucket,key).put(Body=pickle_byte_obj)