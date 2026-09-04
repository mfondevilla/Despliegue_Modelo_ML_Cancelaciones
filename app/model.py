import pickle

MODEL_PATH = "model/xgboost_final.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)