import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(train: pd.DataFrame):
    X = train.drop(columns="label")
    y = train["label"]

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X, y)

    return model

def train_forest(train: pd.DataFrame):
    X = train.drop(columns="label")
    y = train["label"]

    model = RandomForestClassifier(n_estimators=300,
                                   max_depth=3,
                                   min_samples_leaf=20,
                                   random_state=42)
    model.fit(X, y)
    return model

def evaluate(model, data: pd.DataFrame):
    X = data.drop(columns="label")
    y = data["label"]

    predictions = model.predict(X)
    
    accuracy = (predictions == y).mean()
    return accuracy, predictions.mean()
