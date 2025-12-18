import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "magnitude", "cdi", "mmi", "sig", "nst",
    "dmin", "gap", "depth", "latitude",
    "longitude", "Year", "Month"
]

TARGET = "tsunami"

def load_data(path):
    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
