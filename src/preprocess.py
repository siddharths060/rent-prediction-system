import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame):
    # drops empty cells
    df = df.dropna().copy()
    
    if "area_rate" in df.columns:
        df = df.drop("area_rate", axis = 1)

        # check if rent attribute is present in dataset
        if "rent" not in df.columns:
            raise ValueError("Target column 'rent' not found")
        
        X = df["rent"]
        Y = df.drop("rent", axis = 1)

        categorical_cols = ["house_type", "locality", "city", "furnishing"]

        encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        return X,Y, encoders
    
def transform_input(data: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                raise ValueError(
                    f"Value '{df[col].iloc[0]}' not seen in training for column '{col}'"
                )
    return df
