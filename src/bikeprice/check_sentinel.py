import pandas as pd

def check_sentinel(df, sentinel_values=None):
    if sentinel_values is None:
        sentinel_values = [-1, -999, 999, 9999]
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        for value in sentinel_values:
            count = int((df[col] == value).sum())

            if count > 0:
                return {
                    "present": True,
                    "column": col,
                    "value": value,
                    "count": count
                }
    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None
    }

if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")
    print(check_sentinel(df))
