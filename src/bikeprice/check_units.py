
import pandas as pd


def check_units(df, group_col="city_zone", threshold=9):
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [
        col for col in numeric_cols
        if col != "resale_thousand_rs"
    ]

    for col in numeric_cols:
        medians = df.groupby(group_col)[col].median()
        medians = medians[medians > 0]

        if len(medians) < 2:
            continue

        largest_group = medians.idxmax()
        largest_value = medians.max()
        other_values = medians.drop(largest_group)

        comparison_median = other_values.median()
        ratio = largest_value / comparison_median

        if ratio >= threshold:
            factor = round(ratio)

            return {
                "present": True,
                "column": col,
                "group_value": largest_group,
                "factor": factor
            }

    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")
    print(check_units(df))
