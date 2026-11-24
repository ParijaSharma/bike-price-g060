import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column-level data quality issues."""

    df = df.copy()

    # Correct age unit mixup in Central (months -> years)
    central_mask = df["city_zone"] == "Central"
    df.loc[central_mask, "age_years"] = (
        df.loc[central_mask, "age_years"] / 12
    )

    # Normalize brand tier spelling variants
    if "brand_tier" in df.columns:
        df["brand_tier"] = (
            df["brand_tier"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({
                "bud.": "budget",
                "pre.": "premium",
                "mid.": "mid",
            })
        )

    return df