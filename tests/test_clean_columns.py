import pandas as pd
from bikeprice.clean_columns import clean_columns


def test_central_age_converted_from_months():
    df = pd.DataFrame({
        "city_zone": ["Central", "North", "South"],
        "age_years": [120, 10, 8],
        "brand_tier": ["mid", "premium", "budget"],
    })

    cleaned = clean_columns(df)

    assert cleaned.loc[0, "age_years"] == 10
    assert cleaned.loc[1, "age_years"] == 10
    assert cleaned.loc[2, "age_years"] == 8


def test_brand_tier_variants_normalized():
    df = pd.DataFrame({
        "city_zone": ["North"] * 5,
        "age_years": [5] * 5,
        "brand_tier": [
            " MID ",
            "Premium",
            "BUDGET",
            "bud.",
            "pre.",
        ],
    })

    cleaned = clean_columns(df)

    assert cleaned["brand_tier"].tolist() == [
        "mid",
        "premium",
        "budget",
        "budget",
        "premium",
    ]


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({
        "city_zone": ["Central"],
        "age_years": [120],
        "brand_tier": ["Premium"],
    })

    original = df.copy(deep=True)

    clean_columns(df)

    pd.testing.assert_frame_equal(df, original)