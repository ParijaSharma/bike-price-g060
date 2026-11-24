import pandas as pd
from bikeprice.clean_rows import (
    clean_rows, count_duplicates, count_target_typos,
    drop_leaking_column, fix_target_typos, remove_duplicates,
)


def make_df():
    return pd.DataFrame({
        "record_id": [f"R{i}" for i in range(10)],
        "age_years": [3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
        "final_price_from_rc_transfer": [80, 82, 78, 85, 90, 81, 79, 83, 84, 86],
        "resale_thousand_rs": [80, 82, 78, 85, 90, 81, 79, 83, 84, 86],
    })


def test_drop_leaking_column():
    out = drop_leaking_column(make_df())
    assert "final_price_from_rc_transfer" not in out.columns
    assert "resale_thousand_rs" in out.columns


def test_remove_duplicates():
    df = make_df()
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    assert count_duplicates(df) == 2
    assert len(remove_duplicates(df)) == 10


def test_fix_target_typos():
    df = make_df()
    df.loc[2, "resale_thousand_rs"] = 780.0
    assert count_target_typos(df) == 1
    fixed = fix_target_typos(df)
    assert fixed.loc[2, "resale_thousand_rs"] == 78.0
    assert fixed.loc[0, "resale_thousand_rs"] == 80  # untouched


def test_clean_rows_end_to_end():
    df = make_df()
    df.loc[4, "resale_thousand_rs"] = 900.0
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    out = clean_rows(df)
    assert len(out) == 10
    assert out["resale_thousand_rs"].max() < 100
    assert "final_price_from_rc_transfer" not in out.columns