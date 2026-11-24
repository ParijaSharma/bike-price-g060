"""Row-level cleaning, part A: leaking column, duplicate rows, target typos."""
import pandas as pd

LEAK_COLUMN = "final_price_from_rc_transfer"
TARGET = "resale_thousand_rs"


def drop_leaking_column(df: pd.DataFrame, column: str = LEAK_COLUMN) -> pd.DataFrame:
    """Remove a column that is only known after the sale."""
    return df.drop(columns=[column], errors="ignore")


def count_duplicates(df: pd.DataFrame) -> int:
    """Number of EXTRA copies (rows that would be deleted)."""
    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def typo_mask(df: pd.DataFrame, target: str = TARGET, iqr_k: float = 3.0) -> pd.Series:
    """True for targets far above the rest (typed 10x too big)."""
    q1, q3 = df[target].quantile([0.25, 0.75])
    upper = q3 + iqr_k * (q3 - q1)
    return df[target] > upper


def count_target_typos(df: pd.DataFrame, target: str = TARGET) -> int:
    return int(typo_mask(df, target).sum())


def fix_target_typos(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    out = df.copy()
    mask = typo_mask(out, target)
    out.loc[mask, target] = out.loc[mask, target] / 10
    return out


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Part A pipeline: dedupe, fix typos, drop the leaking column."""
    df = remove_duplicates(df)
    df = fix_target_typos(df)
    df = drop_leaking_column(df)
    return df