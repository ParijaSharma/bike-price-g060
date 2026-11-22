"""
Data quality and anomaly inspection module.
Implements detection logic for the seven kinds of data problems described in STUDENT_HANDBOOK.md.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def detect_leaking_columns(train_path: str = "data/raw/train.csv", 
                           holdout_path: str = "data/raw/holdout_features.csv",
                           target_col: str = "resale_thousand_rs") -> dict:
    """
    Identifies features present in train.csv but missing in holdout_features.csv,
    and checks their correlation with the target variable.
    """
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    extra_cols = set(train_df.columns) - set(holdout_df.columns) - {target_col}
    leaks = {}
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            corr = train_df[col].corr(train_df[target_col])
            leaks[col] = float(corr)
        else:
            leaks[col] = None

    return {
        "present": len(leaks) > 0,
        "leaking_columns": leaks
    }


def check_all_data_problems(train_path: str = "data/raw/train.csv",
                            holdout_path: str = "data/raw/holdout_features.csv",
                            later_path: str = "data/raw/later.csv",
                            target_col: str = "resale_thousand_rs") -> dict:
    """Runs checks across the 7 types of data problems."""
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    later_df = pd.read_csv(later_path) if Path(later_path).exists() else None

    results = {}

    # 1. Leaking column
    results["leaking_column"] = detect_leaking_columns(train_path, holdout_path, target_col)

    # 2. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    results["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count
    }

    # 3. Sentinel values (check common negative or sentinel numbers like -1, -999, 999, 9999)
    sentinels = [-1, -999, 999, 9999]
    sentinel_found = {}
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        for s in sentinels:
            cnt = int((train_df[col] == s).sum())
            if cnt > 0:
                sentinel_found[col] = {"value": s, "count": cnt}
    results["sentinel_value"] = {
        "present": len(sentinel_found) > 0,
        "details": sentinel_found
    }

    # 4. Target typos (values ~10x larger than normal range)
    # Target distribution check
    q75 = train_df[target_col].quantile(0.75)
    iqr = q75 - train_df[target_col].quantile(0.25)
    far_outliers = int((train_df[target_col] > (q75 + 5 * iqr)).sum())
    results["target_typos"] = {
        "present": far_outliers > 0,
        "count": far_outliers
    }

    # 5. Unit mixup across group (city_zone)
    unit_mixups = {}
    if "city_zone" in train_df.columns:
        numeric_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != target_col]
        for col in numeric_cols:
            medians = train_df.groupby("city_zone")[col].median()
            if len(medians) > 1 and medians.min() > 0:
                ratio = medians.max() / medians.min()
                if ratio >= 9.0:  # factor ~10, 12, 100, 1000
                    unit_mixups[col] = {
                        "medians": medians.to_dict(),
                        "ratio": float(ratio)
                    }
    results["unit_mixup"] = {
        "present": len(unit_mixups) > 0,
        "details": unit_mixups
    }

    # 6. Stuck sensor (long sequence of identical consecutive values across dates)
    stuck_detected = {}
    sorted_df = train_df.sort_values("date")
    for col in sorted_df.columns:
        if col in ["record_id", "date", target_col]:
            continue
        max_run = 0
        current_run = 0
        prev_val = None
        start_date = None
        max_start = None
        max_end = None
        for _, row in sorted_df.iterrows():
            val = row[col]
            dt = row["date"]
            if val == prev_val:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    max_start = start_date
                    max_end = dt
            else:
                current_run = 1
                prev_val = val
                start_date = dt
        if max_run > 15:  # stuck sensor for weeks
            stuck_detected[col] = {
                "max_consecutive_days": max_run,
                "start_date": str(max_start),
                "end_date": str(max_end)
            }
    results["stuck_sensor"] = {
        "present": len(stuck_detected) > 0,
        "details": stuck_detected
    }

    # 7. Spelling variants in categorical columns
    spelling_issues = {}
    text_cols = train_df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        if col in ["record_id", "date"]:
            continue
        train_cats = set(train_df[col].dropna().unique())
        later_cats = set(later_df[col].dropna().unique()) if later_df is not None else set()
        if len(train_cats) > len(later_cats) and len(later_cats) > 0:
            spelling_issues[col] = {
                "train_categories": list(train_cats),
                "clean_later_categories": list(later_cats),
                "true_levels": len(later_cats)
            }
    results["spelling_variants"] = {
        "present": len(spelling_issues) > 0,
        "details": spelling_issues
    }

    return results


if __name__ == "__main__":
    import pprint
    print("=== Leaking Column Check ===")
    leak_res = detect_leaking_columns()
    pprint.pprint(leak_res)

    print("\n=== Summary of All Data Checks ===")
    all_checks = check_all_data_problems()
    pprint.pprint(all_checks)
