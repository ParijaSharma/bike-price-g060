def check_target_typos(df, target_col="resale_thousand_rs"):
    target = df[target_col].dropna().sort_values()

    median_value = target.median()
    threshold = median_value * 10

    suspicious_values = target[target > threshold]
    typo_count = int(suspicious_values.count())

    return {
        "present": typo_count > 0,
        "count": typo_count
    }