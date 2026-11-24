
def check_duplicates(df):
    duplicate_count = int(df.duplicated().sum())

    return {
        "present": duplicate_count > 0,
        "count": duplicate_count
    }
