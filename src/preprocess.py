"""
Preprocessing module for Google Play banking app reviews.
Cleans raw scraped reviews and generates a data quality report.
"""

import pandas as pd


def preprocess_reviews(df):
    """
    Preprocess scraped reviews DataFrame.

    Steps:
    - Drop duplicates based on 'content' and 'bank'.
    - Drop rows with missing or empty 'content'.
    - Convert 'at' column to datetime, drop invalid dates.
    - Rename columns: 'content' -> 'review', 'score' -> 'rating', 'at' -> 'date'.
    - Ensure 'rating' is integer.
    - Print a data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw reviews DataFrame.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned DataFrame with standardized columns.
    metrics : dict
        Dictionary of data quality metrics.
    """
    # Initialize metrics
    metrics = {}
    initial_count = len(df)
    metrics["initial_count"] = initial_count

    # Drop duplicates
    df = df.drop_duplicates(subset=["content", "bank"], keep="first")
    metrics["duplicates_removed"] = initial_count - len(df)

    # Drop missing or empty content
    before_missing = len(df)
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]
    metrics["missing_text_removed"] = before_missing - len(df)

    # Convert 'at' to datetime and drop invalid
    before_dates = len(df)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["at"])
    metrics["invalid_dates_removed"] = before_dates - len(df)

    # Rename columns
    df = df.rename(
        columns={
            "content": "review",
            "score": "rating",
            "at": "date"
        }
    )

    # Ensure rating is integer
    df["rating"] = df["rating"].astype(int)

    # Final count and retention
    final_count = len(df)
    metrics["final_count"] = final_count
    metrics["retention_rate"] = (
        (final_count / initial_count * 100) if initial_count > 0 else 0
    )

    # Print report
    print("=== Data Quality Report ===")
    print(f"Initial count: {metrics['initial_count']}")
    print(f"Duplicates removed: {metrics['duplicates_removed']}")
    print(f"Missing text removed: {metrics['missing_text_removed']}")
    print(f"Invalid dates removed: {metrics['invalid_dates_removed']}")
    print(f"Final count: {metrics['final_count']}")
    print(f"Retention rate: {metrics['retention_rate']:.2f}%")

    return df, metrics
