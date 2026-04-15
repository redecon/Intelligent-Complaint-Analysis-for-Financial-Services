"""
Unit tests for src/preprocess.py
"""

import pytest
import pandas as pd
from src.preprocess import preprocess_reviews


@pytest.fixture
def sample_df():
    """Return a synthetic DataFrame with duplicates, missing text, and invalid dates."""
    return pd.DataFrame([
        {"bank": "CBE", "content": "Great app!", "score": 5, "at": "2024-01-01", "source": "Google Play", "scrape_date": "2024-01-10"},
        {"bank": "CBE", "content": "Great app!", "score": 5, "at": "2024-01-01", "source": "Google Play", "scrape_date": "2024-01-10"},  # duplicate
        {"bank": "BOA", "content": None, "score": 4, "at": "2024-01-02", "source": "Google Play", "scrape_date": "2024-01-10"},  # missing text
        {"bank": "Dashen", "content": " ", "score": 3, "at": "2024-01-03", "source": "Google Play", "scrape_date": "2024-01-10"},  # empty string
        {"bank": "CBE", "content": "Bad experience", "score": 1, "at": "invalid-date", "source": "Google Play", "scrape_date": "2024-01-10"},  # invalid date
        {"bank": "BOA", "content": "Works fine", "score": 4, "at": "2024-01-04", "source": "Google Play", "scrape_date": "2024-01-10"},
    ])


def test_preprocess_reviews(sample_df):
    """Test preprocessing removes duplicates, missing text, and invalid dates."""
    cleaned_df, metrics = preprocess_reviews(sample_df)

    # Check columns renamed correctly
    expected_cols = ["bank", "review", "rating", "date", "source", "scrape_date"]
    assert list(cleaned_df.columns) == expected_cols

    # Check ratings are integers
    assert pd.api.types.is_integer_dtype(cleaned_df["rating"])

    # Check metrics dictionary has expected keys
    expected_keys = [
        "initial_count",
        "duplicates_removed",
        "missing_text_removed",
        "invalid_dates_removed",
        "final_count",
        "retention_rate",
    ]
    for key in expected_keys:
        assert key in metrics

    # Validate counts
    assert metrics["initial_count"] == 6
    assert metrics["duplicates_removed"] == 1
    assert metrics["missing_text_removed"] == 2
    assert metrics["invalid_dates_removed"] == 1
    assert metrics["final_count"] == 2
    assert metrics["retention_rate"] == pytest.approx(33.33, rel=0.01)

    # Check final DataFrame content
    assert all(cleaned_df["review"].notna())
    assert all(cleaned_df["date"].notna())
