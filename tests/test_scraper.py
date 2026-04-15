"""
Unit tests for src/scraper.py
"""

import pytest
import pandas as pd
from unittest.mock import patch
from src.scraper import scrape_bank_reviews


@pytest.fixture
def fake_reviews_batch():
    """Return a fake batch of reviews for mocking."""
    return [
        {
            "content": "Great app experience!",
            "score": 5,
            "at": pd.Timestamp("2024-01-01"),
        },
        {
            "content": "Needs improvement.",
            "score": 3,
            "at": pd.Timestamp("2024-01-02"),
        },
    ]


@patch("src.scraper.reviews")
def test_scrape_bank_reviews_success(mock_reviews, fake_reviews_batch):
    """Test that scraping returns a DataFrame with expected columns."""
    # Mock reviews() to return fake batch and no continuation token
    mock_reviews.return_value = (fake_reviews_batch, None)

    df = scrape_bank_reviews("CBE", "com.fake.app", target_count=2)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    expected_cols = ["bank", "content", "score", "at", "source", "scrape_date"]
    assert list(df.columns) == expected_cols

    # Check values
    assert (df["bank"] == "CBE").all()
    assert (df["source"] == "Google Play").all()
    assert len(df) == 2


@patch("src.scraper.reviews")
def test_scrape_bank_reviews_empty(mock_reviews):
    """Test that scraping returns empty DataFrame if no reviews."""
    mock_reviews.return_value = ([], None)

    df = scrape_bank_reviews("BOA", "com.fake.app", target_count=2)

    assert df.empty
    assert list(df.columns) == ["bank", "content", "score", "at", "source", "scrape_date"]
