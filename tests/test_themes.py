import pandas as pd
import pytest

from src import themes


def test_preprocess_for_theme_basic():
    series = pd.Series(["Login failed!!!", "App is slow and crashing"])
    cleaned = themes.preprocess_for_theme(series)
    # Ensure output is lowercase, cleaned, and lemmatized
    assert isinstance(cleaned, pd.Series)
    assert all(isinstance(x, str) for x in cleaned)
    assert "login" in cleaned.iloc[0]
    assert "slow" in cleaned.iloc[1]


def test_extract_top_tfidf_keywords_negative_reviews():
    df = pd.DataFrame({
        "bank": ["CBE", "CBE", "BOA"],
        "rating": [1, 5, 2],
        "review_text": [
            "App crashes when I try to login",
            "Great service and easy to use",
            "Transfer failed and payment stuck"
        ]
    })
    keywords = themes.extract_top_tfidf_keywords(df, "review_text", "CBE", n_keywords=5)
    # Should return a list of strings
    assert isinstance(keywords, list)
    assert all(isinstance(k, str) for k in keywords)
    # Keywords should reflect negative review content
    assert any("login" in k or "crash" in k for k in keywords)


def test_assign_themes_by_keywords_single_and_multiple():
    df = pd.DataFrame({
        "review_text": [
            "I cannot login with my password",
            "The app is slow and crashes often",
            "Customer support did not respond",
            "Great design and navigation"
        ]
    })
    mapping = themes.generate_theme_mapping_example()
    df = themes.assign_themes_by_keywords(df, "review_text", mapping)

    assert "theme" in df.columns
    # First review should be Login/Authentication
    assert "Login/Authentication" in df.loc[0, "theme"]
    # Second review should be App Performance
    assert "App Performance" in df.loc[1, "theme"]
    # Third review should be Customer Support
    assert "Customer Support" in df.loc[2, "theme"]
    # Fourth review should be UI/UX
    assert "UI/UX" in df.loc[3, "theme"]


def test_assign_themes_by_keywords_other_case():
    df = pd.DataFrame({"review_text": ["This app is amazing overall"]})
    mapping = themes.generate_theme_mapping_example()
    df = themes.assign_themes_by_keywords(df, "review_text", mapping)
    assert df.loc[0, "theme"] == "Other"
