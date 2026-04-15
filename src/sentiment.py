"""
Sentiment Analysis Module using VADER and DistilBERT.

This module provides:
1. VADER sentiment scoring and labeling.
2. Validation against DistilBERT sentiment pipeline (sample-based).
3. Bank-level sentiment summaries.

Note: DistilBERT comparison requires torch and transformers installed.
"""

import logging
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download

# Ensure VADER lexicon is available
download("vader_lexicon")


def add_vader_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """
    Add VADER sentiment scores and labels to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing review text.
    text_column : str, default 'review_text'
        Column name containing review text.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - vader_compound: compound sentiment score
        - sentiment_label: categorical label ('positive', 'neutral', 'negative')

    Raises
    ------
    KeyError
        If the specified text_column does not exist in df.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame.")

    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df[text_column].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])

    def label_score(score: float) -> str:
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["vader_compound"].apply(label_score)
    return df


def compare_with_distilbert_sample(df: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    """
    Compare VADER sentiment labels with DistilBERT predictions on a sample.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'review_text' and 'sentiment_label'.
    sample_size : int, default 50
        Number of reviews to sample for comparison.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - review_text
        - vader_label
        - distilbert_label
        - agreement (True/False)
        Plus a printed agreement percentage.

    Notes
    -----
    Requires torch and transformers installed.
    May be slow; intended for validation only.
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("transformers and torch must be installed to use DistilBERT comparison.")

    if "review_text" not in df.columns or "sentiment_label" not in df.columns:
        raise KeyError("DataFrame must contain 'review_text' and 'sentiment_label' columns.")

    sample_df = df.sample(min(sample_size, len(df)), random_state=42).copy()

    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    distilbert_results = nlp(sample_df["review_text"].tolist())

    sample_df["distilbert_label"] = [res["label"].lower() for res in distilbert_results]
    sample_df["vader_label"] = sample_df["sentiment_label"]
    sample_df["agreement"] = sample_df["vader_label"] == sample_df["distilbert_label"]

    agreement_pct = sample_df["agreement"].mean() * 100
    logging.info(f"VADER vs DistilBERT agreement: {agreement_pct:.2f}%")

    return sample_df


def sentiment_summary_by_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sentiment by bank and rating.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'bank', 'rating', 'vader_compound', 'sentiment_label'.

    Returns
    -------
    pd.DataFrame
        Grouped summary with:
        - average vader_compound
        - count of reviews
        - percentage of negative/neutral/positive
    """
    required_cols = {"bank", "rating", "vader_compound", "sentiment_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    summary = (
        df.groupby(["bank", "rating"])
        .apply(lambda g: pd.Series({
            "avg_vader_compound": g["vader_compound"].mean(),
            "review_count": len(g),
            "pct_negative": (g["sentiment_label"] == "negative").mean() * 100,
            "pct_neutral": (g["sentiment_label"] == "neutral").mean() * 100,
            "pct_positive": (g["sentiment_label"] == "positive").mean() * 100,
        }))
        .reset_index()
    )

    return summary
