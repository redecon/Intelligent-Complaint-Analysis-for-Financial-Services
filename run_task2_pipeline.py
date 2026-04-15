"""
Task 2 Pipeline Script:
- Reads cleaned reviews
- Computes VADER sentiment
- Preprocesses text for theme extraction
- Extracts TF-IDF keywords per bank
- Defines theme mapping (manual refinement required)
- Assigns themes to reviews
- Saves enriched dataset
- Prints summary tables
"""

import logging
import pandas as pd

from src.sentiment import add_vader_sentiment
from src.themes import preprocess_for_theme, extract_top_tfidf_keywords, assign_themes_by_keywords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_pipeline():
    # Step 1: Load cleaned dataset
    input_path = "data/processed/reviews_clean.csv"
    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Step 2: Add VADER sentiment
    logging.info("Computing VADER sentiment scores...")
    df = add_vader_sentiment(df, text_column="review")

    # Step 3: Preprocess text for theme extraction
    logging.info("Preprocessing text for theme extraction...")
    df["clean_text"] = preprocess_for_theme(df["review"])

    # Step 4: Extract TF-IDF keywords per bank
    banks = df["bank"].unique()
    for bank in banks:
        logging.info(f"Extracting top TF-IDF keywords for {bank} (negative reviews)...")
        keywords = extract_top_tfidf_keywords(df, text_column="clean_text", bank_name=bank, n_keywords=20)
        logging.info(f"Top keywords for {bank}: {keywords}")

    # Step 5: Define theme mapping (manual refinement required)
    # NOTE: Replace with refined mapping after reviewing TF-IDF outputs
    theme_mapping = {
    "Login/Authentication": [
        "login", "password", "fingerprint", "otp", "verification",
        "account", "open", "access"
    ],
    "Transfer/Transaction": [
        "transfer", "send money", "transaction", "payment", "fund",
        "banking", "mobile banking", "banking app"
    ],
    "App Performance": [
        "slow", "crash", "freeze", "lag", "stuck", "loading",
        "update", "version", "fix", "issue", "working", "bad", "worst"
    ],
    "UI/UX": [
        "interface", "design", "navigation", "confusing", "layout",
        "experience"
    ],
    "Customer Support": [
        "support", "help", "response", "call", "email", "service"
    ]
}


    # Step 6: Assign themes
    logging.info("Assigning themes to reviews...")
    df = assign_themes_by_keywords(df, text_column="clean_text", theme_mapping=theme_mapping)

    # Step 7: Save enriched dataset
    output_path = "data/processed/reviews_with_sentiment_themes.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved enriched dataset to {output_path}")

    # Step 8: Print summary tables
    logging.info("Generating summary tables...")

    # Sentiment distribution by bank
    sentiment_summary = df.groupby(["bank", "sentiment_label"]).size().unstack(fill_value=0)
    print("\n=== Sentiment Distribution by Bank ===")
    print(sentiment_summary)

    # Top themes per bank
    theme_summary = df.groupby(["bank", "theme"]).size().reset_index(name="count")
    top_themes = theme_summary.sort_values(["bank", "count"], ascending=[True, False]).groupby("bank").head(5)
    print("\n=== Top Themes per Bank ===")
    print(top_themes)


if __name__ == "__main__":
    run_pipeline()
