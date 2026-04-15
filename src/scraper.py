"""
Scraper module for Google Play reviews of Ethiopian banking apps.
Uses google-play-scraper to fetch reviews and save them into a CSV file.
"""

import logging
from datetime import datetime
import pandas as pd
from google_play_scraper import reviews, Sort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def scrape_bank_reviews(bank_name, app_id, target_count=500):
    """
    Scrape Google Play reviews for a given bank app.

    Parameters
    ----------
    bank_name : str
        Name of the bank (e.g., 'CBE').
    app_id : str
        Google Play app ID.
    target_count : int, optional
        Number of reviews to scrape (default is 500).

    Returns
    -------
    pd.DataFrame
        DataFrame containing reviews with selected columns.
    """
    logging.info("Starting scrape for %s (%s)", bank_name, app_id)

    all_reviews = []
    continuation_token = None

    while len(all_reviews) < target_count:
        try:
            batch, continuation_token = reviews(
                app_id,
                sort=Sort.NEWEST,
                count=100,
                continuation_token=continuation_token
            )
            if not batch:
                break
            all_reviews.extend(batch)
            logging.info(
                "Fetched %d reviews for %s so far...",
                len(all_reviews),
                bank_name
            )
            if continuation_token is None:
                break
        except Exception as e:
            logging.error("Error scraping %s: %s", bank_name, str(e))
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)

    if df.empty:
        logging.warning("No reviews scraped for %s", bank_name)
        return pd.DataFrame(columns=[
            "bank", "content", "score", "at", "source", "scrape_date"
        ])

    # Add metadata columns
    df["bank"] = bank_name
    df["source"] = "Google Play"
    df["scrape_date"] = datetime.today().strftime("%Y-%m-%d")

    # Select required columns
    df = df[["bank", "content", "score", "at", "source", "scrape_date"]]

    logging.info("Finished scrape for %s: %d reviews", bank_name, len(df))
    return df


if __name__ == "__main__":
    # Define apps
    apps = {
        "CBE": "com.combanketh.mobilebanking",
        "BOA": "com.boa.boaMobileBanking",
        "Dashen": "com.dashen.dashensuperapp"
    }

    results = []

    for bank, app_id in apps.items():
        try:
            df = scrape_bank_reviews(bank, app_id, target_count=500)
            results.append(df)
        except Exception as e:
            logging.error("Failed to scrape %s: %s", bank, str(e))

    if results:
        final_df = pd.concat(results, ignore_index=True)
        today_str = datetime.today().strftime("%Y%m%d")
        output_path = f"data/raw/play_store_reviews_{today_str}.csv"
        final_df.to_csv(output_path, index=False)
        logging.info("Saved scraped reviews to %s", output_path)
    else:
        logging.warning("No reviews scraped for any bank.")
