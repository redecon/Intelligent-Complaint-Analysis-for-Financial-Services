# run_pipeline.py
import pandas as pd
from datetime import datetime
from src.scraper import scrape_bank_reviews
from src.preprocess import preprocess_reviews

if __name__ == "__main__":
    # Step 1: Scrape reviews for all banks
    apps = {
        "CBE": "com.combanketh.mobilebanking",
        "BOA": "com.boa.boaMobileBanking",
        "Dashen": "com.dashen.dashensuperapp"
    }

    results = []
    for bank, app_id in apps.items():
        df = scrape_bank_reviews(bank, app_id, target_count=500)
        results.append(df)

    raw_df = pd.concat(results, ignore_index=True)

    # Step 2: Preprocess reviews
    cleaned_df, metrics = preprocess_reviews(raw_df)

    # Step 3: Save processed dataset
    output_path = "data/processed/reviews_clean.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
