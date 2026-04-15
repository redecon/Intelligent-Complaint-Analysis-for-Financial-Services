### Methodology
The dataset was sourced from the Google Play Store, focusing on user reviews for three Ethiopian banking applications: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank.

**Collection Method:** Reviews were collected using the google-play-scraper library, with a target of 500 reviews per bank. Parameters were restricted to the English language, Ethiopia region, and sorted by newest first to ensure recency. Pagination was handled until the target count was reached or no further reviews were available.

**Collection Date:** 4/15/2026

**Preprocessing Steps:**

- Removal of duplicate entries based on review text and bank.

- Exclusion of rows with missing or empty review content.

- Normalization of review dates, with invalid entries discarded.

- Standardization of column names (review, rating, date, bank, source, scrape_date).

- Conversion of ratings to integer format.

**Final Dataset:** After preprocessing, the dataset size was reduced from the initial collection, with a retention rate of  83% (calculated as final count divided by initial count).

**Output File:** The cleaned dataset is stored at: data/processed/reviews_clean.csv

### Results Summary
The scraping pipeline successfully collected 1,500 reviews across three banking applications (CBE, BOA, Dashen), meeting the target of 500 reviews per bank.

**Preprocessing Outcomes:**

- Duplicates removed: 255

- Missing text removed: 0

- Invalid dates removed: 0

- Final dataset size: 1,245 reviews

- Retention rate: 83%

**Interpretation:**
Approximately one in six reviews was discarded due to duplication, while the majority of records were retained. The final dataset exceeds the minimum requirement of 400 reviews per bank, ensuring sufficient coverage for downstream sentiment analysis.

**Output:** 
The cleaned dataset is stored at:
data/processed/reviews_clean.csv
