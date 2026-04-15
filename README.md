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

### Sentiment and Thematic Analysis
For Task 2, Iconducted sentiment and thematic analysis on the cleaned corpus of banking application reviews. The objective was to quantify customer sentiment and identify recurring pain points that could inform product and service improvements.

#### Sentiment Analysis  
The primary engine employed was VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK. VADER was selected because it is optimized for short, informal text such as social media posts and app reviews, and it natively handles emojis, slang, and intensifiers. Its lexicon‑based approach provides transparent scoring, which is advantageous for business explainability. Reviews were classified using the following thresholds: compound score ≥ 0.05 as Positive, ≤ –0.05 as Negative, and values in between as Neutral.

Validation was performed by benchmarking against DistilBERT(distilbert‑base‑uncased‑finetuned‑sst‑2‑english) on a random sample of 50 reviews to assess domain alignment. Aggregated sentiment scores were calculated per bank and per star rating to provide comparative insights.

**Thematic Analysis**
Text preprocessing for theme extraction included lowercasing, removal of punctuation and numbers, tokenization, stopword removal (with negations preserved), and lemmatization using NLTK. Keyword discovery was conducted using TF‑IDF vectorization with an n‑gram range of (1, 2), applied to negative reviews (rating ≤ 2) per bank. The top 20 distinctive keywords and phrases were extracted for each institution.
Keywords were then manually grouped into 3–5 actionable business themes per bank, such as Login/Authentication Issues, Transfer Performance, App Stability, UI/UX Feedback, and Customer Support. Theme assignment was rule‑based, relying on keyword presence within reviews; multiple themes could be applied to a single review if applicable.
As an optional validation step, Latent Dirichlet Allocation (LDA) was run on negative reviews to confirm thematic coherence. Results were visualized with pyLDAvis to provide exploratory insight into topic distributions.

#### Outputs
**Sentiment Distribution by Bank**
sentiment_label  negative  neutral  positive
bank                                        
BOA                    87      154       175
CBE                    62      125       216
Dashen                 61      119       246

**Top Themes per Bank**
BOA     App Performance (69), Transfer/Transaction (19), 
        Login/Authentication (13), Customer Support (12), Other (257)

CBE     App Performance (52), Transfer/Transaction (22), 
        Customer Support (15), Login/Authentication (10), Other (261)

Dashen  App Performance (49), Transfer/Transaction (15), 
        Transfer/Transaction;App Performance (17), Login/Authentication (9), Other (268)

### Insights

**App Performance** emerged as the dominant theme across all three banks, reflecting widespread issues with updates, crashes, and slow performance.

**Transfer/Transaction** concerns were consistently present, particularly in Dashen and BOA.

**Login/Authentication and Customer Support** appeared as secondary but notable themes.

The **“Other”** category decreased compared to initial runs, but still captures reviews outside the defined theme mapping. Further refinement of keyword lists can reduce this residual category.

### Visualizations
To complement the tabular summaries, the following charts were generated and saved in data/outputs/:

**sentiment_distribution.png** – Grouped bar chart showing negative, neutral, and positive review counts per bank.

**theme_counts.png** – Bar chart of theme frequencies per bank, highlighting dominant pain points.

**theme_heatmap.png** – Heatmap matrix of theme counts across banks, useful for comparative analysis.

These visuals provide reviewer‑friendly insights into both sentiment distribution and thematic prevalence, making it easier to identify where customer experience challenges are concentrated.