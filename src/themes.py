"""
Themes Extraction Module for Banking Reviews.

Provides functions to preprocess text, extract TF-IDF keywords,
assign themes based on keyword mapping, and generate a sample theme mapping.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required NLTK resources are available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_for_theme(text_series: pd.Series) -> pd.Series:
    """
    Preprocess text for theme extraction:
    - Lowercase
    - Remove punctuation and numbers
    - Tokenize
    - Remove stopwords (except 'not', 'no', 'nor')
    - Lemmatize
    - Return cleaned strings

    Parameters
    ----------
    text_series : pd.Series
        Series of text reviews.

    Returns
    -------
    pd.Series
        Cleaned and preprocessed text.
    """
    stop_words = set(stopwords.words("english"))
    # Keep negation words
    stop_words -= {"not", "no", "nor"}
    lemmatizer = WordNetLemmatizer()

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)  # remove punctuation/numbers
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        return " ".join(tokens)

    return text_series.astype(str).apply(clean_text)


def extract_top_tfidf_keywords(df: pd.DataFrame, text_column: str, bank_name: str, n_keywords: int = 20) -> list:
    """
    Extract top TF-IDF keywords from negative reviews for a given bank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing reviews.
    text_column : str
        Column name with review text.
    bank_name : str
        Bank to filter reviews for.
    n_keywords : int, default 20
        Number of top keywords to return.

    Returns
    -------
    list
        Top keywords/phrases ranked by mean TF-IDF score.
    """
    bank_df = df[(df["bank"] == bank_name) & (df["rating"] <= 2)]
    if bank_df.empty:
        return []

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(bank_df[text_column])
    feature_names = vectorizer.get_feature_names_out()

    # Compute mean TF-IDF score per feature
    mean_scores = tfidf_matrix.mean(axis=0).A1
    keywords = sorted(zip(feature_names, mean_scores), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keywords[:n_keywords]]


def assign_themes_by_keywords(df: pd.DataFrame, text_column: str, theme_mapping: dict) -> pd.DataFrame:
    """
    Assign themes to reviews based on keyword mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing reviews.
    text_column : str
        Column name with review text.
    theme_mapping : dict
        Dictionary mapping theme names to keyword lists.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'theme' column.
    """
    def find_themes(text: str) -> str:
        text_lower = text.lower()
        matched = []
        for theme, keywords in theme_mapping.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    matched.append(theme)
                    break
        return ";".join(matched) if matched else "Other"

    df["theme"] = df[text_column].astype(str).apply(find_themes)
    return df


def generate_theme_mapping_example() -> dict:
    """
    Generate a sample theme mapping dictionary for banking pain points.

    Returns
    -------
    dict
        Example theme mapping.
    """
    return {
        "Login/Authentication": ["login", "password", "fingerprint", "face id", "biometric", "otp", "verification"],
        "Transfer/Transaction": ["transfer", "send money", "transaction", "payment", "fund"],
        "App Performance": ["slow", "crash", "freeze", "lag", "stuck", "loading"],
        "UI/UX": ["interface", "design", "navigation", "confusing", "layout"],
        "Customer Support": ["support", "help", "response", "call", "email"],
    }
def run_lda_validation(df: pd.DataFrame, text_column: str, num_topics: int = 3):
    """
    Run LDA topic modeling on negative reviews for exploratory validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing reviews.
    text_column : str
        Column name with review text.
    num_topics : int, default 3
        Number of topics to extract.

    Notes
    -----
    - Uses gensim for LDA.
    - Applies the same preprocessing as preprocess_for_theme.
    - Prints top 5 words per topic.
    - Saves pyLDAvis visualization to data/outputs/lda_viz.html.
    - This is for exploratory validation only, not required for the core pipeline.
    """
    try:
        from gensim import corpora, models
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis
    except ImportError:
        raise ImportError("gensim and pyLDAvis must be installed to run LDA validation.")

    # Filter negative reviews
    neg_df = df[df.get("rating", 5) <= 2]
    if neg_df.empty:
        print("No negative reviews available for LDA validation.")
        return None

    # Preprocess text
    processed = preprocess_for_theme(neg_df[text_column])

    # Tokenize into lists
    tokenized = [doc.split() for doc in processed]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    # Train LDA model
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                passes=10,
                                random_state=42)

    # Print top words per topic
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=5):
        print(f"Topic {idx}: {topic}")

    # Save pyLDAvis visualization
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    output_path = "data/outputs/lda_viz.html"
    pyLDAvis.save_html(lda_vis, output_path)
    print(f"LDA visualization saved to {output_path}")

    return lda_model
