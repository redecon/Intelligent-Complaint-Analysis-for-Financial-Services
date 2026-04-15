"""
Visualization Script for Task 2 Results

Generates charts from reviews_with_sentiment_themes.csv:
- Sentiment distribution per bank (bar chart)
- Theme counts per bank (stacked bar chart)
- Word clouds of top TF-IDF keywords per bank (optional)

Saves plots to data/outputs/.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def visualize_results():
    input_path = "data/processed/reviews_with_sentiment_themes.csv"
    df = pd.read_csv(input_path)
    logging.info(f"Loaded enriched dataset from {input_path}")

    # Sentiment distribution per bank
    sentiment_summary = df.groupby(["bank", "sentiment_label"]).size().reset_index(name="count")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=sentiment_summary, x="bank", y="count", hue="sentiment_label")
    plt.title("Sentiment Distribution by Bank")
    plt.ylabel("Number of Reviews")
    plt.savefig("data/outputs/sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved sentiment distribution chart.")

    # Theme counts per bank
    theme_summary = df.groupby(["bank", "theme"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=theme_summary, x="bank", y="count", hue="theme")
    plt.title("Theme Counts by Bank")
    plt.ylabel("Number of Reviews")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig("data/outputs/theme_counts.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved theme counts chart.")

    # Top themes per bank (optional heatmap)
    pivot = theme_summary.pivot(index="theme", columns="bank", values="count").fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Theme Frequency Heatmap")
    plt.savefig("data/outputs/theme_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved theme heatmap chart.")


if __name__ == "__main__":
    visualize_results()
