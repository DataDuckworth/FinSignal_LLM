"""Baselines: Loughran-McDonald lexicon score and FinBERT score.

The TF-IDF + logistic regression baseline needs hyper-parameter tuning, so it lives in notebook 02.
"""
import re

import pandas as pd
from transformers import pipeline


def load_lm_words(path: str):
    """Positive / negative word sets from the Loughran-McDonald master dictionary (see notebook 02)."""
    words = pd.read_csv(path)
    return set(words.loc[words["polarity"] > 0, "word"]), set(words.loc[words["polarity"] < 0, "word"])


def lexicon_score(headlines, pos_words, neg_words):
    """score = (n_positive - n_negative) / n_words.  Simplified: no negation handling, no tf-idf weights."""
    scores = []
    for h in headlines:
        tokens = re.findall(r"[A-Za-z]+", str(h).upper())      # the dictionary is upper case
        if not tokens:
            scores.append(0.0)
            continue
        pos = sum(t in pos_words for t in tokens)
        neg = sum(t in neg_words for t in tokens)
        scores.append((pos - neg) / len(tokens))
    return scores


def finbert_probabilities(headlines, batch_size=32):
    """FinBERT class probabilities (positive, negative, neutral), one row per headline. Used for the Part A check."""
    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    outs = clf(list(headlines), batch_size=batch_size)
    return pd.DataFrame([{d["label"]: d["score"] for d in o} for o in outs])[["positive", "negative", "neutral"]]


def finbert_scores(headlines, batch_size=32):
    """FinBERT baseline: score = P(positive) - P(negative)."""
    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    outs = clf(list(headlines), batch_size=batch_size)
    return [{d["label"]: d["score"] for d in o}.get("positive", 0) -
            {d["label"]: d["score"] for d in o}.get("negative", 0) for o in outs]
