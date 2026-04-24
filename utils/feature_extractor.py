"""
feature_extractor.py
Rule-based NLP feature extraction from conversation turns.
These are handcrafted linguistic signals — no LLM needed.
"""

import re
from collections import Counter
from typing import List, Dict
from textblob import TextBlob

from utils.preprocessor import Turn


# ── Lexicons ──────────────────────────────────────────────────────────────────

RED_FLAG_LEXICON = [
    # control
    "you can't", "you should", "you must", "i forbid", "not allowed",
    "i decide", "do what i say", "listen to me",
    # blame shifting
    "your fault", "you made me", "because of you", "you always", "you never",
    "you ruin", "you caused",
    # gaslighting
    "that never happened", "you're imagining", "you're crazy", "you're overreacting",
    "you're too sensitive", "stop being dramatic", "that's not what i said",
    # threats / ultimatums
    "or else", "if you leave", "you'll regret", "i'll leave you", "no one else",
    "you'll be nothing", "you'd be nothing",
    # dismissiveness
    "i don't care", "whatever", "shut up", "stop talking", "don't bother",
    "you wouldn't understand",
    # possessiveness
    "you're mine", "you belong to me", "i own you", "without my permission",
]

GREEN_FLAG_LEXICON = [
    # empathy
    "i understand", "that makes sense", "i hear you", "i can see why",
    "your feelings", "that must be hard", "i'm sorry you feel",
    # respect
    "your choice", "your decision", "i respect", "up to you", "if you want",
    "whatever makes you happy",
    # support
    "i'm here for you", "i support you", "proud of you", "you've got this",
    "i believe in you", "you can do it",
    # accountability
    "my fault", "i was wrong", "i'm sorry", "i apologize", "i messed up",
    "i take responsibility",
    # encouragement
    "you're doing great", "that's amazing", "well done", "i'm proud",
]

BLAME_WORDS = ["your fault", "you made me", "because of you", "you always", "you never", "you caused"]
ULTIMATUM_WORDS = ["or else", "if you leave", "you'll regret", "never talk to you", "i'll leave"]
GASLIGHTING_WORDS = ["never happened", "imagining", "overreacting", "too sensitive", "dramatic", "crazy"]
APOLOGY_WORDS = ["sorry", "apologize", "my fault", "i was wrong", "i messed up"]
CONDITIONAL_APOLOGY = ["sorry but", "sorry if", "sorry you feel", "sorry that you"]


# ── Core Feature Extraction ───────────────────────────────────────────────────

def extract_features(target_turns: List[Turn], all_turns: List[Turn], target_speaker: str) -> Dict:
    """
    Extract all rule-based features for the target speaker.

    Returns a dict of feature scores (mostly 0–1 normalised).
    """
    if not target_turns:
        return {}

    target_texts = [t.text.lower() for t in target_turns]
    full_text = " ".join(target_texts)
    words = full_text.split()
    total_words = max(len(words), 1)
    n_turns = len(target_turns)

    features = {}

    # 1. Red/Green flag lexicon match rate
    red_hits = sum(1 for phrase in RED_FLAG_LEXICON if phrase in full_text)
    green_hits = sum(1 for phrase in GREEN_FLAG_LEXICON if phrase in full_text)
    features["red_lexicon_score"] = min(red_hits / max(len(RED_FLAG_LEXICON) * 0.15, 1), 1.0)
    features["green_lexicon_score"] = min(green_hits / max(len(GREEN_FLAG_LEXICON) * 0.15, 1), 1.0)

    # 2. Pronoun ratio (I vs You) — high "you" = blame-oriented
    i_count = words.count("i") + words.count("i'm") + words.count("i've")
    you_count = words.count("you") + words.count("you're") + words.count("your")
    features["you_to_i_ratio"] = round(you_count / max(i_count, 1), 2)  # >2 = accusatory

    # 3. Blame language density
    blame_hits = sum(1 for phrase in BLAME_WORDS if phrase in full_text)
    features["blame_density"] = min(blame_hits / 3, 1.0)

    # 4. Ultimatum / threat count
    ult_hits = sum(1 for phrase in ULTIMATUM_WORDS if phrase in full_text)
    features["ultimatum_score"] = min(ult_hits / 2, 1.0)

    # 5. Gaslighting signals
    gas_hits = sum(1 for phrase in GASLIGHTING_WORDS if phrase in full_text)
    features["gaslighting_score"] = min(gas_hits / 2, 1.0)

    # 6. Apology quality
    has_apology = any(w in full_text for w in APOLOGY_WORDS)
    has_conditional = any(phrase in full_text for phrase in CONDITIONAL_APOLOGY)
    if has_apology and not has_conditional:
        features["apology_quality"] = 1.0   # genuine
    elif has_apology and has_conditional:
        features["apology_quality"] = 0.3   # conditional
    else:
        features["apology_quality"] = 0.0   # none

    # 7. Question frequency (curiosity / active listening = green)
    q_count = sum(1 for t in target_turns if "?" in t.text)
    features["question_ratio"] = round(q_count / n_turns, 2)

    # 8. Avg sentiment per turn (TextBlob polarity: -1 to 1)
    polarities = [TextBlob(t.text).sentiment.polarity for t in target_turns]
    features["avg_sentiment"] = round(sum(polarities) / n_turns, 3)
    features["sentiment_volatility"] = round(
        max(polarities) - min(polarities) if len(polarities) > 1 else 0, 3
    )

    # 9. Avg turn length (dismissive = very short)
    avg_len = sum(len(t.text.split()) for t in target_turns) / n_turns
    features["avg_turn_length"] = round(avg_len, 1)

    # 10. Reciprocity — does target speak much less than the other person?
    other_turns = [t for t in all_turns if t.speaker != target_speaker]
    other_words = sum(len(t.text.split()) for t in other_turns)
    target_words = sum(len(t.text.split()) for t in target_turns)
    features["word_share"] = round(target_words / max(target_words + other_words, 1), 2)

    return features


def score_summary(features: Dict) -> Dict:
    """
    Convert raw features → interpretable red/green signal scores (0–1).
    Higher red_signal = more red flag behaviour.
    """
    red_signal = (
        features.get("red_lexicon_score", 0) * 0.25 +
        min(features.get("you_to_i_ratio", 1) / 4, 1) * 0.15 +
        features.get("blame_density", 0) * 0.20 +
        features.get("ultimatum_score", 0) * 0.20 +
        features.get("gaslighting_score", 0) * 0.20
    )

    green_signal = (
        features.get("green_lexicon_score", 0) * 0.30 +
        features.get("apology_quality", 0) * 0.20 +
        features.get("question_ratio", 0) * 0.20 +
        max(features.get("avg_sentiment", 0), 0) * 0.15 +
        min(features.get("avg_turn_length", 10) / 30, 1) * 0.15
    )

    return {
        "red_signal": round(min(red_signal, 1.0), 3),
        "green_signal": round(min(green_signal, 1.0), 3),
    }
