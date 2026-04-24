# 🚩 Red Flag Detector

A hybrid NLP pipeline that analyzes conversations and first-person narratives to assess whether a person exhibits **red flag** or **green flag** behavioral patterns in relationships.

Built as an NLP course project using rule-based feature engineering + open-source LLM behavioral scoring.

---

## What It Does

You paste either a **chat conversation** or a **Reddit-style scenario** (first-person story), and the system:

- Auto-detects which input type it is
- Extracts 10 handcrafted linguistic features (pronoun ratios, gaslighting language, blame density, sentiment trajectory, etc.)
- Sends the text to a large language model for deep behavioral analysis across 7 dimensions
- Fuses both scores into a final weighted verdict
- Explains every decision — not just a label, but *why*

For scenario mode, it can analyze **both** the narrator and the person they're describing, shown as a side-by-side tabbed comparison.

---

## Project Structure

```
behavioural_analysis/
│
├── app.py                     ← Streamlit frontend (main entry point)
│
├── utils/
│   ├── preprocessor.py        ← Auto-detects input mode, parses conversations & narratives
│   ├── feature_extractor.py   ← 10 rule-based NLP features (no LLM)
│   ├── llm_analyzer.py        ← Qwen2.5-7B via HuggingFace Inference Router
│   └── aggregator.py          ← Weighted score fusion → final verdict + explainability
│
├── images/
│   ├── red_flag.png            ← Displayed on 🚩 Red Flag verdict
│   └── green_flag.png          ← Displayed on 💚 Green Flag verdict
│
├── .env                        ← Your HF_API_KEY (never share this)
├── .env.example                ← Template
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

```
Raw Text Input (Chat or Scenario)
            ↓
    [preprocessor.py]
    Auto-detect mode (conversation / scenario)
    Parse into structured Turn objects
    Extract quoted speech from narratives
            ↓
            ├─────────────────────────────────────┐
            ↓                                     ↓
  [feature_extractor.py]              [llm_analyzer.py]
  Rule-based NLP features             Qwen2.5-7B via HuggingFace
  ─────────────────────               ──────────────────────────
  • Red/Green lexicon match           • Empathy
  • Pronoun ratio (I vs You)          • Control behavior
  • Blame density                     • Gaslighting
  • Ultimatum / threat score          • Accountability
  • Gaslighting language              • Respect
  • Apology quality                   • Emotional manipulation
  • Question frequency                • Supportiveness
  • Avg sentiment (TextBlob)          • Key behaviors (3 observed)
  • Sentiment volatility              • One-line behavioral summary
  • Turn length & reciprocity
            │                                     │
            └──────────────┬──────────────────────┘
                           ↓
                  [aggregator.py]
          Final Score = 0.35 × Rule-based + 0.65 × LLM
          Weighted dimensions (emotional_manipulation = 2x)
          Verdict threshold: gap > 0.08 → Red/Green Flag
                           ↓
                    [app.py / Streamlit UI]
          Verdict box + image · Signal bars · Radar chart
          Dimension breakdown · Rule-based flags · Key behaviors
          Quoted speech · Parsed input viewer
          (Scenario: tabbed Narrator vs Other Person comparison)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd redflag_detector
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 2. Get a HuggingFace API key

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → Role: **Read** → enable **"Make calls to Inference Providers"**
3. Copy the token (starts with `hf_...`)

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
HF_API_KEY=hf_your_token_here
```

This key is loaded by the app at startup via `python-dotenv`. It is never shown to users.

### 4. Set your verdict images (optional)

Open `app.py` and find this block near the top:

```python
VERDICT_IMAGES = {
    "red_flag":   r"D:\BehavioralAnalysis\images\red_flag.png",
    "green_flag": r"D:\BehavioralAnalysis\images\green_flag.png",
}
```

Replace the paths with wherever your images are stored.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Supported Input Formats

### Mode 1 — Conversation (auto-detected)

Any consistent `Speaker: message` format:

```
Person A: You never listen to me.
Person B: I'm listening, go ahead.
Person A: If you leave me, you'll regret it.
Person B: That's not okay to say.
```

Also works with `A: / B:`, `John: / Sarah:`, or any two-speaker pattern.

### Mode 2 — Scenario / Reddit-style narrative (auto-detected)

A first-person account describing a situation:

```
Me and my boyfriend have been together for 2 years. Last month I got a
job promotion and when I told him, instead of being happy he got quiet
and said I was probably only chosen because my manager likes me personally...
```

In scenario mode you can optionally analyze **both** the narrator and the
person they're describing — shown as two separate tabbed analyses.

---

## Rule-Based Features

| Feature | Description | Signal |
|---|---|---|
| Red lexicon score | Matches control/manipulation phrases | 🚩 |
| Green lexicon score | Matches empathy/support phrases | 💚 |
| You-to-I pronoun ratio | High "you" = accusatory/blame pattern | 🚩 |
| Blame density | "your fault", "because of you" etc. | 🚩 |
| Ultimatum score | Threats and conditional statements | 🚩 |
| Gaslighting score | Reality-denial language | 🚩 |
| Apology quality | Genuine vs conditional vs absent | 💚 |
| Question ratio | Curiosity = active listening signal | 💚 |
| Avg sentiment | TextBlob polarity per turn | Mixed |
| Sentiment volatility | Tone shifts when challenged | 🚩 |

---

## LLM Behavioral Dimensions

Scored 0.0–1.0 by Qwen2.5-7B:

| Dimension | Type | Meaning |
|---|---|---|
| Empathy | 💚 Green | Acknowledges others' feelings |
| Control behavior | 🚩 Red | Restricts autonomy or monitors |
| Gaslighting | 🚩 Red | Denies/distorts reality |
| Accountability | 💚 Green | Owns mistakes genuinely |
| Respect | 💚 Green | Treats others with dignity |
| Emotional manipulation | 🚩 Red | Uses emotions to control |
| Supportiveness | 💚 Green | Encourages and uplifts |

---

## Scoring & Weights

```
Final Score = 0.35 × Rule-based + 0.65 × LLM
```

LLM is weighted higher because it captures semantic context and nuance
that lexicon matching cannot.

Within the LLM layer, dimensions are weighted unequally:

```python
RED_WEIGHTS = {
    "control_behavior":       1.0,
    "gaslighting":            1.2,
    "emotional_manipulation": 2.0   # highest — covers covert/deceptive behavior
}
```

Verdict is determined by the gap between red and green signals:

```
gap > 0.08  →  🚩 Red Flag
gap < -0.08 →  💚 Green Flag
otherwise   →  🟡 Neutral / Mixed
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Sentiment analysis | TextBlob |
| NLP utilities | NLTK |
| LLM | Qwen2.5-7B-Instruct via HuggingFace Inference Router |
| Charts | Plotly |
| Environment config | python-dotenv |

---

## Why This Is an NLP Application (Not GenAI)

The LLM in this project is used purely as a **classifier and feature extractor** — it reads text and returns structured behavioral scores. No text is generated for the user. This is functionally identical to using BERT for sentiment classification or spaCy for named entity recognition — the model is a pretrained component in an NLP pipeline, not a generative tool.

---

## Academic Notes

**Key insight from development:** Rule-based features detect *verbal* red flags well (gaslighting phrases, ultimatum language, blame-shifting words). But *behavioral* red flags in narratives — secret-keeping, hidden folders, non-consensual behavior, defensive escalation — leave no linguistic fingerprint and require LLM-level semantic understanding. This distinction between verbal and behavioral signal detection is a meaningful contribution of this project.

**Limitations:**
- Scenario analysis relies entirely on the narrator's perspective — the other person cannot defend themselves
- Rule-based lexicon is English-only
- LLM output can vary slightly between runs due to temperature sampling
- Free tier HuggingFace API has rate limits and occasional cold-start delays

---

## Requirements

```
streamlit==1.35.0
requests==2.31.0
textblob==0.18.0
nltk==3.8.1
python-dotenv==1.0.1
plotly==5.22.0
pandas==2.2.2
```
