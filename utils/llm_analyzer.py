"""
llm_analyzer.py
Uses HuggingFace Inference Router (router.huggingface.co) with Mistral-7B.
API key loaded from .env — never exposed to user.
"""

import os
import json
import requests
from typing import List, Dict
from utils.preprocessor import Turn, InputMode

# ── HuggingFace Router config ─────────────────────────────────────────────────
# router.huggingface.co auto-picks the fastest available provider
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
HF_API_KEY = os.getenv("HF_API_KEY", "")

def _get_headers():
    # Re-read at call time so load_dotenv() has had a chance to run
    key = os.getenv("HF_API_KEY", "")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

# ── JSON schema ───────────────────────────────────────────────────────────────
JSON_SCHEMA = """{
  "empathy":               {"score": 0.0, "reason": "max 10 words"},
  "control_behavior":      {"score": 0.0, "reason": "max 10 words"},
  "gaslighting":           {"score": 0.0, "reason": "max 10 words"},
  "accountability":        {"score": 0.0, "reason": "max 10 words"},
  "respect":               {"score": 0.0, "reason": "max 10 words"},
  "emotional_manipulation":{"score": 0.0, "reason": "max 10 words"},
  "supportiveness":        {"score": 0.0, "reason": "max 10 words"},
  "overall_verdict": "red_flag or green_flag or neutral",
  "confidence": 0.0,
  "summary": "one sentence max 20 words",
  "key_behaviors": ["behavior 1", "behavior 2", "behavior 3"]
}"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_conversation_prompt(turns: List[Turn], target_speaker: str) -> str:
    convo_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    return f"""Analyze the conversation below. Focus ONLY on {target_speaker}'s behavior and communication patterns.

CONVERSATION:
{convo_text}

Rate {target_speaker} on each dimension 0.0 to 1.0. Respond ONLY with this JSON and nothing else:
{JSON_SCHEMA}"""


def build_scenario_narrator_prompt(narrative: str) -> str:
    return f"""Read the first-person narrative below.
Analyze the NARRATOR's behavior, intentions, and emotional patterns as revealed by their own words.
Look at what they DID, what they THOUGHT, and what their actions reveal about their character.

NARRATIVE:
{narrative}

Rate the NARRATOR on each dimension 0.0 to 1.0. Respond ONLY with this JSON and nothing else:
{JSON_SCHEMA}"""


def build_scenario_other_prompt(narrative: str) -> str:
    return f"""Read the first-person narrative below.
Analyze the OTHER PERSON described — the person the narrator is interacting with.
Base your analysis ONLY on how the narrator describes the other person's actions and behavior.

Pay special attention to these HIGH-SEVERITY signals if present:
- Any behavior that is illegal or could constitute a criminal act (e.g. non-consensual recording, 
  theft, fraud, assault, harassment, stalking, possession of inappropriate content)
- Secret or hidden behavior deliberately concealed from the narrator
- Defensive or aggressive reactions when confronted with evidence
- Repeated patterns of problematic behavior over time
- Betrayal of trust in a close relationship

If ANY illegal behavior is described, the control_behavior and emotional_manipulation 
scores should be at least 0.8, and overall_verdict must be "red_flag" regardless of 
other dimensions.

NARRATIVE:
{narrative}

Rate the OTHER PERSON on each dimension 0.0 to 1.0. Respond ONLY with this JSON:
{JSON_SCHEMA}"""


# ── API call ──────────────────────────────────────────────────────────────────

def _call_hf(user_prompt: str) -> Dict:
    """Call HuggingFace router chat completions endpoint."""
    key = os.getenv("HF_API_KEY", "")
    if not key:
        return {"success": False, "error": "HF_API_KEY not set in .env file."}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a behavioral analyst. Respond ONLY with valid JSON. No explanation, no markdown, no extra text."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 700,
        "temperature": 0.2,
        "stream": False,
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)

        if response.status_code == 503:
            return {"success": False, "error": "Model loading on HuggingFace. Wait 20s and retry."}
        if response.status_code == 401:
            return {"success": False, "error": "Invalid HuggingFace API key. Check HF_API_KEY in .env."}
        if response.status_code == 429:
            return {"success": False, "error": "Rate limit hit. Wait a moment and retry."}
        if response.status_code != 200:
            return {"success": False, "error": f"HF API error {response.status_code}: {response.text[:300]}"}

        raw = response.json()
        text = raw["choices"][0]["message"]["content"].strip()

        # Strip any accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # Extract JSON block robustly
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"success": False, "error": f"No JSON in response: {text[:300]}"}

        data = json.loads(text[start:end])
        return {"success": True, "data": data}

    except (KeyError, IndexError) as e:
        return {"success": False, "error": f"Unexpected response structure: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse failed: {str(e)}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "HuggingFace API timed out. Try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_with_llm(
    turns: List[Turn],
    target_speaker: str,
    mode: str = InputMode.CONVERSATION,
    analyze_other: bool = False,
) -> Dict:
    """
    Run LLM behavioral analysis via HuggingFace router.

    Returns:
        {
          "narrator": { "success": bool, "data": {...} },
          "other":    { "success": bool, "data": {...} } | None
        }
    """
    narrative = "\n\n".join(t.text for t in turns)

    if mode == InputMode.SCENARIO:
        narrator_result = _call_hf(build_scenario_narrator_prompt(narrative))
        other_result    = _call_hf(build_scenario_other_prompt(narrative)) if analyze_other else None
        return {"narrator": narrator_result, "other": other_result}
    else:
        convo_result = _call_hf(build_conversation_prompt(turns, target_speaker))
        return {"narrator": convo_result, "other": None}


# ── Dimension metadata ────────────────────────────────────────────────────────

def get_dimension_labels() -> List[str]:
    return ["empathy", "control_behavior", "gaslighting",
            "accountability", "respect", "emotional_manipulation", "supportiveness"]

def get_red_dimensions() -> List[str]:
    return ["control_behavior", "gaslighting", "emotional_manipulation"]

def get_green_dimensions() -> List[str]:
    return ["empathy", "accountability", "respect", "supportiveness"]