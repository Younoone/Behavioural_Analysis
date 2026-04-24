"""
preprocessor.py
Handles two input modes:
  1. CONVERSATION — multi-turn chat with Speaker: message format
  2. SCENARIO     — a first-person narrative describing a situation/relationship

Auto-detects which mode the input is, then returns a unified structure
that the rest of the pipeline can consume identically.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str        # "Person A" / "Person B" / "Narrator" (for scenarios)
    text: str
    turn_index: int


class InputMode:
    CONVERSATION = "conversation"
    SCENARIO = "scenario"


# ── Mode detection ────────────────────────────────────────────────────────────

def detect_input_mode(raw_text: str) -> str:
    """
    Heuristic to decide if input is a conversation or a scenario narrative.

    A conversation has lines that start with a consistent "Speaker: " pattern.
    A scenario is flowing prose (Reddit-style story).
    """
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    speaker_pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_ ]{0,20}):\s*.{5,}$")

    matched = sum(1 for l in lines if speaker_pattern.match(l))
    match_ratio = matched / max(len(lines), 1)

    # If more than 40% of lines look like "Speaker: message" → conversation
    return InputMode.CONVERSATION if match_ratio >= 0.4 else InputMode.SCENARIO


# ── Conversation parser ───────────────────────────────────────────────────────

def parse_conversation(raw_text: str) -> Tuple[List[Turn], Dict[str, str]]:
    """
    Parse a multi-turn chat into structured Turn objects.
    Supports: "A: msg", "Person A: msg", "John: msg" patterns.
    """
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    speaker_pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_ ]{0,20}):\s*(.+)$")

    detected_speakers = set()
    parsed_lines = []

    for line in lines:
        match = speaker_pattern.match(line)
        if match:
            speaker_raw = match.group(1).strip()
            text = match.group(2).strip()
            detected_speakers.add(speaker_raw)
            parsed_lines.append((speaker_raw, text))
        else:
            # Continuation line — append to previous turn
            if parsed_lines:
                prev_speaker, prev_text = parsed_lines[-1]
                parsed_lines[-1] = (prev_speaker, prev_text + " " + line)

    # Normalise to "Person A", "Person B", etc.
    speaker_list = sorted(detected_speakers)
    speaker_map = {sp: f"Person {chr(65 + i)}" for i, sp in enumerate(speaker_list)}

    turns = []
    for idx, (speaker_raw, text) in enumerate(parsed_lines):
        normalised = speaker_map.get(speaker_raw, speaker_raw)
        turns.append(Turn(speaker=normalised, text=text, turn_index=idx))

    return turns, speaker_map


# ── Scenario parser ───────────────────────────────────────────────────────────

def parse_scenario(raw_text: str) -> Tuple[List[Turn], Dict[str, str]]:
    """
    Parse a first-person narrative/scenario into pseudo-turns.

    Strategy:
    - Split by paragraph or sentence boundaries into chunks
    - Assign all chunks to a single "Narrator" speaker
    - The narrator IS the person being assessed

    This lets the rest of the pipeline (feature extractor, LLM) work
    identically on scenario text as on conversation turns.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text.strip()) if p.strip()]

    # If only one paragraph, split by sentences instead
    if len(paragraphs) == 1:
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        paragraphs = [s.strip() for s in sentence_pattern.split(paragraphs[0]) if s.strip()]

    turns = []
    for idx, chunk in enumerate(paragraphs):
        turns.append(Turn(speaker="Narrator", text=chunk, turn_index=idx))

    speaker_map = {"Narrator": "Narrator"}
    return turns, speaker_map


# ── Quoted speech extractor (for scenarios) ───────────────────────────────────

def extract_quoted_speech(raw_text: str) -> List[str]:
    """
    Pull out direct quotes from a scenario narrative.
    Useful for detecting what the person actually said vs. what they described.
    e.g. 'I said "friends?" and she said "if that's what you want"'
    """
    # Match both "..." and '...'
    quotes = re.findall(r'["\u201c\u201d]([^""\u201c\u201d]{3,})["\u201c\u201d]', raw_text)
    return [q.strip() for q in quotes]


# ── Unified entry point ───────────────────────────────────────────────────────

def parse_input(raw_text: str) -> Tuple[List[Turn], Dict[str, str], str]:
    """
    Auto-detect input type and parse accordingly.

    Returns:
        turns       — list of Turn objects
        speaker_map — original → normalised speaker name map
        mode        — InputMode.CONVERSATION or InputMode.SCENARIO
    """
    mode = detect_input_mode(raw_text)
    if mode == InputMode.CONVERSATION:
        turns, speaker_map = parse_conversation(raw_text)
    else:
        turns, speaker_map = parse_scenario(raw_text)
    return turns, speaker_map, mode


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_target_turns(turns: List[Turn], target_label: str) -> List[Turn]:
    """Filter turns for the person being assessed."""
    return [t for t in turns if t.speaker == target_label]


def get_available_speakers(turns: List[Turn]) -> List[str]:
    """Return unique speakers in order of first appearance."""
    seen = []
    for t in turns:
        if t.speaker not in seen:
            seen.append(t.speaker)
    return seen


def get_conversation_stats(turns: List[Turn], mode: str) -> dict:
    """Basic stats about the input."""
    if not turns:
        return {}

    speakers = get_available_speakers(turns)
    turn_counts = {sp: sum(1 for t in turns if t.speaker == sp) for sp in speakers}
    word_counts = {sp: sum(len(t.text.split()) for t in turns if t.speaker == sp) for sp in speakers}

    stats = {
        "mode": mode,
        "total_turns": len(turns),
        "total_words": sum(word_counts.values()),
        "speakers": speakers,
        "turn_counts": turn_counts,
        "word_counts": word_counts,
        "avg_words_per_turn": {
            sp: round(word_counts[sp] / max(turn_counts[sp], 1), 1)
            for sp in speakers
        }
    }
    return stats
