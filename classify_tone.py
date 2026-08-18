"""
classify_tone.py
================
Classify the *tone* (emotion) of Reddit comments — finer-grained than
positive/negative. Each comment gets a probability across seven emotions:

    anger, disgust, fear, joy, neutral, sadness, surprise

plus a single `tone_top` label and its `tone_confidence`.

Why seven emotions (and how to go finer)
---------------------------------------
Plain positive/negative collapses very different reactions — fear of
escalation, anger at an aggressor, grief over casualties all read as "negative"
but mean different things for a discourse study. The default model
(j-hartmann/emotion-english-distilroberta-base) predicts Ekman's six basic
emotions plus neutral, which is a good fit for short, punchy Reddit comments.
For even finer tone, pass --model SamLowe/roberta-base-go_emotions (27 GoEmotions
labels: admiration, annoyance, grief, optimism, ...); the script adapts to
whatever label set the model returns.

Backends
--------
1. PRIMARY: a Hugging Face text-classification model (neural, context-aware —
   understands "freeze their balls off" as hostile, catches sarcasm far better
   than word lists). Needs the model cached / one-time internet to download.
2. FALLBACK: a small self-contained emotion lexicon (no downloads, always runs)
   so the pipeline works offline. It is keyword-based and much weaker — use it
   only to smoke-test the plumbing; trust the neural backend for real results.

Non-text rows
-------------
[removed] / [deleted] / empty bodies carry no tone. They are NOT sent to the
model; they get tone_top = "no_text" and NaN emotion scores, so they stay in
the table (row counts line up with the input) without polluting the analysis.

Usage
-----
    python classify_tone.py \
        --in comments_for_links.csv \
        --out comments_toned.csv \
        --text-col body

Then, e.g., mean fear per event:
    df.groupby("event_id")[["anger","fear","sadness","joy"]].mean()
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Canonical 7-way schema. The neural default emits exactly these; the lexicon
# fallback fills the same columns so downstream code is backend-agnostic. (A
# 27-label GoEmotions model overrides this list at runtime — see classify().)
CANON_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_DEAD_BODIES = {"[removed]", "[deleted]", ""}


# --------------------------------------------------------------------------- #
# Fallback lexicon (deliberately small; only for offline smoke-testing)
# --------------------------------------------------------------------------- #
_LEXICON = {
    "anger": [
        "angry", "furious", "outrage", "outraged", "rage", "hate", "disgusting",
        "pathetic", "coward", "bastard", "idiot", "stupid", "damn", "fuck",
        "fucking", "aggressor", "invaders", "criminal", "war criminal", "insane",
    ],
    "disgust": [
        "disgust", "disgusting", "sick", "sickening", "vile", "revolting",
        "gross", "repulsive", "vomit", "nauseating", "atrocity", "atrocities",
    ],
    "fear": [
        "afraid", "scared", "terrified", "terrifying", "fear", "feared", "worry",
        "worried", "anxious", "nervous", "dread", "threat", "danger", "dangerous",
        "nuclear", "wwiii", "ww3", "escalate", "escalation", "panic", "uh oh",
        "rut roh", "here we go",
    ],
    "joy": [
        "happy", "glad", "great", "awesome", "love", "hope", "hopeful", "relief",
        "relieved", "good news", "yay", "lol", "lmao", "haha", "funny", "based",
    ],
    "sadness": [
        "sad", "sadly", "tragic", "tragedy", "heartbreaking", "heartbroken",
        "cry", "crying", "grief", "mourn", "devastating", "devastated", "sorry",
        "rip", "casualties", "civilians", "innocent", "poor", "suffering",
    ],
    "surprise": [
        "wow", "shocked", "shocking", "unexpected", "suddenly", "unbelievable",
        "cant believe", "can't believe", "wtf", "whoa", "omg", "holy",
        "did not expect", "didn't expect", "no way", "really?",
    ],
}
_LEX_COMPILED = {
    emo: [re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in words]
    for emo, words in _LEXICON.items()
}


def _lexicon_scores(text: str) -> dict:
    """Return a normalized 7-way distribution from keyword hits.

    If no emotion words match, all mass goes to 'neutral'. Otherwise the six
    non-neutral emotions are normalized to sum to 1 (neutral = 0). This is
    intentionally crude; it exists so the script runs with no model available.
    """
    hits = {emo: 0 for emo in CANON_EMOTIONS}
    for emo, patterns in _LEX_COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                hits[emo] += 1
    total = sum(hits.values())
    scores = {emo: 0.0 for emo in CANON_EMOTIONS}
    if total == 0:
        scores["neutral"] = 1.0
    else:
        for emo in _LEXICON:
            scores[emo] = hits[emo] / total
    return scores


# --------------------------------------------------------------------------- #
# Backend loader
# --------------------------------------------------------------------------- #
def _load_hf(model_name: str, device: int):
    """Try to build a HF pipeline returning ALL label scores. None on failure."""
    try:
        from transformers import pipeline

        clf = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,        # return every label's score, not just the top one
            truncation=True,   # comments longer than the model max are truncated
            device=device,
        )
        return clf
    except Exception as exc:  # noqa: BLE001 - fallback is deliberate
        print(
            f"[warn] HF model '{model_name}' unavailable ({str(exc)[:100]}...). "
            f"Falling back to the built-in lexicon.",
            file=sys.stderr,
        )
        return None


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def classify(
    texts: Sequence[str],
    model_name: str = "SamLowe/roberta-base-go_emotions",
    batch_size: int = 64,
    device: int = 0,
) -> tuple[pd.DataFrame, list[str], str]:
    """Classify a list of texts. Returns (scores_df, emotion_cols, backend).

    Non-text placeholders should already be filtered by the caller; every text
    here is scored. scores_df has one column per emotion label + no index tricks.
    """
    clf = _load_hf(model_name, device=device)

    if clf is not None:
        # Discover the label set from the model itself (7-way or 27-way GoEmotions)
        results = clf(list(texts), batch_size=batch_size)
        # results: list (per text) of list of {label, score} dicts
        emotion_cols = sorted({d["label"] for d in results[0]})
        rows = []
        for res in results:
            row = {d["label"]: float(d["score"]) for d in res}
            rows.append(row)
        df = pd.DataFrame(rows, columns=emotion_cols).fillna(0.0)
        return df, emotion_cols, f"hf:{model_name}"

    # ---- lexicon fallback ---------------------------------------------------
    rows = [_lexicon_scores(t) for t in texts]
    df = pd.DataFrame(rows, columns=CANON_EMOTIONS)
    return df, list(CANON_EMOTIONS), "lexicon"


def classify_tone(
    in_path: str,
    out_path: str = "comments_toned.csv",
    text_col: str = "body",
    model_name: str = "SamLowe/roberta-base-go_emotions",
    batch_size: int = 64,
    device: int = 0,
) -> pd.DataFrame:
    """Read a comments CSV, classify tone, write + return the annotated frame.

    Adds one column per emotion (probability), plus `tone_top`,
    `tone_confidence`, and `tone_backend`. Dead/empty bodies get tone_top
    'no_text' and NaN scores. All original columns are preserved.
    """
    df = pd.read_csv(in_path)
    if text_col not in df.columns:
        raise ValueError(
            f"text column '{text_col}' not in {in_path}; columns are {list(df.columns)}"
        )

    bodies = df[text_col].fillna("").astype(str)
    is_dead = bodies.str.strip().isin(_DEAD_BODIES)
    live_mask = ~is_dead
    n_live = int(live_mask.sum())
    print(
        f"[info] {len(df):,} rows; {n_live:,} with text, "
        f"{len(df) - n_live:,} dead/empty (skipped).",
        file=sys.stderr,
    )

    live_texts = bodies[live_mask].tolist()
    scores_df, emotion_cols, backend = classify(
        live_texts, model_name=model_name, batch_size=batch_size, device=device
    )
    print(f"[info] tone backend: {backend} | labels: {emotion_cols}", file=sys.stderr)
    scores_df.index = df.index[live_mask]

    # top label + confidence for the live rows
    top_label = scores_df.idxmax(axis=1)
    top_conf = scores_df.max(axis=1)

    # ---- assemble output: original columns + emotion columns ---------------
    out = df.copy()
    for col in emotion_cols:
        out[col] = np.nan
        out.loc[live_mask, col] = scores_df[col].values

    out["tone_top"] = "no_text"
    out.loc[live_mask, "tone_top"] = top_label.values
    out["tone_confidence"] = np.nan
    out.loc[live_mask, "tone_confidence"] = top_conf.values
    out["tone_backend"] = backend

    out.to_csv(out_path, index=False)
    print(f"[done] wrote {len(out):,} rows -> {out_path}", file=sys.stderr)

    # quick console summary of the tone mix (live rows only)
    if n_live:
        mix = out.loc[live_mask, "tone_top"].value_counts(normalize=True)
        print("[summary] tone_top distribution (text rows):", file=sys.stderr)
        for label, frac in mix.items():
            print(f"    {label:10s} {frac:6.1%}", file=sys.stderr)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, help="input comments CSV")
    ap.add_argument("--out", dest="out_path", default="comments_toned.csv")
    ap.add_argument("--text-col", default="body", help="column holding comment text")
    ap.add_argument("--model", default="jSamLowe/roberta-base-go_emotions",
                    help="HF text-classification model (or GoEmotions for 27 labels)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    classify_tone(
        in_path=args.in_path,
        out_path=args.out_path,
        text_col=args.text_col,
        model_name=args.model,
        batch_size=args.batch_size,
        device=1,
    )


if __name__ == "__main__":
    main()