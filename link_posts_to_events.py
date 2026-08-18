"""
link_posts_to_events.py
=======================
Link Reddit posts (r/worldnews, r/geopolitics submissions) to a list of GENOME
events, where a real connection exists.

Approach
--------
Two signals decide whether a post is "about" an event:

  1. SEMANTIC SIMILARITY (the ML part).
     Each event and each post is turned into an embedding vector with a
     sentence-transformer model (default: all-MiniLM-L6-v2). Cosine similarity
     between a post and an event measures how much they talk about the same
     thing. Embeddings capture meaning, so "Kremlin orders troops across the
     border" matches an "ASSAULT" invasion event even with no shared words.

  2. A DATE WINDOW.
     All of these posts and events are about the same war, so similarity alone
     over-links: any Russia/Ukraine post looks a bit like any Russia/Ukraine
     event. Reddit discussion clusters in time around when something happened,
     so we only let a post be a *candidate* match for an event if the post was
     made within a window around the event date (a couple of days before to a
     week after, by default). This is what turns "topically related" into
     "actually about this event".

A post links to an event when it is inside that event's date window AND its
cosine similarity clears a threshold. A post may link to more than one event;
each (event, post) link is one output row.

Offline fallback
----------------
If the sentence-transformer model can't be downloaded (e.g. no internet), the
script automatically falls back to TF-IDF cosine similarity so it still runs.
TF-IDF is keyword-based and weaker than the neural model, so the default
threshold is adjusted for it. Prefer the neural model when you can reach the
internet once to cache it.

Usage
-----
    python link_posts_to_events.py \
        --posts 2022_02_reddit_submission.csv \
        --events EVENTS_2022_02.csv \
        --out 2022_02_posts_linked_events.csv

Key options: --threshold, --days-before, --days-after, --top-k, --model.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Default similarity thresholds, per backend. The neural and TF-IDF spaces have
# very different similarity scales, so each needs its own default. Used when the
# caller passes threshold=None.
DEFAULT_THRESHOLD = {"neural": 0.45, "tfidf": 0.12}


# --------------------------------------------------------------------------- #
# Text builders: turn a row into the string we embed
# --------------------------------------------------------------------------- #
def build_event_text(row: pd.Series) -> str:
    """Rich text for an event: the human summary plus the structured core.

    We fold in the summary (what happened, in prose), the core sentence
    (actor-VERB-recipient), and the normalized actors/locations so the vector
    is anchored on the right entities.
    """
    parts = [
        str(row.get("event_summary", "") or ""),
        str(row.get("core_sentence", "") or ""),
        str(row.get("actor_normalized_names", "") or ""),
        str(row.get("recipient_normalized_names", "") or ""),
        str(row.get("location_normalized_names", "") or ""),
    ]
    return " ".join(p for p in parts if p.strip())


def build_post_text(row: pd.Series) -> str:
    """Text for a post: title plus selftext (selftext is usually empty for
    link submissions, but include it when present)."""
    title = str(row.get("title", "") or "")
    body = str(row.get("text", "") or "")
    # 'text' can literally be the string 'nan' after a CSV round-trip; guard it
    if body.strip().lower() in {"nan", "none"}:
        body = ""
    return (title + " " + body).strip()


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def event_date_to_utc(date_str: str) -> pd.Timestamp:
    """'YYYY-MM-DD' -> tz-aware UTC midnight timestamp."""
    dt = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return pd.Timestamp(dt)


# --------------------------------------------------------------------------- #
# Embeddings: neural model with a TF-IDF fallback
# --------------------------------------------------------------------------- #
def embed_texts(event_texts, post_texts, model_name: str):
    """Return (event_vecs, post_vecs, backend) as L2-normalized float arrays.

    Tries sentence-transformers first; on any failure (no internet, missing
    package) falls back to a shared TF-IDF space fit on both corpora. With
    normalized vectors, cosine similarity is just a dot product.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        ev = model.encode(
            event_texts, normalize_embeddings=True, show_progress_bar=False
        )
        pv = model.encode(
            post_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=256,
        )
        return np.asarray(ev), np.asarray(pv), f"sentence-transformers:{model_name}"
    except Exception as exc:  # noqa: BLE001 - fallback is deliberate
        print(
            f"[warn] sentence-transformers unavailable ({exc}). "
            f"Falling back to TF-IDF.",
            file=sys.stderr,
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        # Fit on the union so both live in the same vocabulary space
        vec.fit(list(event_texts) + list(post_texts))
        ev = normalize(vec.transform(event_texts)).toarray()
        pv = normalize(vec.transform(post_texts)).toarray()
        return ev, pv, "tfidf"


# --------------------------------------------------------------------------- #
# Core linking
# --------------------------------------------------------------------------- #
def link(
    posts: pd.DataFrame,
    events: pd.DataFrame,
    threshold: float | None,
    days_before: int,
    days_after: int,
    top_k: int | None,
    model_name: str,
) -> pd.DataFrame:
    """Return a long-format DataFrame of (event, post) links passing filters.

    If threshold is None, a default appropriate to the backend that actually
    ran is chosen after embedding (see DEFAULT_THRESHOLD).
    """
    posts = posts.copy()
    events = events.copy()

    # ---- prepare text + dates -----------------------------------------------
    posts["_text"] = posts.apply(build_post_text, axis=1)
    posts["_dt"] = pd.to_datetime(posts["created_iso"], utc=True)
    posts = posts[posts["_text"].str.strip().astype(bool)].reset_index(drop=True)

    events["_text"] = events.apply(build_event_text, axis=1)
    events["_dt"] = events["event_date"].apply(event_date_to_utc)

    # ---- embed once ---------------------------------------------------------
    ev_vecs, post_vecs, backend = embed_texts(
        events["_text"].tolist(), posts["_text"].tolist(), model_name
    )
    print(f"[info] embedding backend: {backend}", file=sys.stderr)

    # Resolve the threshold default now that we know which backend actually ran.
    # (We can't decide this earlier from whether sentence-transformers *imports*:
    # the package can be installed yet still fall back to TF-IDF at runtime, e.g.
    # when the model can't be downloaded. The two spaces have very different
    # similarity scales, so the default must follow the real backend.)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD["tfidf" if backend == "tfidf" else "neural"]
        print(f"[info] auto threshold for {backend}: {threshold}", file=sys.stderr)

    # Similarity matrix: rows = events, cols = posts (normalized -> dot = cosine)
    sims = ev_vecs @ post_vecs.T  # shape (n_events, n_posts)

    post_days = posts["_dt"].values.astype("datetime64[ns]")

    rows = []
    before = np.timedelta64(days_before, "D")
    after = np.timedelta64(days_after, "D")

    for ei, ev in events.iterrows():
        ev_day = np.datetime64(ev["_dt"].tz_convert("UTC").tz_localize(None))
        # date-window mask: post within [event - before, event + after]
        delta = post_days - ev_day
        in_window = (delta >= -before) & (delta <= after)

        sim_row = sims[ei].copy()
        sim_row[~in_window] = -1.0  # exclude out-of-window posts

        # candidates above threshold
        cand_idx = np.where(sim_row >= threshold)[0]
        if cand_idx.size == 0:
            continue

        # rank by similarity, optionally cap at top_k
        cand_idx = cand_idx[np.argsort(-sim_row[cand_idx])]
        if top_k is not None:
            cand_idx = cand_idx[:top_k]

        for pi in cand_idx:
            p = posts.iloc[pi]
            day_diff = int((post_days[pi] - ev_day) / np.timedelta64(1, "D"))
            rows.append(
                {
                    "event_id": ev.get("id", ""),
                    "event_date": ev["event_date"],
                    "event_type": ev.get("event_type", ""),
                    "event_category": ev.get("category", ""),
                    "core_sentence": ev.get("core_sentence", ""),
                    "event_summary": str(ev.get("event_summary", ""))[:200],
                    "post_id": p.get("id", ""),
                    "subreddit": p.get("subreddit", ""),
                    "post_date": p["_dt"].isoformat(),
                    "day_diff": day_diff,
                    "similarity": round(float(sims[ei, pi]), 4),
                    "score": p.get("score", ""),
                    "title": p.get("title", ""),
                    "permalink": p.get("permalink", ""),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["event_id", "similarity"], ascending=[True, False]
        ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Callable entry point (import this from main.py)
# --------------------------------------------------------------------------- #
def run_linking(
    posts_path: str,
    events_path: str,
    out_path: str = "2022_02_posts_linked_events.csv",
    threshold: float | None = None,
    days_before: int = 2,
    days_after: int = 7,
    top_k: int | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Load the CSVs, link posts to events, write the result, and return it.

    This is the function to call from main.py. Every knob the CLI exposed is a
    keyword argument here with the same default.

    Parameters
    ----------
    posts_path   : path to the posts CSV (from the dump parser).
    events_path  : path to the events CSV.
    out_path     : where to write the links CSV.
    threshold    : min cosine similarity to link. If None, the default follows
                   the backend that actually ran (DEFAULT_THRESHOLD: 0.45 for the
                   neural model, 0.12 for the TF-IDF fallback).
    days_before  : how many days a post may precede the event and still match.
    days_after   : how many days after the event a post still counts.
    top_k        : keep only the top-K posts per event (None = all above
                   threshold).
    model_name   : sentence-transformers model name.

    Returns
    -------
    The links DataFrame (also written to out_path).
    """
    posts = pd.read_csv(posts_path)
    events = pd.read_csv(events_path)
    print(f"[info] {len(posts):,} posts, {len(events):,} events", file=sys.stderr)

    # threshold=None is resolved inside link(), once the real backend is known.
    links = link(
        posts,
        events,
        threshold=threshold,
        days_before=days_before,
        days_after=days_after,
        top_k=top_k,
        model_name=model_name,
    )

    links.to_csv(out_path, index=False)
    n_events_linked = links["event_id"].nunique() if not links.empty else 0
    print(
        f"[done] {len(links):,} links across {n_events_linked} events "
        f"-> {out_path}",
        file=sys.stderr,
    )
    return links


# --------------------------------------------------------------------------- #
# CLI (thin wrapper: parse args, delegate to run_linking)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posts", required=True, help="posts CSV (from the dump parser)")
    ap.add_argument("--events", required=True, help="events CSV")
    ap.add_argument("--out", default="2022_02_posts_linked_events.csv", help="output CSV")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="min cosine similarity to link (default: 0.45 neural / 0.12 tfidf)",
    )
    ap.add_argument("--days-before", type=int, default=2,
                    help="how many days a post may precede the event")
    ap.add_argument("--days-after", type=int, default=7,
                    help="how many days after the event a post still counts")
    ap.add_argument("--top-k", type=int, default=None,
                    help="keep only the top-K posts per event (default: all above threshold)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="sentence-transformers model name")
    args = ap.parse_args()

    run_linking(
        posts_path=args.posts,
        events_path=args.events,
        out_path=args.out,
        threshold=args.threshold,
        days_before=args.days_before,
        days_after=args.days_after,
        top_k=args.top_k,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()