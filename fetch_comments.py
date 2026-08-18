"""
fetch_comments_for_links.py
===========================
Given (a) the post->event links CSV and (b) the whole-site monthly *comment*
dump RC_YYYY-MM.zst, pull every comment that belongs to a linked post, and
carry the event linkage onto each comment.

How comments attach to posts
----------------------------
In the Reddit dumps a *comment* record has a `link_id` field of the form
`t3_<submission_id>` — the fullname of the submission it lives under (true for
top-level comments AND deep replies alike; `parent_id` is the immediate parent,
`link_id` is always the post). So:

    comments on post `t038lc`  <=>  comment.link_id == "t3_t038lc"

We take the set of linked post_ids from the links CSV, turn each into its
`t3_<id>` fullname, and keep comments whose link_id is in that set.

Why this is the right file
--------------------------
`RC_2022-02.zst` is the whole-site comments dump for Feb 2022 (RC = comments,
RS = submissions). It contains comments from every subreddit, so we filter it
down to the ones sitting under your linked posts.

Prefilter / performance
-----------------------
The dump is enormous, so before decoding+JSON-parsing a line we do a cheap
byte-level test: does the raw line contain any of our `t3_<id>` patterns? Only
candidates are parsed; the `link_id` check afterwards is authoritative and
rejects the rare false positive (an id string appearing elsewhere in the JSON).
This mirrors the subreddit prefilter in the dump parser. If your linked-post set
is very large (thousands), see the note in fetch_comments() about switching the
prefilter to subreddit names instead.

Output
------
One row per (comment, event). A comment under a post that links to several
events appears once per event, so the table lines up with 2022_02_posts_linked_events.csv
(which is one row per (post, event)). Dedupe on comment_id if you want unique
comments for a headcount.

Usage
-----
    python fetch_comments_for_links.py \
        --links 2022_02_posts_linked_events.csv \
        --dump RC_2022-02.zst \
        --out comments_for_links.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from reddit_parser import read_lines_zst

_DELETED = {"[deleted]", "[removed]", ""}

_OUT_FIELDS = [
    "comment_id", "post_id", "link_id", "parent_id", "subreddit",
    "comment_created_utc", "comment_created_iso", "author", "score", "body",
    "permalink",
    # carried from the links table:
    "event_id", "event_date", "event_type", "event_category", "core_sentence",
    "post_event_similarity", "post_title",
]


def _to_epoch(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_comments(
    links_path: str,
    dump_path: str,
    out_path: str = "comments_for_links.csv",
    drop_deleted: bool = True,
    min_score: Optional[int] = None,
    limit: Optional[int] = None,
    progress_every: int = 1_000_000,
) -> pd.DataFrame:
    """Stream the comment dump, keep comments on linked posts, annotate + write.

    Parameters
    ----------
    links_path   : 2022_02_posts_linked_events.csv (needs post_id + event columns).
    dump_path    : RC_YYYY-MM.zst whole-site comment dump.
    out_path     : output CSV path.
    drop_deleted : drop comments whose body is [deleted]/[removed]/empty.
    min_score    : if set, drop comments below this score.
    limit        : stop after keeping this many comment rows (for testing).
    progress_every : print progress to stderr every N scanned lines.

    Returns
    -------
    DataFrame of (comment, event) rows (also written to out_path).
    """
    links = pd.read_csv(links_path)

    # post_id -> list of link rows (a post may link to multiple events)
    # Build the lookup once; we attach event metadata per matched comment.
    link_cols = [
        "event_id", "event_date", "event_type", "event_category",
        "core_sentence", "similarity", "title",
    ]
    # Guard against columns that might be absent in a custom links file.
    link_cols = [c for c in link_cols if c in links.columns]
    events_by_post: dict[str, list[dict]] = {}
    for _, r in links.iterrows():
        events_by_post.setdefault(str(r["post_id"]), []).append(
            {c: r[c] for c in link_cols}
        )

    wanted_post_ids = set(events_by_post)
    # link_id fullnames we want, and the byte prefilter patterns.
    wanted_link_ids = {f"t3_{pid}" for pid in wanted_post_ids}
    prefilter = [lid.encode("utf-8").lower() for lid in wanted_link_ids]
    # NOTE: for a very large post set, swap the line above for a subreddit-name
    # prefilter, e.g. [f'"{s}"'.encode().lower() for s in links.subreddit.unique()],
    # then rely on the link_id check below. Fewer patterns => faster per-line test.

    print(
        f"[info] {len(wanted_post_ids):,} linked posts -> scanning {dump_path}",
        file=sys.stderr,
    )

    seen = kept = bad_json = 0
    out_rows: list[dict] = []

    for line in read_lines_zst(dump_path, prefilter=prefilter):
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(f"  ...scanned {seen:,} candidate lines, kept {kept:,}",
                  file=sys.stderr)

        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_json += 1
            continue

        # ---- authoritative post membership (catches prefilter collisions) --
        link_id = obj.get("link_id", "")
        if link_id not in wanted_link_ids:
            continue
        post_id = link_id[3:]  # strip 't3_'

        # ---- body / deleted filter ----------------------------------------
        body = obj.get("body", "") or ""
        if drop_deleted and body.strip() in _DELETED:
            continue

        # ---- score filter --------------------------------------------------
        score = obj.get("score", 0)
        if min_score is not None:
            try:
                if int(score) < min_score:
                    continue
            except (TypeError, ValueError):
                continue

        ts = _to_epoch(obj.get("created_utc"))
        iso = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            if ts is not None else ""
        )

        base = {
            "comment_id": obj.get("id", ""),
            "post_id": post_id,
            "link_id": link_id,
            "parent_id": obj.get("parent_id", ""),
            "subreddit": obj.get("subreddit", ""),
            "comment_created_utc": int(ts) if ts is not None else "",
            "comment_created_iso": iso,
            "author": obj.get("author", ""),
            "score": score,
            "body": body.replace("\r", " "),
            "permalink": obj.get("permalink", ""),
        }

        # ---- explode across every event this post links to -----------------
        for ev in events_by_post[post_id]:
            row = dict(base)
            row["event_id"] = ev.get("event_id", "")
            row["event_date"] = ev.get("event_date", "")
            row["event_type"] = ev.get("event_type", "")
            row["event_category"] = ev.get("event_category", "")
            row["core_sentence"] = ev.get("core_sentence", "")
            row["post_event_similarity"] = ev.get("similarity", "")
            row["post_title"] = ev.get("title", "")
            out_rows.append(row)

        kept += 1
        if limit is not None and kept >= limit:
            break

    print(
        f"[{dump_path}] done: scanned {seen:,} candidate lines, "
        f"kept {kept:,} comments ({len(out_rows):,} comment-event rows), "
        f"skipped {bad_json:,} malformed JSON lines.",
        file=sys.stderr,
    )

    df = pd.DataFrame(out_rows, columns=_OUT_FIELDS)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[done] wrote {len(df):,} rows -> {out_path}", file=sys.stderr)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--links", required=True, help="2022_02_posts_linked_events.csv")
    ap.add_argument("--dump", required=True, help="RC_YYYY-MM.zst comment dump")
    ap.add_argument("--out", default="comments_for_links.csv")
    ap.add_argument("--min-score", type=int, default=None)
    ap.add_argument("--keep-deleted", action="store_true",
                    help="keep [deleted]/[removed] comment bodies")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after keeping this many comments (testing)")
    args = ap.parse_args()

    fetch_comments(
        links_path=args.links,
        dump_path=args.dump,
        out_path=args.out,
        drop_deleted=not args.keep_deleted,
        min_score=args.min_score,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()