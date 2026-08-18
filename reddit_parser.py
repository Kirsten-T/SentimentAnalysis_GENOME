"""
reddit_dump_parser.py
=====================
Stream-parse the pushshift / RaiderBDev Reddit dumps (the per-subreddit
`*_comments.zst` and `*_submissions.zst` files from Academic Torrents) into a
clean table for downstream sentiment analysis against GENOME events.

Why this exists
---------------
The dumps are:
  * zstandard-compressed  -> need streaming decompression (files are far too big
    to load into memory; a single busy subreddit-month can be many GB).
  * newline-delimited JSON -> one JSON object per line.
  * compressed with a LARGE window -> a default ZstdDecompressor raises
    "frame requires too much memory". You MUST pass max_window_size.

This module gives you three things:
  1. read_lines_zst()  - the robust streaming line reader (the fiddly bit).
  2. iter_records()    - parse + filter (date window, keywords, score) -> dicts.
  3. extract_to_csv()  - convenience: run a filter and write a CSV you can
                         load straight into pandas for the sentiment step.

Typical use for the GENOME / Ukraine case study:

    from reddit_dump_parser import extract_to_csv

    extract_to_csv(
        "worldnews_comments.zst",
        "worldnews_2022Q1.csv",
        start_date="2022-02-01",
        end_date="2022-04-30",
        keywords=["ukraine", "russia", "putin", "zelensky", "kyiv", "kremlin"],
        drop_deleted=True,
    )
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional, Sequence

import zstandard

# The dumps use a large compression window. This is the value the pushshift
# tooling uses; without it decompression fails on many files.
_MAX_WINDOW = 2 ** 31          # 2 GiB
_READ_CHUNK = 2 ** 27          # 128 MiB read chunks


# --------------------------------------------------------------------------- #
# 1. Streaming line reader
# --------------------------------------------------------------------------- #
def read_lines_zst(
    file_path: str,
    read_chunk: int = _READ_CHUNK,
    prefilter: Optional[Sequence[bytes]] = None,
) -> Iterator[str]:
    """Yield decoded text lines from a zstandard-compressed file, one at a time.

    Reads in binary chunks and splits on b'\\n' so that multi-byte UTF-8
    characters straddling a chunk boundary are never decoded mid-character:
    we only decode *complete* lines. The trailing partial line is carried over
    into the next chunk.

    prefilter : optional list of lowercased byte patterns. When given, a line is
        only decoded and yielded if its lowercased bytes contain at least one
        pattern. This lets a whole-site dump skip the expensive decode + JSON
        parse for the ~99.9% of lines that aren't in your target subreddits.
        It may admit rare false positives (a pattern appearing in body text);
        the caller re-checks authoritatively after JSON parsing.
    """
    with open(file_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=_MAX_WINDOW)
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(read_chunk)
                if not chunk:
                    break
                buffer += chunk
                # split off complete lines; keep the last (possibly partial) piece
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if prefilter is not None:
                        lb = line.lower()
                        if not any(p in lb for p in prefilter):
                            continue
                    yield line.decode("utf-8", errors="replace")
            if buffer:  # final line with no trailing newline
                if prefilter is not None:
                    lb = buffer.lower()
                    if not any(p in lb for p in prefilter):
                        return
                yield buffer.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# 2. Parse + filter -> normalised records
# --------------------------------------------------------------------------- #
_DELETED = {"[deleted]", "[removed]", ""}


def _to_epoch(value) -> Optional[float]:
    """created_utc is sometimes an int, sometimes a string. Coerce to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_to_epoch(date_str: str, end: bool = False) -> float:
    """'YYYY-MM-DD' -> UTC unix timestamp. end=True gives 23:59:59 of that day."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.timestamp()


def iter_records(
    file_path: str,
    subreddits: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keywords: Optional[Sequence[str]] = None,
    keyword_field: str = "both",   # "title", "text", or "both",
    min_score: Optional[int] = None,
    drop_deleted: bool = True,
    kind: str = "auto",
    limit: Optional[int] = None,
    progress_every: int = 1_000_000,
) -> Iterator[dict]:
    """Stream a dump and yield normalised records that pass all filters.

    Parameters
    ----------
    subreddits           : list of subreddit names (without 'r/'); keep only
                           records from these. Matching is case-insensitive.
                           Essential for the whole-site monthly dumps: it drives
                           a byte-level prefilter so most lines skip decode+JSON.
    start_date, end_date : 'YYYY-MM-DD', inclusive UTC day bounds.
    keywords             : list of terms; a record is kept if ANY term appears
                           (case-insensitive substring) in its text. Use this
                           for coarse entity matching (agent names, aliases).
    min_score            : drop records below this score.
    drop_deleted         : drop '[deleted]'/'[removed]'/empty-text records.
    kind                 : 'comment', 'submission', or 'auto' (infer from the
                           filename, falling back to the JSON fields).
    limit                : stop after yielding this many records (handy for
                           testing on a huge file).
    progress_every       : print a progress line to stderr every N raw lines.

    Yields
    ------
    dict with a unified schema:
        id, type, subreddit, created_utc, created_iso, author, score,
        num_comments, title, text, permalink
    where `text` is the field to run sentiment on
    (comment body, or submission selftext; combine with `title` as you like).
    """
    start_ts = _date_to_epoch(start_date) if start_date else None
    end_ts = _date_to_epoch(end_date, end=True) if end_date else None
    kw = [k.lower() for k in keywords] if keywords else None

    # subreddit set (authoritative, post-parse) + byte prefilter (fast, pre-parse)
    sub_set = {s.lower() for s in subreddits} if subreddits else None
    # Match the quoted subreddit *name* only, so we're robust to compact vs.
    # spaced JSON ("subreddit":"x" vs "subreddit": "x"). Rare false positives
    # (the quoted name appearing elsewhere) are rejected by the check below.
    prefilter = [f'"{s}"'.encode("utf-8") for s in sub_set] if sub_set else None

    if kind == "auto":
        low = file_path.lower()
        if "comment" in low:
            kind = "comment"
        elif "submission" in low:
            kind = "submission"
        # else stay 'auto' and infer per-line

    seen = 0
    kept = 0
    bad_json = 0

    for line in read_lines_zst(file_path, prefilter=prefilter):
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(f"  ...scanned {seen:,} lines, kept {kept:,}", file=sys.stderr)

        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_json += 1
            continue

        # ---- authoritative subreddit filter (catches prefilter collisions) -
        if sub_set is not None:
            if obj.get("subreddit", "").lower() not in sub_set:
                continue

        # ---- determine type ------------------------------------------------
        this_kind = kind
        if this_kind == "auto":
            this_kind = "submission" if "title" in obj else "comment"

        # ---- time filter (numeric compare, fast) ---------------------------
        ts = _to_epoch(obj.get("created_utc"))
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue

        # ---- pull the text -------------------------------------------------
        if this_kind == "comment":
            text = obj.get("body", "") or ""
            title = ""
            num_comments = ""
        else:
            text = obj.get("selftext", "") or ""
            title = obj.get("title", "") or ""
            num_comments = obj.get("num_comments", "")

        # ---- deleted / empty filter ---------------------------------------
        if drop_deleted:
            if this_kind == "comment":
                # a comment is dead if its body is empty / [deleted] / [removed]
                if text.strip() in _DELETED:
                    continue
            else:
                # a submission survives if EITHER the title or the selftext is
                # meaningful (link posts have a real title but empty selftext)
                title_ok = title.strip() and title.strip() not in _DELETED
                body_ok = text.strip() and text.strip() not in _DELETED
                if not (title_ok or body_ok):
                    continue

        # ---- score filter --------------------------------------------------
        score = obj.get("score", 0)
        if min_score is not None:
            try:
                if int(score) < min_score:
                    continue
            except (TypeError, ValueError):
                continue

        # ---- keyword / entity filter --------------------------------------
        if kw is not None:
            if keyword_field == "title":
                hay = title.lower()
            elif keyword_field == "text":
                hay = text.lower()
            else:  # "both"
                hay = (title + " " + text).lower()
            if not any(k in hay for k in kw):
                continue

        # ---- emit ----------------------------------------------------------
        rec = {
            "id": obj.get("id", ""),
            "type": this_kind,
            "subreddit": obj.get("subreddit", ""),
            "created_utc": int(ts),
            "created_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "author": obj.get("author", ""),
            "score": score,
            "num_comments": num_comments,
            "title": title.replace("\r", " "),
            "text": text.replace("\r", " "),
            "permalink": obj.get("permalink", ""),
        }
        kept += 1
        yield rec

        if limit is not None and kept >= limit:
            break

    print(
        f"[{file_path}] done: scanned {seen:,} lines, kept {kept:,}, "
        f"skipped {bad_json:,} malformed JSON lines.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# 3. Convenience writer
# --------------------------------------------------------------------------- #
_FIELDS = [
    "id", "type", "subreddit", "created_utc", "created_iso",
    "author", "score", "num_comments", "title", "text", "permalink",
]


def extract_to_csv(file_path: str, out_path: str, **filters) -> int:
    """Run iter_records() over a dump and write matching rows to a CSV.

    Returns the number of rows written. CSV is written with full quoting so
    that newlines / commas / emoji inside comment text are preserved and load
    cleanly with pandas.read_csv(out_path).
    """
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for rec in iter_records(file_path, **filters):
            writer.writerow(rec)
            n += 1
    print(f"Wrote {n:,} rows -> {out_path}", file=sys.stderr)
    return n


if __name__ == "__main__":
    # Tiny CLI: python reddit_dump_parser.py IN.zst OUT.csv 2022-02-01 2022-04-30
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(0)
    in_path, out_path = args[0], args[1]
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    extract_to_csv(in_path, out_path, start_date=start, end_date=end)