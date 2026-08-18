"""
visualize_tone.py
=================
Turn a tone-classified comments CSV (output of classify_tone.py) into a single
self-contained, INTERACTIVE HTML dashboard, styled after the GENOME event
profile: dark canvas, stat cards, and a chart grid.

Data model it expects (one row per comment, exploded per linked event):
    events  ->  posts  ->  comments
each row carrying: event_id, post_id, subreddit, score, comment_created_iso,
body, the seven emotion probabilities, tone_top, tone_confidence, ...

Interactivity
-------------
Two dropdowns + a subreddit toggle let the reader choose the scope, and every
chart/stat recomputes in the browser (no re-run needed):
    Event = All   + Post = All        -> all events
    Event = X     + Post = All        -> one event
    Event = X     + Post = Y          -> one post
    (+ optional subreddit filter, composes with the above)

All per-comment rows are baked into the page as compact columnar arrays and
every aggregate (mix, profile, timeline, upvote-weighting, valence histogram,
exemplars) is computed client-side for the current scope.

Views: GENOME event card · stat cards · emotion mix donut · mean emotion profile
· tone over time (stacked share) · raw vs upvote-weighted share · valence
histogram · a strip of the highest-confidence comment per emotion.

The GENOME event card at the top shows the selected event's full record and
needs the events CSV, so pass --events to enable it.

Usage
-----
    python visualize_tone.py \
        --in comments_toned.csv \
        --events events.csv \
        --out tone_dashboard.html \
        --max-body 200            # truncate comment bodies baked into the page

Open the resulting HTML in any browser. For very large inputs, --max-rows will
randomly sample down to keep the file light.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Canonical emotions + a fixed, meaningful colour per emotion (warm = negative,
# cool/green = positive, grey = neutral). Kept stable so every chart agrees.
EMO_COLORS = {
    "anger":    "#e5484d",
    "disgust":  "#8e4ec6",
    "fear":     "#f76b15",
    "sadness":  "#3b82f6",
    "surprise": "#f5d90a",
    "joy":      "#30a46c",
    "neutral":  "#7c8695",
}
# Valence weight per emotion, on the same -1..+1 scale as GENOME intensity.
EMO_VALENCE = {
    "anger": -0.9, "disgust": -0.8, "fear": -0.7, "sadness": -0.6,
    "neutral": 0.0, "surprise": 0.15, "joy": 0.9,
}
NEG_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
_DEAD = "no_text"

# GENOME event fields carried into the event card (from the events CSV).
_DETAIL_FIELDS = [
    "event_summary", "source_quote", "core_sentence", "article_count",
    "article_links", "actor_raw_names", "actor_normalized_names",
    "actor_countries", "recipient_raw_names", "recipient_normalized_names",
    "recipient_countries", "third_party_raw_names",
    "third_party_normalized_names", "third_party_countries",
    "location_raw_names", "location_normalized_names", "location_countries",
]


def _emotion_cols(df: pd.DataFrame) -> list[str]:
    """Emotion probability columns present, in a stable order (known first)."""
    known = [e for e in EMO_COLORS if e in df.columns]
    tops = set(df.get("tone_top", pd.Series(dtype=str)).unique()) - {_DEAD}
    extra = [c for c in tops if c not in known and c in df.columns]
    return known + extra


def _require_toned(df: pd.DataFrame, in_path: str) -> None:
    """Fail early with a clear message if the input isn't a classify_tone output."""
    if "tone_top" in df.columns and any(e in df.columns for e in EMO_COLORS):
        return
    looks_raw = "body" in df.columns
    hint = (
        "This file hasn't been tone-classified yet — run classify_tone.py on it "
        "first, then point visualize_tone.py at that output:\n"
        f"    python classify_tone.py --in \"{in_path}\" --out comments_toned.csv\n"
        "    python visualize_tone.py --in comments_toned.csv"
        if looks_raw else
        "Expected a classify_tone.py output with a 'tone_top' column and emotion "
        "columns (anger, fear, ...); this file has neither."
    )
    raise ValueError(
        f"'{in_path}' is missing the tone columns.\n{hint}\n"
        f"(columns found: {list(df.columns)})"
    )


def build_payload(
    df: pd.DataFrame, max_body: int, event_intensity: Optional[float],
    events_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Bake per-comment columnar data + the index tables the UI needs."""
    df = df.copy()
    for c in ("event_id", "post_id", "subreddit", "tone_top"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    emo = _emotion_cols(df)

    # ---- GENOME event detail from the events CSV (optional) ----------------
    # Keyed by event id; used to fill the event card. intensity/category are
    # also surfaced on each event entry.
    ev_intensity, ev_category, ev_detail = {}, {}, {}
    if events_df is not None and "id" in events_df.columns:
        e2 = events_df.copy()
        e2["id"] = e2["id"].astype(str)
        for _, r in e2.iterrows():
            eid = r["id"]
            v = pd.to_numeric(r.get("intensity"), errors="coerce")
            if pd.notna(v):
                ev_intensity[eid] = float(v)
            if "category" in e2.columns and pd.notna(r.get("category")):
                ev_category[eid] = str(r["category"])
            det = {}
            for f in _DETAIL_FIELDS:
                if f in e2.columns:
                    val = r[f]
                    det[f] = "" if pd.isna(val) else str(val)
            ev_detail[eid] = det

    # ---- build index tables (events, posts, subreddits) from ALL rows ------
    def _first(series):
        return series.iloc[0] if len(series) else ""

    subs = sorted(df["subreddit"].dropna().unique().tolist()) if "subreddit" in df else []
    sub_ix = {s: i for i, s in enumerate(subs)}

    ev_meta = {}
    for eid, g in df.groupby("event_id"):
        # category: prefer the events file, fall back to the toned file's column
        cat = ev_category.get(eid)
        if cat is None and "event_category" in g:
            cat = str(_first(g.get("event_category", pd.Series([""]))))
        ev_meta[eid] = {
            "id": eid,
            "type": str(_first(g.get("event_type", pd.Series([""])))),
            "date": str(_first(g.get("event_date", pd.Series([""])))),
            "core": str(_first(g.get("core_sentence", pd.Series([""])))),
            "intensity": ev_intensity.get(eid),          # None if no events file
            "category": cat or "",
            "detail": ev_detail.get(eid, {}),            # full record if provided
        }
    # order events by date then id, for a tidy dropdown
    events = sorted(ev_meta.values(), key=lambda d: (d["date"], d["id"]))
    ev_ix = {e["id"]: i for i, e in enumerate(events)}

    posts = []
    post_ix = {}
    for pid, g in df.groupby("post_id"):
        post_ix[pid] = len(posts)
        evs = sorted({ev_ix[e] for e in g["event_id"].unique() if e in ev_ix})
        posts.append({
            "id": pid,
            "sub": sub_ix.get(str(_first(g.get("subreddit", pd.Series([""])))), -1),
            "title": str(_first(g.get("post_title", pd.Series([""]))))[:120],
            "evs": evs,
        })

    # ---- split live / dead --------------------------------------------------
    live = df[df["tone_top"] != _DEAD]
    dead = df[df["tone_top"] == _DEAD]

    def _idx_arrays(sub_df):
        return (
            [ev_ix.get(e, -1) for e in sub_df["event_id"]],
            [post_ix.get(p, -1) for p in sub_df["post_id"]],
            [sub_ix.get(s, -1) for s in sub_df.get("subreddit", pd.Series([""] * len(sub_df)))],
        )

    # optional sampling handled by caller; here we bake whatever we're given
    lei, lpi, lsi = _idx_arrays(live)
    dei, dpi, dsi = _idx_arrays(dead)

    top_ix = {e: i for i, e in enumerate(emo)}
    comments = {
        "ei": lei, "pi": lpi, "si": lsi,
        "sc": [int(x) if pd.notna(x) else 0 for x in live.get("score", pd.Series([0] * len(live)))],
        "t":  [int(pd.Timestamp(x).timestamp()) if pd.notna(x) else 0
               for x in pd.to_datetime(live.get("comment_created_iso"), utc=True, errors="coerce")],
        "top": [top_ix.get(t, 0) for t in live["tone_top"]],
        "cf": [round(float(x), 3) if pd.notna(x) else 0.0
               for x in live.get("tone_confidence", pd.Series([0.0] * len(live)))],
        "pr": np.round(live[emo].fillna(0.0).to_numpy(), 3).tolist(),
        "b":  [str(x).replace("\r", " ").replace("\n", " ")[:max_body]
               for x in live.get("body", pd.Series([""] * len(live)))],
    }
    dead_cols = {"ei": dei, "pi": dpi, "si": dsi}

    return {
        "emotions": emo,
        "colors": [EMO_COLORS.get(e, "#7c8695") for e in emo],
        "valence": [EMO_VALENCE.get(e, 0.0) for e in emo],
        "negIdx": [emo.index(e) for e in NEG_EMOTIONS if e in emo],
        "events": events,
        "posts": posts,
        "subs": subs,
        "comments": comments,
        "dead": dead_cols,
        "refIntensity": event_intensity,
        "hasEventData": any(e.get("detail") for e in events),
        "backend": str(df["tone_backend"].iloc[0]) if "tone_backend" in df else "",
    }


def visualize_tone(
    in_path: str,
    out_path: str = "tone_dashboard_new_data.html",
    event_intensity: Optional[float] = None,
    max_body: int = 200,
    max_rows: Optional[int] = None,
    events_path: Optional[str] = None,
) -> str:
    df = pd.read_csv(in_path)
    _require_toned(df, in_path)

    if max_rows is not None and len(df) > max_rows:
        # keep all dead rows small; sample live rows to hit the budget
        live = df[df["tone_top"] != _DEAD]
        dead = df[df["tone_top"] == _DEAD]
        live = live.sample(n=min(max_rows, len(live)), random_state=0)
        df = pd.concat([live, dead], ignore_index=True)
        print(f"[note] sampled to {len(live):,} live rows (--max-rows).", file=sys.stderr)

    events_df = pd.read_csv(events_path) if events_path else None
    payload = build_payload(df, max_body=max_body, event_intensity=event_intensity,
                            events_df=events_df)
    out = _TEMPLATE.replace("__DATA__", json.dumps(payload))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)

    n_live = len(payload["comments"]["ei"])
    print(f"[done] {n_live:,} classified comments, {len(payload['events'])} events, "
          f"{len(payload['posts'])} posts -> {out_path}", file=sys.stderr)
    if not payload["hasEventData"]:
        print("[note] no event detail loaded — pass --events EVENTS.csv to show the "
              "GENOME event card.", file=sys.stderr)
    if payload["backend"] == "lexicon":
        print("[note] backend=lexicon: numbers are illustrative; rerun the "
              "classifier with the neural model for real tone.", file=sys.stderr)
    return out_path


# --------------------------------------------------------------------------- #
# HTML template. All aggregation happens client-side so filters are instant.
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comment tone · dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0c1626; --panel:#12213a; --panel2:#0f1c31; --line:#22344f;
    --ink:#e8eef7; --muted:#8698b1; --accent:#4ea1ff;
    --font:"Inter",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);-webkit-font-smoothing:antialiased}
  .wrap{max-width:1240px;margin:0 auto;padding:24px 20px 60px}
  .head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  .head h1{font-size:20px;margin:0;letter-spacing:.2px}
  .head .sub{color:var(--muted);font-size:13px}
  /* GENOME event detail card */
  .evcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:18px 0}
  .evcard .top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px}
  .evcard .date{font-size:20px;font-weight:700}
  .evcard .etag{font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;letter-spacing:.4px;color:#fff}
  .evcard .ctag{font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:.4px;background:transparent;border:1px solid var(--line);color:var(--muted)}
  .evcard .meta{color:var(--muted);font-size:12px}
  .evcard .sec{margin-top:14px}
  .evcard .sec .lab{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:3px}
  .evcard .sec .val{font-size:13px;color:#d8e2f0;line-height:1.5}
  .evcard .val.srcq{font-style:italic;color:#b9c6da}
  .agrid{display:grid;grid-template-columns:1fr 1fr;gap:14px 28px;margin-top:14px}
  .agent .role{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:4px}
  .agent .kv{font-size:12px;color:#cdd8e8;line-height:1.5}
  .agent .kv b{color:var(--muted);font-weight:600}
  .evprompt{color:var(--muted);font-size:13px}
  /* controls */
  .controls{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;
    background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:18px 0}
  .ctl{display:flex;flex-direction:column;gap:5px}
  .ctl label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  .ctl select{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
    border-radius:8px;padding:8px 10px;font-size:13px;font-family:inherit;min-width:220px;max-width:420px}
  .pills{display:flex;gap:6px}
  .pill{background:var(--panel2);color:var(--muted);border:1px solid var(--line);
    border-radius:20px;padding:6px 12px;font-size:12px;cursor:pointer}
  .pill.on{background:var(--accent);color:#04101f;border-color:var(--accent);font-weight:600}
  .scope{margin-left:auto;color:var(--muted);font-size:12px;align-self:center;text-align:right}
  .scope b{color:var(--ink)}
  /* cards */
  .cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-height:74px}
  .card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  .card .v{font-size:23px;font-weight:700;margin-top:6px}
  .card .n{color:var(--muted);font-size:11px;margin-top:2px}
  /* grid */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;min-height:120px}
  .panel h2{font-size:13px;font-weight:600;margin:0 0 2px;letter-spacing:.3px}
  .panel .hint{color:var(--muted);font-size:11px;margin-bottom:10px}
  .full{grid-column:1 / -1}
  canvas{max-height:300px}
  .ex{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
  .quote{background:var(--panel2);border:1px solid var(--line);border-left-width:3px;border-radius:10px;padding:11px 13px}
  .quote .lab{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
  .quote .body{font-size:13px;color:#d3ddec;margin-top:6px;line-height:1.45}
  .quote .meta{font-size:11px;color:var(--muted);margin-top:8px}
  .empty{color:var(--muted);text-align:center;padding:40px;font-size:14px}
  @media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.scope{margin-left:0}.agrid{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
  <div class="head"><h1>Comment tone</h1><span class="sub">emotional reaction across events · posts · comments</span></div>

  <div class="evcard" id="evcard"></div>

  <div class="controls">
    <div class="ctl"><label>Event</label><select id="fEvent"></select></div>
    <div class="ctl"><label>Post</label><select id="fPost"></select></div>
    <div class="ctl"><label>Subreddit</label><div class="pills" id="fSubs"></div></div>
    <div class="scope" id="scope"></div>
  </div>

  <div class="cards" id="cards"></div>
  <div class="grid" id="grid">
    <div class="panel"><h2>Emotion mix</h2><div class="hint">Dominant emotion per comment</div><canvas id="mix"></canvas></div>
    <div class="panel"><h2>Emotion profile</h2><div class="hint">Mean probability across comments — the tonal fingerprint</div><canvas id="profile"></canvas></div>
    <div class="panel full"><h2>Tone over time</h2><div class="hint" id="timehint">Share of dominant emotion per time bin</div><canvas id="time"></canvas></div>
    <div class="panel"><h2>What the crowd amplified</h2><div class="hint">Raw share vs upvote-weighted share per emotion</div><canvas id="weight"></canvas></div>
    <div class="panel"><h2>Valence distribution</h2><div class="hint" id="valhint">Per-comment valence on the −1…+1 scale</div><canvas id="hist"></canvas></div>
    <div class="panel full"><h2>Representative comments</h2><div class="hint">Highest-confidence comment per emotion, in scope</div><div class="ex" id="ex"></div></div>
  </div>
  <div class="empty" id="empty" style="display:none">No comments match this scope.</div>
</div>
<script>
const D = __DATA__;
const C = D.comments, EMO = D.emotions, COL = D.colors, VAL = D.valence;
const CAP = s => s.charAt(0).toUpperCase()+s.slice(1);
const N = C.ei.length;
Chart.defaults.color = "#8698b1";
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.borderColor = "rgba(255,255,255,0.05)";

// hide the event card entirely when no events CSV was supplied
if(!D.hasEventData){ const ec=document.getElementById("evcard"); if(ec) ec.style.display="none"; }

// ---------- state ----------
let selEvent = -1;   // -1 = all
let selPost  = -1;   // -1 = all
let selSub   = -1;   // -1 = all

// ---------- populate controls ----------
const fEvent = document.getElementById("fEvent");
const fPost  = document.getElementById("fPost");
function fillEvents(){
  fEvent.innerHTML = `<option value="-1">All events (${D.events.length})</option>` +
    D.events.map((e,i)=>`<option value="${i}">${(e.type||"event").toUpperCase()} · ${e.date} · ${(e.core||"").slice(0,60)}</option>`).join("");
}
function fillPosts(){
  const inScope = D.posts.map((p,i)=>({p,i}))
    .filter(o => selEvent<0 || o.p.evs.includes(selEvent));
  fPost.innerHTML = `<option value="-1">All posts (${inScope.length})</option>` +
    inScope.map(o=>{const sub = o.p.sub>=0?D.subs[o.p.sub]:""; 
      return `<option value="${o.i}">${sub} · ${(o.p.title||o.p.id).slice(0,70)}</option>`;}).join("");
  selPost = -1;
}
const fSubs = document.getElementById("fSubs");
fSubs.innerHTML = `<div class="pill on" data-s="-1">All</div>` +
  D.subs.map((s,i)=>`<div class="pill" data-s="${i}">${s}</div>`).join("");
fSubs.querySelectorAll(".pill").forEach(el=>el.onclick=()=>{
  selSub = parseInt(el.dataset.s);
  fSubs.querySelectorAll(".pill").forEach(p=>p.classList.remove("on"));
  el.classList.add("on"); render();
});
fEvent.onchange = ()=>{ selEvent = parseInt(fEvent.value); fillPosts(); render(); };
fPost.onchange  = ()=>{ selPost  = parseInt(fPost.value); render(); };
fillEvents(); fillPosts();

// ---------- GENOME event card ----------
function renderEventCard(){
  if(!D.hasEventData) return;
  const el = document.getElementById("evcard");
  if(selEvent<0){
    el.innerHTML = `<div class="evprompt">Showing <b style="color:var(--ink)">all ${D.events.length} events</b>. `
      + `Pick an event from the dropdown to see its full GENOME record here.</div>`;
    return;
  }
  const e = D.events[selEvent], d = e.detail || {};
  const etagColor = /CONFLICT/i.test(e.category) ? "#e5484d" : /COOP/i.test(e.category) ? "#30a46c" : "#7c8695";
  const esc = s => (s||"").replace(/</g,"&lt;");
  const sec = (lab,val,cls="") => val ? `<div class="sec"><div class="lab">${lab}</div><div class="val ${cls}">${esc(val)}</div></div>` : "";
  const agent = (role,raw,norm,ctry) => {
    if(!raw && !norm && !ctry) return "";
    const line=(k,v)=>`<div class="kv"><b>${k}:</b> ${esc(v)||"—"}</div>`;
    return `<div class="agent"><div class="role">${role}</div>${line("Raw",raw)}${line("Norm",norm)}${line("Countries",ctry)}</div>`;
  };
  const artLink = (d.article_links && /^https?:\/\//.test(d.article_links))
    ? `<a href="${d.article_links.split(/[;, ]/)[0]}" target="_blank" style="color:var(--accent);font-size:12px">View article ↗</a>` : "";
  el.innerHTML = `
    <div class="top">
      <span class="date">${esc(e.date)}</span>
      <span class="etag" style="background:${etagColor}">${(e.type||"EVENT").toUpperCase()}</span>
      ${e.category?`<span class="ctag">${esc(e.category)}</span>`:""}
      <span class="meta">Intensity ${e.intensity??"—"}${d.article_count?` · ${d.article_count} linked article${d.article_count=="1"?"":"s"}`:""}</span>
      <span style="margin-left:auto">${artLink}</span>
    </div>
    ${sec("Summary", d.event_summary)}
    ${sec("Core sentence", e.core || d.core_sentence)}
    ${sec("Source quote", d.source_quote, "srcq")}
    <div class="agrid">
      ${agent("Actor", d.actor_raw_names, d.actor_normalized_names, d.actor_countries)}
      ${agent("Recipient", d.recipient_raw_names, d.recipient_normalized_names, d.recipient_countries)}
      ${agent("Third party", d.third_party_raw_names, d.third_party_normalized_names, d.third_party_countries)}
      ${agent("Location", d.location_raw_names, d.location_normalized_names, d.location_countries)}
    </div>`;
}

// ---------- filtering ----------
function filteredIdx(){
  const out = [];
  for(let i=0;i<N;i++){
    if(selEvent>=0 && C.ei[i]!==selEvent) continue;
    if(selPost >=0 && C.pi[i]!==selPost)  continue;
    if(selSub  >=0 && C.si[i]!==selSub)   continue;
    out.push(i);
  }
  return out;
}
function deadCount(){
  let n=0; const d=D.dead;
  for(let i=0;i<d.ei.length;i++){
    if(selEvent>=0 && d.ei[i]!==selEvent) continue;
    if(selPost >=0 && d.pi[i]!==selPost)  continue;
    if(selSub  >=0 && d.si[i]!==selSub)   continue;
    n++;
  }
  return n;
}

// ---------- charts (created once, updated on render) ----------
const col = i => COL[i];
const mixCh = new Chart(mix,{type:"doughnut",
  data:{labels:EMO.map(CAP),datasets:[{data:EMO.map(_=>0),backgroundColor:COL,borderColor:"#0c1626",borderWidth:2}]},
  options:{cutout:"62%",plugins:{legend:{position:"right",labels:{boxWidth:12,padding:8}}}}});
const profCh = new Chart(profile,{type:"bar",
  data:{labels:EMO.map(CAP),datasets:[{data:EMO.map(_=>0),backgroundColor:COL}]},
  options:{indexAxis:"y",plugins:{legend:{display:false}},
    scales:{x:{beginAtZero:true,max:1,grid:{color:"rgba(255,255,255,.05)"}},y:{grid:{display:false}}}}});
const timeCh = new Chart(time,{type:"line",
  data:{labels:[],datasets:EMO.map((e,i)=>({label:CAP(e),data:[],backgroundColor:COL[i],borderColor:COL[i],
    fill:true,stack:"s",pointRadius:0,tension:.3,borderWidth:1}))},
  options:{plugins:{legend:{labels:{boxWidth:12}}},
    scales:{y:{stacked:true,max:1,grid:{color:"rgba(255,255,255,.05)"}},x:{stacked:true,grid:{display:false},ticks:{maxTicksLimit:10}}}}});
const wCh = new Chart(weight,{type:"bar",
  data:{labels:EMO.map(CAP),datasets:[
    {label:"Raw share",data:EMO.map(_=>0),backgroundColor:"#3a4c68"},
    {label:"Upvote-weighted",data:EMO.map(_=>0),backgroundColor:COL}]},
  options:{plugins:{legend:{labels:{boxWidth:12}}},
    scales:{y:{beginAtZero:true,grid:{color:"rgba(255,255,255,.05)"}},x:{grid:{display:false}}}}});
const HLAB = Array.from({length:20},(_,i)=>(-1+i*0.1).toFixed(1));
const HCOL = HLAB.map(l=>{const v=parseFloat(l)+0.05; return v<-0.05?"#e5484d":v>0.05?"#30a46c":"#7c8695";});
const histCh = new Chart(hist,{type:"bar",
  data:{labels:HLAB,datasets:[{data:HLAB.map(_=>0),backgroundColor:HCOL}]},
  options:{plugins:{legend:{display:false}},
    scales:{y:{beginAtZero:true,grid:{color:"rgba(255,255,255,.05)"}},x:{grid:{display:false},ticks:{maxTicksLimit:11}}}}});

function fmtT(t){const d=new Date(t*1000);const p=n=>String(n).padStart(2,"0");
  return `${p(d.getUTCMonth()+1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;}

// ---------- render ----------
function render(){
  renderEventCard();
  const idx = filteredIdx();
  const nDead = deadCount();
  const grid = document.getElementById("grid");
  const empty = document.getElementById("empty");
  if(idx.length===0){ grid.style.display="none"; empty.style.display="block";
    document.getElementById("cards").innerHTML=""; updateScope(0,nDead); return; }
  grid.style.display=""; empty.style.display="none";

  const E = EMO.length;
  const mixCnt = new Array(E).fill(0);
  const profSum = new Array(E).fill(0);
  const wRawSum = new Array(E).fill(0), wWtSum = new Array(E).fill(0);
  let wTot = 0, confSum = 0, valSum = 0, valWSum = 0, negCnt = 0;
  const valById = new Float64Array(idx.length);
  const exBest = new Array(E).fill(null);

  for(let k=0;k<idx.length;k++){
    const i = idx[k];
    const pr = C.pr[i], top = C.top[i];
    mixCnt[top]++;
    const wt = Math.max(C.sc[i],0)+1; wTot += wt;
    let v=0;
    for(let e=0;e<E;e++){ profSum[e]+=pr[e]; wRawSum[e]+=pr[e]; wWtSum[e]+=pr[e]*wt; v+=pr[e]*VAL[e]; }
    valById[k]=v; valSum+=v; valWSum+=v*wt; confSum+=C.cf[i];
    if(D.negIdx.includes(top)) negCnt++;
    const b = exBest[top];
    if(!b || C.cf[i]>b.cf) exBest[top] = {cf:C.cf[i], sc:C.sc[i], body:C.b[i]};
  }
  const n = idx.length;

  // cards
  const dominant = mixCnt.indexOf(Math.max(...mixCnt));
  const rawVal = valSum/n, wtVal = valWSum/wTot, skew = wtVal-rawVal;
  const cards = [
    ["Comments", n.toLocaleString(), nDead.toLocaleString()+" without text"],
    ["Negative", (negCnt/n*100).toFixed(1)+"%", "anger/fear/sadness/disgust"],
    ["Dominant tone", CAP(EMO[dominant]), "most common label"],
    ["Mean confidence", (confSum/n).toFixed(2), "model certainty"],
    ["Mean valence", rawVal.toFixed(2), "−1 hostile … +1 warm"],
    ["Upvote skew", (skew>=0?"+":"")+skew.toFixed(2), skew>=0?"crowd warmer":"crowd angrier"],
  ];
  document.getElementById("cards").innerHTML = cards.map(c=>
    `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="n">${c[2]}</div></div>`).join("");

  // mix + profile
  mixCh.data.datasets[0].data = mixCnt.map(c=>c/n); mixCh.update();
  profCh.data.datasets[0].data = profSum.map(s=>s/n); profCh.update();

  // weighted
  wCh.data.datasets[0].data = wRawSum.map(s=>s/n);
  wCh.data.datasets[1].data = wWtSum.map(s=>s/wTot); wCh.update();

  // valence histogram
  const hc = new Array(20).fill(0);
  for(let k=0;k<n;k++){ let b=Math.floor((valById[k]+1)/0.1); b=Math.max(0,Math.min(19,b)); hc[b]++; }
  histCh.data.datasets[0].data = hc; histCh.update();
  document.getElementById("valhint").textContent =
    (D.refIntensity!==null && D.refIntensity!==undefined)
      ? `Per-comment valence (−1…+1). Event coded intensity: ${D.refIntensity}`
      : "Per-comment valence on the −1…+1 scale (warm = positive)";

  // tone over time — bin the scoped comments
  let tmin=Infinity,tmax=-Infinity;
  for(const i of idx){ if(C.t[i]){ tmin=Math.min(tmin,C.t[i]); tmax=Math.max(tmax,C.t[i]); } }
  const labels=[], series=EMO.map(()=>[]);
  if(isFinite(tmin) && tmax>tmin){
    const spanMin=(tmax-tmin)/60, binMin=Math.max(1,Math.round(spanMin/24)), binSec=binMin*60;
    const bins=new Map();
    for(const i of idx){ if(!C.t[i]) continue;
      const b=Math.floor(C.t[i]/binSec)*binSec;
      if(!bins.has(b)) bins.set(b, new Array(E).fill(0));
      bins.get(b)[C.top[i]]++; }
    [...bins.keys()].sort((a,b)=>a-b).forEach(b=>{
      const arr=bins.get(b); const tot=arr.reduce((x,y)=>x+y,0);
      labels.push(fmtT(b)); arr.forEach((c,e)=>series[e].push(tot?c/tot:0)); });
    document.getElementById("timehint").textContent = `Share of dominant emotion per ${binMin}-min bin`;
  } else {
    document.getElementById("timehint").textContent = "Not enough time span in scope to bin";
  }
  timeCh.data.labels = labels;
  timeCh.data.datasets.forEach((ds,e)=>ds.data=series[e]); timeCh.update();

  // exemplars
  document.getElementById("ex").innerHTML = exBest.map((x,e)=> x?
    `<div class="quote" style="border-left-color:${col(e)}">
       <div class="lab" style="color:${col(e)}">${CAP(EMO[e])} · ${(x.cf*100).toFixed(0)}%</div>
       <div class="body">${x.body.replace(/</g,"&lt;")}</div>
       <div class="meta">▲ ${x.sc}</div></div>`:"").join("");

  updateScope(n, nDead);
}
function updateScope(n,nDead){
  const ev = selEvent<0 ? "All events" : (D.events[selEvent].type||"event").toUpperCase()+" · "+D.events[selEvent].date;
  const po = selPost<0 ? "all posts" : "1 post";
  const su = selSub<0 ? "" : " · r/"+D.subs[selSub];
  document.getElementById("scope").innerHTML =
    `<b>${n.toLocaleString()}</b> comments in scope${su}<br>${ev} · ${po}`;
}
render();
</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, help="toned comments CSV")
    ap.add_argument("--out", dest="out_path", default="tone_dashboard.html")
    ap.add_argument("--events", dest="events_path", default=None,
                    help="GENOME events CSV (enables the event detail card)")
    ap.add_argument("--event-intensity", type=float, default=None,
                    help="optional GENOME coded intensity to show on the valence panel")
    ap.add_argument("--max-body", type=int, default=200,
                    help="truncate comment bodies baked into the page (chars)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="sample live rows down to this many to keep the file light")
    args = ap.parse_args()

    visualize_tone(
        in_path=args.in_path,
        out_path=args.out_path,
        event_intensity=args.event_intensity,
        max_body=args.max_body,
        max_rows=args.max_rows,
        events_path=args.events_path,
    )

    #visualize_tone(in_path="data/tone_comments/new_model_2022_02_tone_comments.csv", events_path="data/events/EVENTS_2022_02.csv",out_path="2022_02_dashboard.html")


if __name__ == "__main__":
    main()