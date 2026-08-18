# Public Sentiment — a new GENOME domain angle

**A public-reaction layer for GENOME: for each geopolitical event, measure the
sentiment and tone of the general public's response on social media.**

GENOME currently navigates events through three domain angles: country,
organisation, and leader. All three describe *elite* behaviour: what states and
institutions do to one another. This project adds a fourth angle that describes
how the *governed* react to those actions, by attaching a measured emotional
signature to each GENOME event.

---

## 1. The domain angle and why it is interesting

### Background

The current GENOME platform revolves around three main entity types: countries,
organisations and leaders. Capturing world events along this axis can miss a
crucial factor in the understanding of current geopolitical events: the general
public, especially their fears and frustrations. Especially since the start of
social media, the sentiment among the general public can play a big role in the
unfolding of geopolitical events.

The Arab Spring, for example, was caused by widespread anger about the general
state of affairs, but the immediate trigger was the self-immolation of Tunisian
street vendor Mohamed Bouazizi in December 2010, which ignited mass protests
across the Middle East and North Africa. These protests were quickly organised
by social media. Another, more recent example is the "Cockroach" movement in
India. This movement started initially as a joke movement on social media, but
quickly led to large-scale protests on the street.

### The proposal

My proposal is to capture, per news event, the sentimentality and tone of the
reactions on social media to this news.

This fits GENOME's own ontology rather than sitting outside it. GENOME organises
events around *agents* that can act and be acted
upon. The public is exactly such an agent: a social group reacting to events
initiated by state actors. This angle therefore activates an agent type GENOME
already recognises but does not yet let an analyst navigate by.

### What questions it lets an analyst or policymaker answer

Because the angle measures public emotion on the same events GENOME already
codes, it makes several questions askable that no existing angle can pose:

- **Does public tone track, lag, or lead the coded severity of an event?** If
  public anger rises before the event stream escalates, sentiment is an
  early-warning signal; if it only reacts afterward, it is a barometer.
- **Where does public reaction diverge from the analyst-coded intensity?** A
  muted response to a severe event, or public fury at an event coded as minor,
  is the kind of mismatch that precedes protest, policy pressure, or escalation.
- **When a state escalates, does the affected public skew toward fear
  (deterrence) or anger (backlash)?** These imply opposite policy responses.
- **Does the emotional register of the same event differ between communities?**
  (e.g. a mass-audience subreddit vs a more analytical one.)
- **Two years into a conflict, has the public's emotional baseline shifted from
  acute fear/anger to fatigue and neutrality** — i.e. is salience decaying while
  the conflict continues?

The angle also captures *trajectory*, not just a snapshot: the sequence of
emotions (for example fear → anger, or anger → neutral) carries meaning that an
averaged sentiment score would destroy. Fear curdling into anger is a
mobilisation pattern; anger decaying into neutrality is disengagement.

---

## 2. Data

The analysis is scoped as a **case study** of the Ukraine–Russia conflict. This
conflict was chosen as it is still going on and is clearly documented, and it has
had large consequences for the geopolitical landscape, especially in Europe. Two
time points are examined: sentiment during the month of the Ukraine invasion
(**February 2022**), and sentiment two years later (**January 2024**).

A case study, rather than full coverage of every GENOME event, is a deliberate
response to a data constraint: the Reddit data is only available as whole-month
dumps, either all posts (~10GB) or all comments (~25GB) per month, and
filtering these first requires large storage space, which is currently
constrained. The case study therefore limits *how many* events are covered.
Within that scope, the analysis draws on two data sources.

### GENOME events

GENOME events are the unit of analysis. The relevant events are retrieved by
filtering on **Location country = Ukraine** and **Actor country = Ukraine or
Russia**. This filter is driven by the angle itself, not merely by the case
study: a public-sentiment layer can only enrich events for which public
discourse exists in the selected communities, so restricting to Ukraine/Russia
events in these subreddits' domain is the correct data-preparation step for the
angle.

### Reddit

Reddit was selected for this study due to its unique structure, which
facilitates detailed discussions within specific communities. Unlike other
social media platforms such as Twitter, where individual accounts drive content
visibility, Reddit is organised around subreddits, niche communities dedicated
to various topics. This structure allows for a more focused and in-depth
analysis of topics, making it easier to examine how extreme opinions are formed
and evolve within specific discourse spaces. Furthermore, Reddit's emphasis on
anonymity and the relatively unfiltered nature of many subreddits foster a space
where users can express polarised or extreme opinions that may not be as
prevalent in mainstream media.[Source](https://www.sciencedirect.com/science/article/pii/S2949719125000329)

Also, compared to other social media sites which require payment for accessing
the API or a long authentication process, scraped Reddit data was available on
HuggingFace. To fit the domain of the GENOME tool, the subreddits
**r/worldnews** and **r/geopolitics** were selected for analysis.




## 3. Method

The pipeline runs per time period (2022-02 and 2024-01). Each numbered step maps
to a script in this repository (see Section 6).

1. **Download and filter the Reddit posts.** Filter the monthly submissions dump
   for r/worldnews and r/geopolitics, keeping any post whose **title** contains
   one of the keywords *Ukraine, Russia, Putin, Zelensky*.
   → `reddit_parser.py`
2. **Retrieve the GENOME event data**, filtered on Location country = Ukraine and
   Actor country = Ukraine or Russia.
3. **Link posts to events.** Use the `all-MiniLM-L6-v2` model to embed the post
   titles and the GENOME event descriptions into a shared vector space, and
   compute similarity to link each event to the set of Reddit posts talking about
   it. A date window around each event is applied so that topical similarity is
   only accepted as a real link when the post falls near the event in time.
   → `link_posts_to_events.py`
4. **Fetch the comments.** Based on the ID of each linked Reddit post, filter the
   monthly Reddit comments dump on the post ID (`link_id`) to retrieve the
   comments belonging to those posts. → `fetch_comments.py`
5. **Classify tone.** Compute the tone and sentimentality of each comment using
   the `emotion-english-distilroberta-base` model (seven emotions: anger,
   disgust, fear, joy, neutral, sadness, surprise). → `classify_tone.py`

### Making it GENOME-centric: enrichment and analysis

The steps above produce comments linked to events. The analysis then **flips the
unit of analysis from the comment to the event** (`event_sentiment_analysis.py`):
each GENOME event is enriched with a public-emotion signature aggregated from its
linked comments, and that signature is analysed against GENOME's own structured
fields. Two kinds of preparation are done on the GENOME events:

- **Enrichment (external signal).** Each event gains a crowd-emotion signature:
  mean valence (mapped onto the same −1…+1 scale GENOME uses for intensity),
  dominant emotion, per-emotion means, engagement (summed upvotes), and a
  **reaction gap** (crowd valence − coded intensity).
- **Feature derivation (intrinsic).** The conflict/cooperation axis is made an
  explicit column derived from the event category.

---

## 4. Key design choices, assumptions, and trade-offs


- **Streaming the compressed dumps.** The monthly dumps are far too large to hold
  in memory, so they are read as a stream with a byte-level prefilter (on post ID
  or subreddit name) that lets the ~99.9% of irrelevant lines skip decoding and
  JSON parsing.
- **Seven emotions, not positive/negative.** Fear, anger, and sadness are all
  "negative" but mean very different things for this angle, so a seven-emotion
  classifier is used. A 27-label GoEmotions model is a drop-in for finer detail.
- **Event as the unit of analysis.** Sentiment is treated as a measured attribute
  *of a GENOME event*, which keeps the analysis anchored in GENOME data rather
  than in Reddit alone.
- **Upvote weighting.** Alongside raw emotion shares, an upvote-weighted variant
  measures which emotions the community *amplified*, not merely expressed.
- **Data assumptions / limitations.** Comments on posts made late in a month can
  spill into the next month's dump, and live-thread posts (running chats) can
  dominate comment volume while being only loosely tied to a single event.

---

## 5. What the angle makes visible

The angle produces two complementary outputs: an event-level analysis table for
rigour, and an interactive dashboard for exploration and communication.

### The interactive dashboard

`tone_dashboard.html` is a single self-contained page (open it in any browser; no
server or build step) that presents the results for a non-technical reader. Every
view can be filtered to **all events, a single event, or a single subreddit
post**, with an additional subreddit toggle, and all charts recompute instantly
in the browser. It is built by `visualize_tone.py` and has the following
components:

- **GENOME event card (top).** The selected event's full record straight from the
  GENOME data — date, event type and category, coded intensity, article count,
  summary, core sentence, source quote, and the actor / recipient / third-party /
  location agents. This anchors the whole page in the event that generated the
  discussion.
- **Headline stat cards.** Comment volume, share of negative comments, dominant
  tone, mean model confidence, mean valence, and the upvote skew (whether the
  community amplified warmer or angrier comments).
- **Emotion mix.** A ring of the dominant emotion per comment.
- **Emotion profile.** The mean probability of each emotion — the event's tonal
  fingerprint, using the full soft scores rather than only the top label.
- **Tone over time.** A stacked area of each emotion's share across time bins,
  showing how the mood of the discussion evolves.
- **What the crowd amplified.** Raw emotion share vs upvote-weighted share, per
  emotion.
- **Valence distribution.** A histogram of per-comment valence on the −1…+1 scale.
- **GENOME view.** The event-level analysis, visualised: one bubble per event with
  x = coded intensity, y = public mean valence, bubble size = upvote engagement,
  and colour = category. A diagonal marks where public tone equals the coded
  intensity, so the reaction gap reads off as distance from the line.
- **Representative comments.** The highest-confidence comment per emotion, keeping
  the numbers tied to real text.
---

## 6. How to run

### Dependencies

- Python 3.10+
- `pandas`, `numpy`, `zstandard`
- `sentence-transformers` (post→event linking)
- `transformers`, `torch` (tone classification; CUDA used automatically if available)
- `tqdm` (progress bars)
- The dashboard loads Chart.js from a CDN in the browser; no build step.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install pandas numpy zstandard sentence-transformers transformers torch tqdm
```

### The two stages

The project has two entry points:

- **`process_data.py`** — runs the data-processing pipeline (filter the Reddit
  dumps → link posts to GENOME events → fetch the comments on those posts →
  classify their tone) and writes the tone-classified comment CSVs that the
  dashboard reads.
- **`visualize_tone.py`** — reads one of those tone-classified CSVs and generates
  the interactive HTML dashboard.

### Stage 1 — process the data
- The reddit data files are available on HuggingFace: https://huggingface.co/datasets/peternasser99/reddit/tree/main
- Place the monthly Reddit dumps (`RS_YYYY-MM.zst`, `RC_YYYY-MM.zst`) and the
GENOME events CSV under `data/reddit_dump`, set the input paths and the time period near the
top of `process_data.py`, then run:

```bash
python process_data.py
```

This produces the tone-classified comments for that period, e.g.
`data/tone_comments/new_model_2022_02_tone_comments.csv`. 

### Stage 2 — generate the dashboard

The dashboard is built **from the output of Stage 1**, so that CSV must be present
in the `data/tone_comments/` folder first. If you ran `process_data.py` on another
machine (for example a GPU box or a server), **download its output CSV into
`data/tone_comments/`** before continuing. Then run the visualizer, pointing
`--in` at that file:

```bash
python visualize_tone.py \
    --in data/tone_comments/new_model_2022_02_tone_comments.csv \
    --out outputs/2022_02_dashboard.html
```


Open the resulting `.html` in any browser to explore the results interactively.

---

## 7. Repository structure

```
.
├── process_data/                        # data-processing pipeline modules
│   ├── reddit_parser.py                 # stream + filter the monthly Reddit dumps (.zst)
│   ├── link_posts_to_events.py          # MiniLM similarity + date window: posts -> events
│   ├── fetch_comments.py                # pull comments on the linked posts (by link_id)
│   └── classify_tone.py                 # 7-emotion tone classifier (DistilRoBERTa)
├── process_data.py                      # runs the full pipeline -> tone_comments CSVs
├── visualize_tone.py                    # generate the interactive dashboard (HTML) from a tone CSV
├── data/
│   ├── reddit_dump/                     # inputs: monthly RS_/RC_ dumps (.zst)
│   ├── events/                          # inputs: GENOME events CSVs (EVENTS_YYYY_MM.csv)
│   ├── posts/                           # filtered submissions (intermediate)
│   ├── posts_linked_events/             # posts linked to events (intermediate)
│   ├── comments/                        # fetched comments (intermediate)
│   └── tone_comments/                   # tone-classified CSVs — the dashboard's input
├── 2022_02_dashboard.html               # generated dashboard (Feb 2022)
├── 2024_01_dashboard.html               # generated dashboard (Jan 2024)
└── README.md
```

---

## 8. Challenges

- **Size of the data.** Reddit data at the scale needed is only available as
  whole-month dumps (~10GB posts, ~25GB comments); filtering requires
  substantial storage, which drove the decision to run a case study rather than
  cover every GENOME event.
- **Accessibility of the data.** Reliable, size-unlimited Reddit data is not
  available through the official API without payment or a long authentication
  process; the analysis depends on the archived HuggingFace dumps.
- **Model access and runtime.** The tone model must be downloaded before it can
  run offline, and classifying tens of thousands of comments on CPU is slow;
  the classifier auto-detects a GPU to mitigate this.
- **Accuracy of models.** The LLM models, used for linking GENOME events and subreddit posts, and for analysing the tone of the comments, are not always accurate. Also, within the scope of this assignment, it was not possible to benchmark and/or validate different models to test whether they perform well on our specific dataset.
- **Linkage precision.** Because all posts share one topic, semantic similarity
  alone over-links events to posts; a date window was needed to keep links
  meaningful.

---

## 9. Future work

- **Tag users based on geolocation** to understand how general sentiment differs
  per country.
- **Include other social media platforms:** Facebook, Twitter/X, Instagram,
  Bluesky.
- **Include non-English social media:** VK (Russia), Weibo and WeChat (China),
  to capture publics outside the English-speaking Reddit sphere.
- **Extend beyond the case study to the full, continuously-updated GENOME
  corpus,** computing the sentiment layer for every event where sufficient public
  discourse exists, and designing storage/streaming so the monthly dumps do not
  need to be held in full.
- **Deepen the intrinsic GENOME feature engineering,** e.g. normalising the
  actor/recipient country strings into structured fields and deriving
  actor-vs-recipient asymmetry features per event.

