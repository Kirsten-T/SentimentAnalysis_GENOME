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
averaged sentiment score would destroy.

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

GENOME events are the unit of analysis. From the full GENOME dataset, we keep
only events where **Location country = Ukraine** and **Actor country = Ukraine or
Russia**.

This filter follows from the angle itself, not just from the case study. The
angle enriches an event with the public's reaction to it, so it only works for
events that people have widely discussed in the two selected subreddits.

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
to a file in this repository (see Section 6).

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

---

## 4. Key design choices, assumptions, and trade-offs


- **Streaming the compressed dumps.** The monthly dumps are far too large to hold
  in memory, so they are read as a stream with a byte-level prefilter (on post ID
  or subreddit name) that lets the ~99.9% of irrelevant lines skip decoding and
  JSON parsing.
- **Seven emotions, not positive/negative.** Fear, anger, and sadness are all
  "negative" but mean very different things for this angle, so a seven-emotion
  classifier is used. 
- **Upvote weighting.** Alongside raw emotion shares, an upvote-weighted variant
  measures which emotions the community *amplified*, not merely expressed.
- **Data assumptions / limitations.** Comments on posts made late in a month can
  spill into the next month's dump, and live-thread posts (running chats) can
  dominate comment volume while being only loosely tied to a single event.

---

## 5. What the angle makes visible



### The interactive dashboard

Each generated dashboard is a single self-contained HTML page (open it in any
browser — no server or build step, though it loads Chart.js from a CDN so it needs
internet the first time) that presents the results for a non-technical reader. It
is built by `visualize_tone.py` from one tone-classified comment CSV.

**Filtering.** Two dropdowns and a subreddit toggle set the scope, and every stat
and chart recomputes instantly in the browser:

- *Event* — all events, or one specific event. Choosing an event narrows the Post
  dropdown to that event's posts.
- *Post* — all posts in the current scope, or a single subreddit post.
- *Subreddit* — a toggle (all / r/worldnews / r/geopolitics) that composes with
  the two dropdowns, so you can, for example, isolate one community's reaction to
  one event.

Comments with no usable text (`[deleted]` / `[removed]`) are counted separately
and never fold into the emotion figures. The components are:

- **GENOME event card (top).** The selected event's full record straight from the
  GENOME data — date, event type and category (colour-coded conflict vs
  cooperation), coded intensity, article count, the event summary, the core
  sentence, the source quote, and the actor / recipient / third-party / location
  agents with their raw, normalised, and country values. This anchors the whole
  page in the event that generated the discussion, mirroring GENOME's own event
  view. When *all events* is selected it collapses to a short overview line.

- **Headline stat cards.** Six numbers for the current scope: comment volume
  (and how many were text-less), the share of negative comments
  (anger/fear/sadness/disgust), the dominant tone, mean model confidence, mean
  valence on a −1 (hostile) to +1 (warm) scale, and the **upvote skew** — the gap
  between upvote-weighted and raw valence, i.e. whether the crowd upvoted warmer
  or angrier comments than the average.

- **Emotion mix.** A ring showing the share of comments whose *dominant* emotion
  is each of the seven — the quickest read of "what did people mostly feel."

- **Emotion profile.** The *mean probability* of each emotion across all comments
  in scope — the event's tonal fingerprint. Unlike the mix ring, this uses the
  model's full soft scores, so a comment that is 45% anger / 40% fear contributes
  to both rather than being forced into one bucket; it reveals blends the ring
  hides.

- **Tone over time.** A stacked-area chart of each emotion's share across time
  bins, showing how the mood of the discussion evolves — for example surprise
  spiking at an event's onset, then giving way to anger or sadness. The bin width
  adapts to the time span in scope; a note appears if the span is too short to
  bin meaningfully.

- **What the crowd amplified.** Raw emotion share vs *upvote-weighted* share, side
  by side per emotion. Where the weighted bar exceeds the raw bar, the community
  endorsed that emotion by upvoting it — separating what people *expressed* from
  what the crowd *amplified*.

- **Valence distribution.** A histogram of per-comment valence across the −1…+1
  scale (red below zero, green above), showing whether the reaction is uniformly
  hostile, split into warm and cold camps, or clustered near neutral — detail that
  a single mean valence figure flattens.

- **Representative comments.** The highest-confidence comment for each emotion,
  shown as quote cards with the score, keeping every number tied to real human
  text so the reader can sanity-check what the model is picking up.
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

This ### Stage 2 — generate the dashboard (without running Stage 1)

Stage 1 is the stage that requires a lot of computational power. You
don't need to run it to see the dashboard: the tone-classified CSVs it produces
are already committed to this repository under `data/tone_comments/`, so you can
build the dashboard directly from them.

From a clean machine:

```bash
# 1. clone the repository
git clone <REPO_URL>
cd <REPO_NAME>

# 2. set up the environment (only pandas + numpy are needed for this stage)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install pandas numpy

# 3. build the dashboard from the pre-computed tone CSV
python visualize_tone.py \
    --in data/tone_comments/new_model_2022_02_tone_comments.csv \
    --out 2022_02_dashboard.html
```

Then open `2022_02_dashboard.html` in any browser. For the January 2024 period,
point `--in` at `data/tone_comments/new_model_2024_01_tone_comments.csv` instead.

The dashboard is a single self-contained HTML file (it loads Chart.js from a CDN,
so an internet connection is needed the first time you open it, but no server or
build step). You can also just open a pre-generated `2022_02_dashboard.html` /
`2024_01_dashboard.html` from the repository root directly, without running
step 3 at all.
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

