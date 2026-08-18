from process_data.reddit_parser import extract_to_csv
from process_data.link_posts_to_events import run_linking
from process_data.fetch_comments import fetch_comments
from process_data.classify_tone import classify_tone
import torch

def process_data():
    subs = ["worldnews", "geopolitics"]

    """
     Extract reddit posts from the .zst file. Filter based on the date, listed subreddits and listed keywords.
    """
    extract_to_csv(
        "data/reddit_dump/RS_2024-01.zst",
        "data/posts/2024_01_reddit_submission.csv",
        start_date="2024-01-01",
        end_date="2024-01-31",
        keywords=["ukraine", "russia", "putin", "zelensky"],
        keyword_field="title",
        subreddits=subs,
        drop_deleted=True,
    )

    """
     Links reddit posts to GENOME Event data. Days_after is a threshold enforcing the maximum day delay from the GENOME event and the posted subreddit post.
    """
    links = run_linking(
        posts_path="data/posts/2024_01_reddit_submission.csv",
        events_path="data/events/EVENTS_2024_01.csv",
        out_path="data/posts_linked_events/2024_01_posts_linked_events.csv",
        days_after=28,
        top_k=6,
    )

    print(f"Linked {len(links)} pairs across {links['event_id'].nunique()} events.")

    """
     Per reddit post that has been linked to GENOME events, the comments are fetched.
    """
    fetch_comments(
        links_path="data/posts_linked_events/2024_01_posts_linked_events.csv",
        dump_path="data/reddit_dump/RC_2024-01.zst",
        out_path="data/comments/2024_01_comments.csv"
    )

    """
     Per comment, classify the tone.
    """
    classify_tone(in_path="data/comments/2024_01_comments.csv",
                  out_path="data/tone_comments/new_model_2024_01_tone_comments.csv",
                  text_col="body",
                  model_name="j-hartmann/emotion-english-distilroberta-base",
                  batch_size=124)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.device)  # True => you have a usable GPU

    process_data()






