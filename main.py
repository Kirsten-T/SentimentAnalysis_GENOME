# This is a sample Python script.
from torch.cuda import device

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from reddit_parser import extract_to_csv
from link_posts_to_events import run_linking
from fetch_comments import fetch_comments
from classify_tone import classify_tone
import torch

def main():
    # links = run_linking(
    #     posts_path="data/posts/2024_01_reddit_submission.csv",
    #     events_path="data/events/EVENTS_2024_01.csv",
    #     out_path="data/posts_linked_events/2024_01_posts_linked_events.csv",
    #     days_after=28,
    #     top_k=6,
    # )
    # #
    # print(f"Linked {len(links)} pairs across {links['event_id'].nunique()} events.")

    # fetch_comments(
    #     links_path="data/posts_linked_events/2024_01_posts_linked_events.csv",
    #     dump_path="data/reddit_dump/RC_2024-01.zst",
    #     out_path="data/comments/2024_01_comments.csv"
    # )
    #
    classify_tone(in_path="data/comments/2024_01_comments.csv",
                  out_path="data/tone_comments/new_model_2024_01_tone_comments.csv",
                  text_col="body",
                  model_name="j-hartmann/emotion-english-distilroberta-base",
                  batch_size=124,
                  device=0)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.device)  # True => you have a usable GPU


    subs = ["worldnews", "geopolitics"]

    # extract_to_csv(
    #     "data/reddit_dump/RS_2024-01.zst",
    #     "data/posts/2024_01_reddit_submission.csv",
    #     start_date="2024-01-01",
    #     end_date="2024-01-31",
    #     keywords=["ukraine", "russia", "putin", "zelensky"],
    #     keyword_field="title",
    #     subreddits=subs,
    #     drop_deleted=True,
    # )

    main()






