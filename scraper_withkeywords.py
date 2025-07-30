import datetime
import json
import praw
import pandas as pd

# Reddit API credentials
CLIENT_ID = json.load(open('config.json'))['CLIENT_ID']
CLIENT_SECRET = json.load(open('config.json'))['CLIENT_SECRET']
USER_AGENT = 'DebatesScraper by /u/starstrucksalad'

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Specify the subreddit target
subreddit_name = 'Discussion'
subreddit = reddit.subreddit(subreddit_name)
keyword = 'sexist'  # Change this to the keyword we want to search for

# Scrape posts from the subreddit
NUM_POSTS = 2000
submissions_data = []
all_comments_data = []

print(f"Scraping {NUM_POSTS} posts from r/{subreddit_name}...")

"""
You can choose sorting methods like 
'hot', 'new', 'top', 'controversial', 'rising', 'gilded'
For 'top' and 'controversial',
you can also specify a time filter
(e.g., 'day', 'week', 'month', 'year', 'all')
"""

for submission in subreddit.top(limit=NUM_POSTS, time_filter='all'):
    if keyword.lower() in submission.title.lower() or keyword.lower() in submission.selftext.lower():
        # Extract info from each submission
        submissions_data.append({
            'author': submission.author.name if submission.author else 'deleted',
            'created_utc': datetime.datetime.fromtimestamp(submission.created_utc),
            'id': submission.id,
            'num_comments': submission.num_comments,
            'permalink': submission.permalink,
            'score': submission.score, # upvote - downvote
            'selftext': submission.selftext,
            'title': submission.title,
            'url': submission.url
        })
        # Scrape top 10 comments for each submission
        submission.comments.replace_more(limit=0)
        comments_processed = 0
        for comment in submission.comments.list():
            if comments_processed >= 10:
                break
            if comment.author:
                comment_info = {
                    'author': comment.author.name,
                    'body': comment.body,
                    'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                    'comment_id': comment.id,
                    'parent_id': comment.parent_id,
                    'score': comment.score,
                    'submission_id': submission.id
                }
                all_comments_data.append(comment_info)
                comments_processed += 1
            else:
                # Skip deleted comments
                pass
            
# Convert dictionary to DataFrame
df_submissions = pd.DataFrame(submissions_data)
df_comments = pd.DataFrame(all_comments_data)

OUTPUT_SUBMISSIONS_FILE = f'scrapes/discussion-{keyword}_submissions.json'
OUTPUT_COMMENTS_FILE = f'scrapes/discussion-{keyword}_comments.json'

# Save DataFrame to JSON
df_submissions.to_json(OUTPUT_SUBMISSIONS_FILE, orient='records', lines=True)
print(f"Scraped {len(df_submissions)} submissions related to '{keyword}'.")
print(f"Data saved to {OUTPUT_SUBMISSIONS_FILE}")

df_comments.to_json(OUTPUT_COMMENTS_FILE, orient='records', lines=True)
print(f"Scraped {len(df_comments)} comments.")
print(f"Data saved to {OUTPUT_COMMENTS_FILE}")

# Check head
print("\nFirst 5 rows of Submissions Data:")
print(df_submissions.head())
print("\nFirst 5 rows of Comments Data:")
print(df_comments.head())