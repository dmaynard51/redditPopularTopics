import praw
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

nltk.download("punkt")
nltk.download("stopwords")

# Replace the following placeholders with your actual credentials
client_id = "dEsJH7mLkCq9Hw575aHANw"
client_secret = "0v-78wQMQDL3YcJ1qNHPhcDtA2wSEQ"
user_agent = "python:my_reddit_scraper:v1.0 (by /u/Least-Result-45)"

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

subreddit_name = "csMajors"
subreddit = reddit.subreddit(subreddit_name)

# Fetch the top 100 posts from the subreddit
top_posts = list(subreddit.top(limit=100))

# Tokenize and clean the titles
stop_words = set(stopwords.words("english"))
titles = []
for post in top_posts:
    words = word_tokenize(post.title.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    titles.extend(words)

# Count the most frequent words
word_counts = Counter(titles)
most_common_words = word_counts.most_common(5)

# Group posts by the most common words (topics)
topics = defaultdict(list)
for post in top_posts:
    for word, _ in most_common_words:
        if word in post.title.lower():
            topics[word].append(post)
            break

# Print the most frequent words (topics) with the corresponding posts
for word, posts in topics.items():
    print(f"\nTopic: {word}")
    for post in posts:
        print(f" - {post.title}")
