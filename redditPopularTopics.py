import praw
import pandas as pd
import spacy
from collections import Counter
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

def main():
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Replace the placeholders with your own API credentials
    client_id = "dEsJH7mLkCq9Hw575aHANw"
    client_secret = "0v-78wQMQDL3YcJ1qNHPhcDtA2wSEQ"
    user_agent = "python:my_reddit_scraper:v1.0 (by /u/Least-Result-45)"

    # Authenticate with the Reddit API
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    # Define the subreddit you want to scrape
    subreddit_name = "csMajors"
    subreddit = reddit.subreddit(subreddit_name)


    # controversial_posts = subreddit.controversial(limit=100)

    # Fetch the top 100 posts from the subreddit
    top_posts = subreddit.top(limit=100)

    # Extract relevant data from the posts
    data = []

    for post in top_posts:
        post_data = {
            "title": post.title,
            "score": post.score,
            "id": post.id,
            "url": post.url,
            "created": post.created_utc,
            "num_comments": post.num_comments,
            "flair": post.link_flair_text
        }
        data.append(post_data)


    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Preprocess the text data
    def preprocess(text):
        result = []
        stop_words = set(stopwords.words('english'))
        tokens = gensim.utils.simple_preprocess(text)
        for token in tokens:
            if token not in stop_words:
                result.append(token)
        return result

    # Preprocess the titles
    df['preprocessed_titles'] = df['title'].apply(preprocess)

    # Create the Dictionary and Corpus needed for the LDA model
    dictionary = corpora.Dictionary(df['preprocessed_titles'])
    corpus = [dictionary.doc2bow(title) for title in df['preprocessed_titles']]

    # Build the LDA model
    num_topics = 5
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)

    # Print the top 5 topics
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx + 1} \nWords: {topic}\n")

if __name__ == '__main__':
    main()