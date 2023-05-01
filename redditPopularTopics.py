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
    subreddit_name = "cscareerquestions"
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

    def generate_topic_names(lda_model, num_topics):
        topic_names = []
        for i in range(num_topics):
            topic_terms = lda_model.show_topic(i, topn=2)  # Get the top 2 words for each topic
            topic_name = " / ".join([term[0] for term in topic_terms])
            topic_names.append(topic_name)
        return topic_names

        
    # Preprocess the titles
    df['preprocessed_titles'] = df['title'].apply(preprocess)

    # Create the Dictionary and Corpus needed for the LDA model
    dictionary = corpora.Dictionary(df['preprocessed_titles'])
    corpus = [dictionary.doc2bow(title) for title in df['preprocessed_titles']]

    # Build the LDA model
    num_topics = 10

    # Run LDA model and generate topic names
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)
    topic_names = generate_topic_names(lda_model, num_topics)

    # Print topics with their names
    for i, topic in enumerate(lda_model.print_topics(num_words=5)):
        print(f"Topic {i + 1} ({topic_names[i]}): {topic}")
if __name__ == '__main__':
    main()