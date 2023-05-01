# Reddit Topic Analyzer

This application fetches the top 100 posts from a specified subreddit and uses Latent Dirichlet Allocation (LDA) to identify the most popular topics among these posts. The LDA model analyzes the titles of the posts and extracts 5 main topics, each represented by a group of words that signify the main themes or subjects of the topic.

## Requirements

- Python 3.6+
- `praw` library for accessing the Reddit API
- `pandas` library for handling data
- `nltk` library for natural language processing tasks
- `gensim` library for topic modeling

## Installation

1. Clone this repository or download the source code.
2. Install the required libraries by running: `pip install praw pandas nltk gensim`
3. Set up a Reddit API application and replace the placeholders in the script with your Reddit API credentials (client ID, client secret, and user agent).

## Usage

1. Replace `your_subreddit_name` in the script with the name of the subreddit you want to analyze.
2. Run the script: `python reddit_topic_analyzer.py`
3. The script will fetch the top 100 posts from the specified subreddit and extract the most popular topics using the LDA model.
4. The output will display the top 5 topics, each represented by a group of words with their corresponding weights, indicating the importance of each word within the topic.

## Customization

You can customize the application by changing the following parameters:

- `num_topics`: Change the number of topics you want the LDA model to find.
- `passes`: Change the number of passes the LDA model goes through the dataset to update the topic-word and document-topic distributions. More passes can lead to a better model but will take longer to train.
- `workers`: Change the number of parallel processes to use for training the LDA model. This speeds up the process on multi-core machines.

## Limitations

The LDA model is an unsupervised learning method, and the results may vary each time you run the script. You may need to tweak the number of topics, passes, or other parameters to get better and more meaningful results.
