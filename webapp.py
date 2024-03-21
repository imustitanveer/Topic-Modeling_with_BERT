import streamlit as st
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
import numpy as np

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('model_epoch_7/')
model = BertForSequenceClassification.from_pretrained('model_epoch_7/')
model.eval()

# Load the LDA model
vectorizer = CountVectorizer(stop_words='english')
lda = LatentDirichletAllocation(n_components=2, random_state=42)

# Load the Word2Vec model
word2vec_model = Word2Vec.load('word2vec model/word2vec.model')

# Define the mapping from numeric labels to topic names
label_to_topic = {
    0: 'Politics',
    1: 'Sport',
    2: 'Technology',
    3: 'Entertainment',
    4: 'Business'
}


# Define the function to classify tweets
def classify_tweet(tweet, model, tokenizer):
    inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    predicted_topic = label_to_topic[predicted_label]
    return predicted_topic



# Define the function to get BERT embeddings for a tweet
def get_bert_embeddings(tweet, model, tokenizer):
    inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.logits.squeeze(0).detach().numpy()  # Use logits for classification tasks
    return embeddings

    

def calculate_similarity(embeddings):
    try:
        if isinstance(embeddings[0], np.ndarray):
            # Assuming BERT-style embeddings
            embeddings = np.array(embeddings)
            return np.mean(cosine_similarity(embeddings))
        elif isinstance(embeddings[0], list):
            # Assuming LDA-style embeddings
            flattened_embeddings = [emb for topic in embeddings for emb in topic]
            flattened_embeddings = np.array(flattened_embeddings)
            return np.mean(cosine_similarity(flattened_embeddings))
    except IndexError:
        return 0.0
    

# Define the function to extract Word2Vec embeddings for a tweet
def get_embeddings(user_input, word2vec_model):
    words = user_input.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None

# Title of the app
st.title("""
         #Tweet Topic Identifier!

        - Find a tweet and paste it below
        - Our BERT and LDA models will classify its topic
        - Semantic Similarity available for comparison!

         """)

# Text input box for user input
user_input = st.text_area('Enter your tweet here:', '')

# Button to classify the tweet
if st.button('Identify'):
    # Classify the tweet using the BERT model
    predicted_topic = classify_tweet(user_input, model, tokenizer)
    
    # Get BERT embeddings for the tweet
    tweet_embeddings = get_bert_embeddings(user_input, model, tokenizer)
    # Calculate cosine similarity with a sample embedding
    bert_cosine_sim = calculate_similarity([tweet_embeddings])
    
    # Display the BERT predicted topic and semantic similarity
    st.write(f'BERT Predicted Topic: The tweet looks like it falls into the {predicted_topic} category')
    st.write(f'Semantic Similarity with BERT: {bert_cosine_sim}')