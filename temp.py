from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_most_relevant_articles(titles, articles):
    title_embeddings = [embed_text(title, tokenizer, model) for title in titles]
    article_embeddings = [embed_text(article, tokenizer, model) for article in articles]
    
    relevance_scores = cosine_similarity(np.vstack(title_embeddings), np.vstack(article_embeddings))
    
    most_relevant_articles = relevance_scores.argmax(axis=1)
    return most_relevant_articles

# Example lists
titles = ["Example title 1", "Example title 2", "Example title 3"]
articles = ["This is the content of the first article.", "This is the second article content.", "Content of the third article."]

most_relevant_articles_indices = find_most_relevant_articles(titles, articles)
for i, index in enumerate(most_relevant_articles_indices):
    print(f"Title: {titles[i]}\nMost Relevant Article: {articles[index]}\n")
