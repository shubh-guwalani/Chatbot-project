import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_most_relevant_articles(titles, articles):
    title_embeddings = []
    for title in tqdm(titles, desc="Embedding Titles"):
        title_embeddings.append(embed_text(title, tokenizer, model))
    
    article_embeddings = []
    for article in tqdm(articles, desc="Embedding Articles"):
        article_embeddings.append(embed_text(article, tokenizer, model))
    
    relevance_scores = cosine_similarity(np.vstack(title_embeddings), np.vstack(article_embeddings))
    
    most_relevant_articles = relevance_scores.argmax(axis=1)
    return most_relevant_articles

# Path to the Excel file
excel_file_path = 'your_excel_file.xlsx'

# Read the Excel file
df = pd.read_excel(excel_file_path)

# Extract titles and articles from the DataFrame
titles = df['Title'].tolist()
articles = df['Article'].tolist()

# Find the most relevant articles for each title
most_relevant_articles_indices = find_most_relevant_articles(titles, articles)

# Create a new DataFrame to store the results
results = pd.DataFrame({
    'Title': titles,
    'Most Relevant Article': [articles[i] for i in most_relevant_articles_indices]
})

# Path to save the CSV file
csv_file_path = 'mapped_titles_to_articles.csv'

# Save the results to a CSV file
results.to_csv(csv_file_path, index=False)

print(f"Mapped titles to articles have been saved to {csv_file_path}")
