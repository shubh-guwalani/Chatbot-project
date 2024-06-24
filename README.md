import pandas as pd

# Load your dataset
df = pd.read_csv('reddit_posts.csv')

# Ensure the creation_date is in datetime format
df['creation_date'] = pd.to_datetime(df['creation_date'])


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download stopwords and other necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)


df['day'] = df['creation_date'].dt.date
df['week'] = df['creation_date'].dt.to_period('W')
df['month'] = df['creation_date'].dt.to_period('M')

daily_groups = df.groupby('day')['processed_text'].apply(list)
weekly_groups = df.groupby('week')['processed_text'].apply(list)
monthly_groups = df.groupby('month')['processed_text'].apply(list)


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def extract_top_terms(texts, top_n=10):
    combined_texts = [' '.join(text) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_texts)
    terms = vectorizer.get_feature_names_out()
    sums = X.sum(axis=0)
    term_freq = [(terms[i], sums[0, i]) for i in range(sums.shape[1])]
    term_freq = sorted(term_freq, key=lambda x: x[1], reverse=True)
    return term_freq[:top_n]

daily_topics = daily_groups.apply(extract_top_terms)
weekly_topics = weekly_groups.apply(extract_top_terms)
monthly_topics = monthly_groups.apply(extract_top_terms)



# Example of accessing top topics
print(daily_topics)
print(weekly_topics)
print(monthly_topics)
