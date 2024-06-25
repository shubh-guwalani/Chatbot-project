import pandas as pd
from tqdm import tqdm
import numpy as np

# Example DataFrame and keyphrase extraction model (replace with your actual model and data)
data = [...]  # Your list of documents
df_sample = pd.DataFrame(data, columns=['document'])
vectorizer = ...  # Your vectorizer
kw_model = ...  # Your keyword extraction model

# Parameters
batch_size = 10  # Adjust batch size as needed

# Split data into batches
batches = np.array_split(data, np.ceil(len(data) / batch_size))

# Initialize list to store keyphrases
all_keyphrases = []

# Process each batch with progress bar
for batch in tqdm(batches, desc="Processing Batches"):
    keyphrases_batch = kw_model.extract_keywords(docs=batch, vectorizer=vectorizer)
    all_keyphrases.extend(keyphrases_batch)

# Update DataFrame with keyphrases
df_sample['keyphrases'] = all_keyphrases

# Check the results
print(df_sample.head())
